#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

POSE_POINTS = 25
HAND_POINTS = 21
FACE_POINTS = 70

TOKEN_RE = re.compile(r"[a-z0-9']+|[.,!?;:-]")


@dataclass(frozen=True)
class SampleRecord:
    key: str
    text: str
    keypoint_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    if not text:
        return []
    tokens = TOKEN_RE.findall(text)
    if tokens:
        return tokens
    return text.split()


class Vocab:
    def __init__(self, stoi: dict[str, int], itos: list[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(cls, texts: list[str], min_freq: int) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        itos = list(specials)
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in specials:
                itos.append(token)

        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str, max_len: int) -> list[int]:
        token_ids = [self.stoi.get(tok, self.unk_id) for tok in tokenize(text)]
        if max_len > 0:
            token_ids = token_ids[:max_len]
        if not token_ids:
            token_ids = [self.unk_id]
        return token_ids

    def decode(self, token_ids: list[int], stop_at_eos: bool = True) -> str:
        out: list[str] = []
        for idx in token_ids:
            tok = self.itos[idx] if 0 <= idx < len(self.itos) else UNK_TOKEN
            if tok in {PAD_TOKEN, BOS_TOKEN}:
                continue
            if stop_at_eos and tok == EOS_TOKEN:
                break
            out.append(tok)
        return " ".join(out).strip()

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        itos = list(data["itos"])
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


def keypoint_feature_dim(use_face: bool) -> int:
    point_count = POSE_POINTS + HAND_POINTS + HAND_POINTS
    if use_face:
        point_count += FACE_POINTS
    return point_count * 3


def parse_keypoint_block(values: list[float], n_points: int) -> torch.Tensor:
    out = torch.zeros((n_points, 3), dtype=torch.float32)
    if not values:
        return out

    usable = min(len(values) // 3, n_points)
    if usable <= 0:
        return out

    block = torch.tensor(values[: usable * 3], dtype=torch.float32).view(usable, 3)
    out[:usable] = block
    return out


def person_confidence_score(person: dict) -> float:
    keys = [
        "pose_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
        "face_keypoints_2d",
    ]
    score = 0.0
    for key in keys:
        values = person.get(key, [])
        if values:
            score += float(sum(values[2::3]))
    return score


def frame_features_from_json(json_path: Path, use_face: bool) -> torch.Tensor:
    expected_dim = keypoint_feature_dim(use_face)
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    people = payload.get("people", [])
    if not people:
        return torch.zeros(expected_dim, dtype=torch.float32)

    person = max(people, key=person_confidence_score)
    parts = [
        parse_keypoint_block(person.get("pose_keypoints_2d", []), POSE_POINTS),
        parse_keypoint_block(person.get("hand_left_keypoints_2d", []), HAND_POINTS),
        parse_keypoint_block(person.get("hand_right_keypoints_2d", []), HAND_POINTS),
    ]
    if use_face:
        parts.append(parse_keypoint_block(person.get("face_keypoints_2d", []), FACE_POINTS))

    points = torch.cat(parts, dim=0)
    xy = points[:, :2]
    conf = points[:, 2]
    valid = conf > 0

    if valid.any():
        valid_xy = xy[valid]
        mean = valid_xy.mean(dim=0, keepdim=True)
        std = valid_xy.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        xy = (xy - mean) / std
        xy[~valid] = 0.0
    else:
        xy.zero_()

    normed = torch.cat([xy, conf.unsqueeze(1)], dim=1)
    flat = normed.flatten()
    if flat.numel() != expected_dim:
        out = torch.zeros(expected_dim, dtype=torch.float32)
        usable = min(expected_dim, flat.numel())
        out[:usable] = flat[:usable]
        return out
    return flat


def sample_frame_paths(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frame_paths) <= max_frames:
        return frame_paths

    idx = torch.linspace(0, len(frame_paths) - 1, steps=max_frames)
    chosen = idx.round().long().tolist()
    return [frame_paths[i] for i in chosen]


def load_clip_features(clip_dir: Path, max_frames: int, use_face: bool) -> torch.Tensor:
    frame_paths = sorted(clip_dir.glob("*_keypoints.json"))
    frame_paths = sample_frame_paths(frame_paths, max_frames=max_frames)

    expected_dim = keypoint_feature_dim(use_face)
    if not frame_paths:
        return torch.zeros((1, expected_dim), dtype=torch.float32)

    frames: list[torch.Tensor] = []
    for frame_path in frame_paths:
        try:
            features = frame_features_from_json(frame_path, use_face=use_face)
        except (json.JSONDecodeError, OSError, ValueError):
            features = torch.zeros(expected_dim, dtype=torch.float32)
        frames.append(features)

    return torch.stack(frames, dim=0)


def read_split_records(tsv_path: Path, keypoint_root: Path, max_samples: int = 0) -> tuple[list[SampleRecord], int]:
    records: list[SampleRecord] = []
    skipped = 0

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {tsv_path}")
        required = {"SENTENCE_NAME", "SENTENCE"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"TSV missing columns {sorted(missing)}: {tsv_path}")

        for row in reader:
            key = (row.get("SENTENCE_NAME") or "").strip()
            text = (row.get("SENTENCE") or "").strip()
            if not key or not text:
                skipped += 1
                continue

            clip_dir = keypoint_root / key
            if not clip_dir.exists() or not clip_dir.is_dir():
                skipped += 1
                continue

            if not any(clip_dir.glob("*_keypoints.json")):
                skipped += 1
                continue

            records.append(SampleRecord(key=key, text=text, keypoint_dir=clip_dir))
            if max_samples > 0 and len(records) >= max_samples:
                break

    return records, skipped


class How2SignKeypointDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        vocab: Vocab,
        max_frames: int,
        max_target_tokens: int,
        use_face: bool,
        cache_size: int,
    ) -> None:
        self.records = records
        self.vocab = vocab
        self.max_frames = max_frames
        self.max_target_tokens = max_target_tokens
        self.use_face = use_face
        self.cache_size = max(0, cache_size)
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _cache_get(self, key: str) -> torch.Tensor | None:
        val = self.cache.get(key)
        if val is not None:
            self.cache.move_to_end(key)
        return val

    def _cache_put(self, key: str, val: torch.Tensor) -> None:
        if self.cache_size <= 0:
            return
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[key] = val

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int], str]:
        record = self.records[idx]
        cached = self._cache_get(record.key)
        if cached is None:
            features = load_clip_features(record.keypoint_dir, max_frames=self.max_frames, use_face=self.use_face)
            self._cache_put(record.key, features)
        else:
            features = cached

        token_ids = self.vocab.encode(record.text, max_len=self.max_target_tokens)
        return features, token_ids, record.key


class BatchCollator:
    def __init__(self, pad_id: int, bos_id: int, eos_id: int) -> None:
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, batch: list[tuple[torch.Tensor, list[int], str]]) -> dict[str, torch.Tensor | list[str]]:
        src_list, token_list, keys = zip(*batch)

        batch_size = len(src_list)
        max_src_len = max(src.size(0) for src in src_list)
        feature_dim = src_list[0].size(1)

        src = torch.zeros((batch_size, max_src_len, feature_dim), dtype=torch.float32)
        src_pad_mask = torch.ones((batch_size, max_src_len), dtype=torch.bool)
        for i, seq in enumerate(src_list):
            length = seq.size(0)
            src[i, :length] = seq
            src_pad_mask[i, :length] = False

        tgt_in_seqs: list[list[int]] = []
        tgt_out_seqs: list[list[int]] = []
        for token_ids in token_list:
            tgt_in = [self.bos_id] + token_ids
            tgt_out = token_ids + [self.eos_id]
            tgt_in_seqs.append(tgt_in)
            tgt_out_seqs.append(tgt_out)

        max_tgt_len = max(len(seq) for seq in tgt_in_seqs)
        tgt_in = torch.full((batch_size, max_tgt_len), self.pad_id, dtype=torch.long)
        tgt_out = torch.full((batch_size, max_tgt_len), self.pad_id, dtype=torch.long)
        tgt_pad_mask = torch.ones((batch_size, max_tgt_len), dtype=torch.bool)
        for i, (seq_in, seq_out) in enumerate(zip(tgt_in_seqs, tgt_out_seqs)):
            length = len(seq_in)
            tgt_in[i, :length] = torch.tensor(seq_in, dtype=torch.long)
            tgt_out[i, :length] = torch.tensor(seq_out, dtype=torch.long)
            tgt_pad_mask[i, :length] = False

        return {
            "src": src,
            "src_pad_mask": src_pad_mask,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "tgt_pad_mask": tgt_pad_mask,
            "keys": list(keys),
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros((max_len, d_model), dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SignToTextTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.src_proj = nn.Linear(input_dim, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.src_pos = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.tgt_pos = PositionalEncoding(d_model=d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_proj(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos(src_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_emb(tgt_in) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos(tgt_emb)
        tgt_mask = self._causal_mask(tgt_in.size(1), device=tgt_in.device)

        hidden = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out_proj(hidden)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src=src, src_pad_mask=src_pad_mask)
        return self.decode(
            tgt_in=tgt_in,
            memory=memory,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
        )


@torch.no_grad()
def greedy_decode(
    model: SignToTextTransformer,
    src: torch.Tensor,
    src_pad_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
) -> list[int]:
    model.eval()
    memory = model.encode(src=src, src_pad_mask=src_pad_mask)

    generated = torch.full((1, 1), bos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len):
        tgt_pad_mask = generated.eq(pad_id)
        logits = model.decode(
            tgt_in=generated,
            memory=memory,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
        )
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())
        generated = torch.cat(
            [generated, torch.tensor([[next_token]], dtype=torch.long, device=src.device)],
            dim=1,
        )
        if next_token == eos_id:
            break

    return generated.squeeze(0).tolist()


def move_to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    moved: dict[str, torch.Tensor | list[str]] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def train_one_epoch(
    model: SignToTextTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    grad_clip: float,
    amp: bool,
    log_every: int,
) -> tuple[float, float]:
    model.train()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    step_count = 0

    for step_count, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        src = batch["src"]
        src_pad_mask = batch["src_pad_mask"]
        tgt_in = batch["tgt_in"]
        tgt_out = batch["tgt_out"]
        tgt_pad_mask = batch["tgt_pad_mask"]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(src=src, tgt_in=tgt_in, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=pad_id,
            )

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = tgt_out.ne(pad_id)
            total_correct += int((pred.eq(tgt_out) & mask).sum().item())
            total_tokens += int(mask.sum().item())

        total_loss += float(loss.item())
        if log_every > 0 and step_count % log_every == 0:
            print(f"  step {step_count:5d} | loss {loss.item():.4f}")

    avg_loss = total_loss / max(step_count, 1)
    token_acc = total_correct / max(total_tokens, 1)
    return avg_loss, token_acc


@torch.no_grad()
def validate(
    model: SignToTextTransformer,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    step_count = 0

    for step_count, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        src = batch["src"]
        src_pad_mask = batch["src_pad_mask"]
        tgt_in = batch["tgt_in"]
        tgt_out = batch["tgt_out"]
        tgt_pad_mask = batch["tgt_pad_mask"]

        logits = model(src=src, tgt_in=tgt_in, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_id,
        )

        pred = logits.argmax(dim=-1)
        mask = tgt_out.ne(pad_id)
        total_correct += int((pred.eq(tgt_out) & mask).sum().item())
        total_tokens += int(mask.sum().item())
        total_loss += float(loss.item())

    avg_loss = total_loss / max(step_count, 1)
    token_acc = total_correct / max(total_tokens, 1)
    return avg_loss, token_acc


def maybe_print_predictions(
    model: SignToTextTransformer,
    dataset: How2SignKeypointDataset,
    vocab: Vocab,
    device: torch.device,
    count: int,
) -> None:
    if count <= 0 or len(dataset) == 0:
        return

    print("Sample predictions:")
    for i in range(min(count, len(dataset))):
        src, token_ids, key = dataset[i]
        src_batch = src.unsqueeze(0).to(device)
        src_pad_mask = torch.zeros((1, src.size(0)), dtype=torch.bool, device=device)
        pred_ids = greedy_decode(
            model=model,
            src=src_batch,
            src_pad_mask=src_pad_mask,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id,
            max_len=max(len(token_ids) + 10, 20),
        )

        target = vocab.decode(token_ids + [vocab.eos_id])
        pred = vocab.decode(pred_ids)
        print(f"  key={key}")
        print(f"    target: {target}")
        print(f"    pred  : {pred}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a How2Sign keypoint-to-text Transformer model")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Path to dataset root")
    parser.add_argument("--train-tsv", type=str, default="how2sign_train.csv")
    parser.add_argument("--val-tsv", type=str, default="how2sign_val.csv")
    parser.add_argument("--train-keypoints", type=str, default="train_2D_keypoints/openpose_output/json")
    parser.add_argument("--val-keypoints", type=str, default="val_2D_keypoints/openpose_output/json")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--max-target-tokens", type=int, default=64)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--use-face", action="store_true", help="Include face keypoints in input features")
    parser.add_argument("--cache-size", type=int, default=256, help="Clip feature cache size per dataset")

    parser.add_argument("--max-train-samples", type=int, default=0, help="For quick experiments")
    parser.add_argument("--max-val-samples", type=int, default=0, help="For quick experiments")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")

    parser.add_argument("--save-dir", type=Path, default=Path("runs/how2sign_keypoint2text"))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--print-samples", type=int, default=2, help="Show sample predictions each epoch")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = args.data_root.resolve()
    train_tsv = root / args.train_tsv
    val_tsv = root / args.val_tsv
    train_kp_root = root / args.train_keypoints
    val_kp_root = root / args.val_keypoints

    for required in [train_tsv, val_tsv, train_kp_root, val_kp_root]:
        if not required.exists():
            raise FileNotFoundError(f"Required path missing: {required}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    print(f"Device: {device}")
    print("Loading split metadata...")
    train_records, skipped_train = read_split_records(
        tsv_path=train_tsv,
        keypoint_root=train_kp_root,
        max_samples=args.max_train_samples,
    )
    val_records, skipped_val = read_split_records(
        tsv_path=val_tsv,
        keypoint_root=val_kp_root,
        max_samples=args.max_val_samples,
    )

    if not train_records:
        raise RuntimeError("No usable training samples found after filtering missing keypoint folders")
    if not val_records:
        raise RuntimeError("No usable validation samples found after filtering missing keypoint folders")

    print(
        f"Train samples: {len(train_records)} (skipped {skipped_train}) | "
        f"Val samples: {len(val_records)} (skipped {skipped_val})"
    )

    print("Building vocabulary...")
    vocab = Vocab.build([r.text for r in train_records], min_freq=args.min_token_freq)
    print(f"Vocab size: {len(vocab.itos)}")

    train_dataset = How2SignKeypointDataset(
        records=train_records,
        vocab=vocab,
        max_frames=args.max_frames,
        max_target_tokens=args.max_target_tokens,
        use_face=args.use_face,
        cache_size=args.cache_size,
    )
    val_dataset = How2SignKeypointDataset(
        records=val_records,
        vocab=vocab,
        max_frames=args.max_frames,
        max_target_tokens=args.max_target_tokens,
        use_face=args.use_face,
        cache_size=args.cache_size,
    )

    collate = BatchCollator(pad_id=vocab.pad_id, bos_id=vocab.bos_id, eos_id=vocab.eos_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    input_dim = keypoint_feature_dim(args.use_face)
    model = SignToTextTransformer(
        input_dim=input_dim,
        vocab_size=len(vocab.itos),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "metrics.csv"

    with (save_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, ensure_ascii=True, indent=2)

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_token_acc", "val_loss", "val_token_acc", "seconds"])

    best_val_loss = float("inf")
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=vocab.pad_id,
            grad_clip=args.grad_clip,
            amp=amp_enabled,
            log_every=args.log_every,
        )
        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            device=device,
            pad_id=vocab.pad_id,
        )

        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_tok_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_tok_acc={val_acc:.4f} | time={elapsed:.1f}s"
        )

        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, elapsed])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "vocab": vocab.to_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  Saved new best checkpoint to: {save_dir / 'best.pt'}")

        maybe_print_predictions(
            model=model,
            dataset=val_dataset,
            vocab=vocab,
            device=device,
            count=args.print_samples,
        )

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved in: {save_dir}")


if __name__ == "__main__":
    main()