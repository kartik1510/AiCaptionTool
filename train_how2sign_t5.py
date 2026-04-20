#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

try:
    from nltk.translate.meteor_score import single_meteor_score  # type: ignore
except Exception:
    single_meteor_score = None


POSE_POINTS = 25
HAND_POINTS = 21
FACE_POINTS = 70

POSE_LEFT_RIGHT_SWAP = [
    (2, 5),
    (3, 6),
    (4, 7),
    (9, 12),
    (10, 13),
    (11, 14),
    (15, 16),
    (17, 18),
    (19, 22),
    (20, 23),
    (21, 24),
]

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 2
REDUCE_RE = re.compile(r"[^a-z0-9' ]+")


@dataclass(frozen=True)
class SampleRecord:
    key: str
    text: str
    keypoint_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_joints(use_face: bool) -> int:
    joints = POSE_POINTS + HAND_POINTS + HAND_POINTS
    if use_face:
        joints += FACE_POINTS
    return joints


def feature_dim(use_face: bool) -> int:
    # Per joint channels: x_norm, y_norm, confidence, missing_flag.
    return num_joints(use_face) * 4


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


def person_confidence(person: dict) -> float:
    score = 0.0
    for key in [
        "pose_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
        "face_keypoints_2d",
    ]:
        values = person.get(key, [])
        if values:
            score += float(sum(values[2::3]))
    return score


def read_frame_points(json_path: Path, use_face: bool) -> torch.Tensor:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    joints = num_joints(use_face)
    zero = torch.zeros((joints, 3), dtype=torch.float32)
    people = payload.get("people", [])
    if not people:
        return zero

    person = max(people, key=person_confidence)
    parts = [
        parse_keypoint_block(person.get("pose_keypoints_2d", []), POSE_POINTS),
        parse_keypoint_block(person.get("hand_left_keypoints_2d", []), HAND_POINTS),
        parse_keypoint_block(person.get("hand_right_keypoints_2d", []), HAND_POINTS),
    ]
    if use_face:
        parts.append(parse_keypoint_block(person.get("face_keypoints_2d", []), FACE_POINTS))
    return torch.cat(parts, dim=0)


def interpolate_missing(points: torch.Tensor, max_gap: int, min_conf: float) -> torch.Tensor:
    # points: [T, J, 3]
    if points.size(0) <= 2 or max_gap <= 0:
        return points

    out = points.clone()
    conf = out[:, :, 2]

    for j in range(out.size(1)):
        valid_idx = torch.where(conf[:, j] > min_conf)[0].tolist()
        if len(valid_idx) < 2:
            continue

        for left, right in zip(valid_idx[:-1], valid_idx[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue

            left_xy = out[left, j, :2]
            right_xy = out[right, j, :2]
            left_c = out[left, j, 2]
            right_c = out[right, j, 2]

            for g in range(1, gap + 1):
                alpha = g / (gap + 1)
                out[left + g, j, :2] = (1.0 - alpha) * left_xy + alpha * right_xy
                out[left + g, j, 2] = (1.0 - alpha) * left_c + alpha * right_c

    return out


def signing_space_center_scale(points: torch.Tensor, min_conf: float) -> tuple[torch.Tensor, torch.Tensor]:
    # points: [T, J, 3]
    centers: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []

    for t in range(points.size(0)):
        frame = points[t]
        ls = frame[LEFT_SHOULDER]
        rs = frame[RIGHT_SHOULDER]

        if float(ls[2]) > min_conf and float(rs[2]) > min_conf:
            center = (ls[:2] + rs[:2]) / 2.0
            scale = torch.norm(ls[:2] - rs[:2], p=2)
            if float(scale) > 1e-6:
                centers.append(center)
                scales.append(scale)
                continue

        valid = frame[:, 2] > min_conf
        if int(valid.sum()) >= 2:
            xy = frame[valid, :2]
            center = xy.mean(dim=0)
            bbox = xy.max(dim=0).values - xy.min(dim=0).values
            scale = torch.norm(bbox, p=2)
            if float(scale) > 1e-6:
                centers.append(center)
                scales.append(scale)

    center = torch.stack(centers, dim=0).median(dim=0).values if centers else torch.zeros(2, dtype=torch.float32)
    scale = torch.stack(scales, dim=0).median().clamp_min(1e-3) if scales else torch.tensor(1.0, dtype=torch.float32)
    return center, scale


def normalize_points(points: torch.Tensor, min_conf: float) -> torch.Tensor:
    out = points.clone()
    center, scale = signing_space_center_scale(out, min_conf=min_conf)
    out[:, :, :2] = (out[:, :, :2] - center.view(1, 1, 2)) / scale

    invalid = out[:, :, 2] <= min_conf
    out[:, :, :2][invalid] = 0.0
    out[:, :, :2] = out[:, :, :2].clamp(min=-3.0, max=3.0)
    return out


def temporal_indices(frame_count: int, max_frames: int, stochastic: bool) -> list[int]:
    if max_frames <= 0 or frame_count <= max_frames:
        return list(range(frame_count))

    if stochastic:
        edges = torch.linspace(0, frame_count, steps=max_frames + 1)
        chosen: list[int] = []
        for i in range(max_frames):
            left = int(edges[i].item())
            right = int(edges[i + 1].item()) - 1
            if right < left:
                right = left
            chosen.append(random.randint(left, right))
        return chosen

    lin = torch.linspace(0, frame_count - 1, steps=max_frames)
    return lin.round().long().tolist()


def maybe_augment(points: torch.Tensor, use_face: bool, flip_prob: float, scale_jitter: float) -> torch.Tensor:
    out = points.clone()

    if random.random() < flip_prob:
        out[:, :, 0] = -out[:, :, 0]

        for a, b in POSE_LEFT_RIGHT_SWAP:
            tmp = out[:, a, :].clone()
            out[:, a, :] = out[:, b, :]
            out[:, b, :] = tmp

        lh_start = POSE_POINTS
        rh_start = POSE_POINTS + HAND_POINTS
        lh = out[:, lh_start : lh_start + HAND_POINTS, :].clone()
        rh = out[:, rh_start : rh_start + HAND_POINTS, :].clone()
        out[:, lh_start : lh_start + HAND_POINTS, :] = rh
        out[:, rh_start : rh_start + HAND_POINTS, :] = lh

        if use_face:
            import warnings
            warnings.warn(
                "Horizontal flip augmentation with use_face=True mirrors face x-coords "
                "but does NOT swap face landmark indices (left-eye to right-eye etc.). "
                "Disable flip_prob or use_face until a full face swap map is added.",
                stacklevel=2,
            )

    if scale_jitter > 0:
        factor = random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        out[:, :, :2] = out[:, :, :2] * factor

    return out


def clip_to_features(
    clip_dir: Path,
    max_frames: int,
    use_face: bool,
    min_conf: float,
    interpolation_gap: int,
    stochastic_sampling: bool,
    do_augment: bool,
    flip_prob: float,
    scale_jitter: float,
) -> torch.Tensor:
    frame_paths = sorted(clip_dir.glob("*_keypoints.json"))
    if not frame_paths:
        return torch.zeros((1, feature_dim(use_face)), dtype=torch.float32)

    keep = temporal_indices(len(frame_paths), max_frames=max_frames, stochastic=stochastic_sampling)
    selected = [frame_paths[i] for i in keep]

    frame_tensors: list[torch.Tensor] = []
    for path in selected:
        try:
            frame = read_frame_points(path, use_face=use_face)
        except (json.JSONDecodeError, OSError, ValueError):
            frame = torch.zeros((num_joints(use_face), 3), dtype=torch.float32)
        frame_tensors.append(frame)

    points = torch.stack(frame_tensors, dim=0)
    points = interpolate_missing(points, max_gap=interpolation_gap, min_conf=min_conf)
    points = normalize_points(points, min_conf=min_conf)

    if do_augment:
        points = maybe_augment(
            points,
            use_face=use_face,
            flip_prob=flip_prob,
            scale_jitter=scale_jitter,
        )

    missing = (points[:, :, 2] <= min_conf).float().unsqueeze(-1)
    feat = torch.cat([points[:, :, :2], points[:, :, 2:3], missing], dim=-1)
    return feat.flatten(start_dim=1)


def gap_hist_for_clip(clip_dir: Path, use_face: bool, min_conf: float) -> Counter[int]:
    frame_paths = sorted(clip_dir.glob("*_keypoints.json"))
    if not frame_paths:
        return Counter()

    frame_tensors: list[torch.Tensor] = []
    for path in frame_paths:
        try:
            frame_tensors.append(read_frame_points(path, use_face=use_face))
        except (json.JSONDecodeError, OSError, ValueError):
            frame_tensors.append(torch.zeros((num_joints(use_face), 3), dtype=torch.float32))

    points = torch.stack(frame_tensors, dim=0)
    missing = points[:, :, 2] <= min_conf  # [T, J]

    hist: Counter[int] = Counter()
    for j in range(missing.size(1)):
        start = None
        for t in range(missing.size(0)):
            is_missing = bool(missing[t, j].item())
            if is_missing and start is None:
                start = t
            elif not is_missing and start is not None:
                hist[t - start] += 1
                start = None
        if start is not None:
            hist[missing.size(0) - start] += 1
    return hist


def collect_gap_audit(
    records: list[SampleRecord],
    use_face: bool,
    min_conf: float,
    max_clips: int,
    show_progress: bool,
) -> dict:
    take = records if max_clips <= 0 else records[:max_clips]
    agg: Counter[int] = Counter()

    gap_iter = tqdm(
        take,
        total=len(take),
        desc="GapAudit",
        leave=False,
        disable=not show_progress,
    )
    for rec in gap_iter:
        agg.update(gap_hist_for_clip(rec.keypoint_dir, use_face=use_face, min_conf=min_conf))

    total_gaps = sum(agg.values())
    le3 = sum(v for k, v in agg.items() if k <= 3)
    gt3 = total_gaps - le3

    hist_sorted = {str(k): int(v) for k, v in sorted(agg.items(), key=lambda x: x[0])}
    return {
        "clips_scanned": len(take),
        "clips_total": len(records),
        "total_gaps": total_gaps,
        "gaps_leq_3": le3,
        "gaps_gt_3": gt3,
        "ratio_gaps_leq_3": (le3 / max(total_gaps, 1)),
        "histogram": hist_sorted,
    }


def reduce_text(text: str) -> str:
    low = text.lower().strip()
    low = REDUCE_RE.sub(" ", low)
    return " ".join(low.split())


def tokenize_words(text: str) -> list[str]:
    return [w for w in reduce_text(text).split(" ") if w]


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pt = tokenize_words(pred)
    rt = tokenize_words(ref)
    if not pt or not rt:
        return 0.0
    lcs = lcs_len(pt, rt)
    p = lcs / max(len(pt), 1)
    r = lcs / max(len(rt), 1)
    if p + r <= 0:
        return 0.0
    return (2.0 * p * r) / (p + r)


def meteor_fallback(pred: str, ref: str) -> float:
    pt = tokenize_words(pred)
    rt = tokenize_words(ref)
    if not pt or not rt:
        return 0.0

    ref_counts = Counter(rt)
    match = 0
    for tok in pt:
        if ref_counts[tok] > 0:
            match += 1
            ref_counts[tok] -= 1

    if match == 0:
        return 0.0
    precision = match / max(len(pt), 1)
    recall = match / max(len(rt), 1)
    return (10.0 * precision * recall) / max(recall + 9.0 * precision, 1e-9)


def corpus_meteor(preds: list[str], refs: list[str]) -> float:
    if not preds:
        return 0.0
    scores: list[float] = []

    for pred, ref in zip(preds, refs):
        if single_meteor_score is not None:
            try:
                scores.append(float(single_meteor_score(tokenize_words(ref), tokenize_words(pred))))
                continue
            except Exception:
                pass
        scores.append(meteor_fallback(pred, ref))

    return (sum(scores) / max(len(scores), 1)) * 100.0


def evaluate_text_metrics(preds: list[str], refs: list[str]) -> dict[str, float]:
    if not preds:
        return {
            "bleu": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "reduced_bleu": 0.0,
            "meteor": 0.0,
            "rouge_l": 0.0,
        }

    bleu = float(sacrebleu.corpus_bleu(preds, [refs]).score)
    bleu1 = float(BLEU(max_ngram_order=1, effective_order=True).corpus_score(preds, [refs]).score)
    bleu2 = float(BLEU(max_ngram_order=2, effective_order=True).corpus_score(preds, [refs]).score)
    bleu3 = float(BLEU(max_ngram_order=3, effective_order=True).corpus_score(preds, [refs]).score)
    bleu4 = float(BLEU(max_ngram_order=4, effective_order=True).corpus_score(preds, [refs]).score)

    preds_reduced = [reduce_text(x) for x in preds]
    refs_reduced = [reduce_text(x) for x in refs]
    reduced_bleu = float(sacrebleu.corpus_bleu(preds_reduced, [refs_reduced]).score)

    meteor = corpus_meteor(preds, refs)
    rouge_l = (sum(rouge_l_f1(p, r) for p, r in zip(preds, refs)) / max(len(preds), 1)) * 100.0

    return {
        "bleu": bleu,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "reduced_bleu": reduced_bleu,
        "meteor": meteor,
        "rouge_l": rouge_l,
    }


def read_split_records(
    tsv_path: Path,
    keypoint_root: Path,
    max_samples: int,
) -> tuple[list[SampleRecord], list[dict[str, str]]]:
    records: list[SampleRecord] = []
    skipped: list[dict[str, str]] = []

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

            if not key:
                skipped.append({"reason": "missing_sentence_name", "key": "", "text": text})
                continue
            if not text:
                skipped.append({"reason": "missing_sentence_text", "key": key, "text": ""})
                continue

            clip_dir = keypoint_root / key
            if not clip_dir.exists() or not clip_dir.is_dir():
                skipped.append({"reason": "missing_keypoint_dir", "key": key, "text": text})
                continue

            if not any(clip_dir.glob("*_keypoints.json")):
                skipped.append({"reason": "empty_keypoint_dir", "key": key, "text": text})
                continue

            records.append(SampleRecord(key=key, text=text, keypoint_dir=clip_dir))
            if max_samples > 0 and len(records) >= max_samples:
                break

    return records, skipped


def reason_counts(skipped: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in skipped:
        reason = row["reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def write_skip_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reason", "key", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class How2SignT5Dataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        tokenizer: T5TokenizerFast,
        max_frames: int,
        max_target_tokens: int,
        use_face: bool,
        min_conf: float,
        interpolation_gap: int,
        training: bool,
        augment: bool,
        flip_prob: float,
        scale_jitter: float,
        cache_size: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_target_tokens = max_target_tokens
        self.use_face = use_face
        self.min_conf = min_conf
        self.interpolation_gap = interpolation_gap
        self.training = training
        self.augment = augment
        self.flip_prob = flip_prob
        self.scale_jitter = scale_jitter

        # Cache only deterministic paths.
        self.cache_size = max(0, cache_size if not training else 0)
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int], str, str]:
        rec = self.records[idx]

        feat = self._cache_get(rec.key)
        if feat is None:
            feat = clip_to_features(
                clip_dir=rec.keypoint_dir,
                max_frames=self.max_frames,
                use_face=self.use_face,
                min_conf=self.min_conf,
                interpolation_gap=self.interpolation_gap,
                stochastic_sampling=self.training,
                do_augment=bool(self.training and self.augment),
                flip_prob=self.flip_prob,
                scale_jitter=self.scale_jitter,
            )
            self._cache_put(rec.key, feat)

        token_ids = self.tokenizer(
            rec.text,
            truncation=True,
            max_length=self.max_target_tokens,
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
        if not token_ids:
            token_ids = [self.tokenizer.unk_token_id]

        return feat, token_ids, rec.key, rec.text


class BatchCollator:
    def __init__(self, tokenizer: T5TokenizerFast) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[tuple[torch.Tensor, list[int], str, str]]) -> dict[str, torch.Tensor | list[str]]:
        src_list, labels_list, keys, ref_texts = zip(*batch)

        bsz = len(src_list)
        max_src = max(x.size(0) for x in src_list)
        src_dim = src_list[0].size(1)

        src = torch.zeros((bsz, max_src, src_dim), dtype=torch.float32)
        src_mask = torch.zeros((bsz, max_src), dtype=torch.long)
        for i, seq in enumerate(src_list):
            ln = seq.size(0)
            src[i, :ln] = seq
            src_mask[i, :ln] = 1

        max_tgt = max(len(x) for x in labels_list)
        labels = torch.full((bsz, max_tgt), -100, dtype=torch.long)
        for i, ids in enumerate(labels_list):
            labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {
            "src": src,
            "src_mask": src_mask,
            "labels": labels,
            "keys": list(keys),
            "ref_texts": list(ref_texts),
        }


class KeypointT5Model(nn.Module):
    def __init__(self, pretrained_name: str, input_dim: int, dropout: float, temporal_stride: int) -> None:
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_name)
        d_model = self.t5.config.d_model
        self.temporal_stride = max(1, temporal_stride)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        if self.temporal_stride > 1:
            self.temporal_conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=self.temporal_stride,
                padding=1,
            )
        else:
            self.temporal_conv = None

        self.ctc_head = nn.Linear(d_model, self.t5.config.vocab_size)

    def _encode_visual(self, src: torch.Tensor, apply_dropout: bool) -> torch.Tensor:
        embeds = self.input_proj(src)
        embeds = self.input_ln(embeds)
        if apply_dropout:
            embeds = self.input_drop(embeds)
        return embeds

    def _compress_time(self, embeds: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_conv is None:
            return embeds, mask

        # [B, T, C] -> [B, C, T]
        x = embeds.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)

        mask_f = mask.float().unsqueeze(1)
        pooled = F.max_pool1d(
            mask_f,
            kernel_size=self.temporal_stride,
            stride=self.temporal_stride,
            ceil_mode=True,
        )
        new_mask = pooled.squeeze(1).long()
        return x, new_mask

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | object]:
        visual = self._encode_visual(src, apply_dropout=self.training)
        enc_inputs, enc_mask = self._compress_time(visual, src_mask)

        out = self.t5(
            inputs_embeds=enc_inputs,
            attention_mask=enc_mask,
            labels=labels,
            return_dict=True,
        )

        encoder_h = out.encoder_last_hidden_state
        ctc_logits = self.ctc_head(encoder_h)

        return {
            "seq2seq": out,
            "visual_embeds": visual,
            "visual_mask": src_mask,
            "ctc_logits": ctc_logits,
            "ctc_mask": enc_mask,
        }

    @torch.no_grad()
    def generate(self, src: torch.Tensor, src_mask: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        visual = self._encode_visual(src, apply_dropout=False)
        enc_inputs, enc_mask = self._compress_time(visual, src_mask)
        # Merge caller kwargs with defaults; caller can override any param.
        # NOTE: num_beam_groups/diversity_penalty removed — requires trust_remote_code
        # in newer transformers and is incompatible with do_sample=True.
        # Sampling-based diversity achieves the same goal without extra deps.
        gen_kwargs = dict(
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.92,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
        )
        gen_kwargs.update(kwargs)
        return self.t5.generate(
            inputs_embeds=enc_inputs,
            attention_mask=enc_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )


def move_to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    out: dict[str, torch.Tensor | list[str]] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def compute_signcl_loss(
    visual_embeds: torch.Tensor,
    visual_mask: torch.Tensor,
    neg_distance: int,
    temperature: float,
    max_anchors: int,
    max_negatives: int,
) -> torch.Tensor:
    visual_embeds = visual_embeds.float()  # avoid FP16 saturation
    # visual_embeds: [B, T, D], visual_mask: [B, T] with 1 for valid steps.
    device = visual_embeds.device
    normed = F.normalize(visual_embeds, p=2, dim=-1)
    losses: list[torch.Tensor] = []

    bsz = normed.size(0)
    for b in range(bsz):
        valid_len = int(visual_mask[b].sum().item())
        if valid_len < 3:
            continue

        candidate = torch.arange(0, valid_len - 1, device=device)
        if max_anchors > 0 and candidate.numel() > max_anchors:
            choice = torch.randperm(candidate.numel(), device=device)[:max_anchors]
            candidate = candidate[choice]

        for a_t in candidate:
            a = int(a_t.item())
            p = a + 1

            neg_idx = torch.arange(0, valid_len, device=device)
            neg_idx = neg_idx[(neg_idx - a).abs() >= max(neg_distance, 2)]
            if neg_idx.numel() == 0:
                continue

            if max_negatives > 0 and neg_idx.numel() > max_negatives:
                pick = torch.randperm(neg_idx.numel(), device=device)[:max_negatives]
                neg_idx = neg_idx[pick]

            anchor = normed[b, a]
            positive = normed[b, p]
            negatives = normed[b, neg_idx]

            pos_logit = torch.matmul(anchor, positive).view(1) / max(temperature, 1e-6)
            neg_logits = torch.matmul(negatives, anchor) / max(temperature, 1e-6)
            logits = torch.cat([pos_logit, neg_logits], dim=0).unsqueeze(0)
            target = torch.zeros(1, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits, target))

    if not losses:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack(losses).mean()


def compute_ctc_aux_loss(
    ctc_logits: torch.Tensor,
    ctc_mask: torch.Tensor,
    labels: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    # ctc_logits: [B, T, V], labels: [B, L] with -100 ignore index.
    device = ctc_logits.device
    keep_rows: list[int] = []
    targets: list[torch.Tensor] = []
    input_lengths: list[int] = []
    target_lengths: list[int] = []

    for b in range(ctc_logits.size(0)):
        inp_len = int(ctc_mask[b].sum().item())
        tgt = labels[b][labels[b] != -100]
        tgt = tgt[tgt != blank_id]
        tgt_len = int(tgt.numel())

        if inp_len <= 0 or tgt_len <= 0 or tgt_len > inp_len:
            continue

        keep_rows.append(b)
        targets.append(tgt)
        input_lengths.append(inp_len)
        target_lengths.append(tgt_len)

    if not keep_rows:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    idx = torch.tensor(keep_rows, dtype=torch.long, device=device)
    logits = ctc_logits.index_select(0, idx)

    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, V]
    target_concat = torch.cat(targets).to(device)
    input_lengths_t = torch.tensor(input_lengths, dtype=torch.long, device=device)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=device)

    return F.ctc_loss(
        log_probs,
        target_concat,
        input_lengths_t,
        target_lengths_t,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def train_one_epoch(
    model: KeypointT5Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    grad_accum_steps: int,
    signcl_weight: float,
    signcl_temperature: float,
    signcl_neg_distance: int,
    signcl_max_anchors: int,
    signcl_max_negatives: int,
    ctc_weight: float,
    ctc_blank_id: int,
    log_every: int,
    show_progress: bool,
) -> dict[str, float]:
    model.train()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    grad_accum_steps = max(1, grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)

    total = 0.0
    total_seq = 0.0
    total_signcl = 0.0
    total_ctc = 0.0
    steps = 0

    train_iter = tqdm(
        loader,
        total=len(loader),
        desc="Train",
        leave=False,
        disable=not show_progress,
    )
    for steps, batch in enumerate(train_iter, start=1):
        batch = move_to_device(batch, device)

        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(src=batch["src"], src_mask=batch["src_mask"], labels=batch["labels"])
            seq_loss = out["seq2seq"].loss

            signcl_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if signcl_weight > 0:
                signcl_loss = compute_signcl_loss(
                    visual_embeds=out["visual_embeds"],
                    visual_mask=out["visual_mask"],
                    neg_distance=signcl_neg_distance,
                    temperature=signcl_temperature,
                    max_anchors=signcl_max_anchors,
                    max_negatives=signcl_max_negatives,
                )

            ctc_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if ctc_weight > 0:
                ctc_loss = compute_ctc_aux_loss(
                    ctc_logits=out["ctc_logits"],
                    ctc_mask=out["ctc_mask"],
                    labels=batch["labels"],
                    blank_id=ctc_blank_id,
                )

            joint_loss = seq_loss + signcl_weight * signcl_loss + ctc_weight * ctc_loss
            loss_for_backward = joint_loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        should_step = (steps % grad_accum_steps == 0) or (steps == len(loader))
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Detect whether the optimizer actually stepped: GradScaler skips the optimizer
            # step when gradients contain inf/NaN and instead reduces the scale factor.
            # Comparing scale before/after is the only reliable cross-version approach.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scaler_stepped = (scaler.get_scale() >= scale_before)
            if scaler_stepped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total += float(joint_loss.item())
        total_seq += float(seq_loss.item())
        total_signcl += float(signcl_loss.item())
        total_ctc += float(ctc_loss.item())

        if log_every > 0 and steps % log_every == 0:
            msg = (
                f"  step {steps:5d} | total {joint_loss.item():.4f} | "
                f"seq2seq {seq_loss.item():.4f} | signcl {signcl_loss.item():.4f} | ctc {ctc_loss.item():.4f}"
            )
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)

        if show_progress and (steps % max(log_every, 1) == 0 or steps == 1):
            train_iter.set_postfix(
                total=f"{joint_loss.item():.3f}",
                seq=f"{seq_loss.item():.3f}",
                signcl=f"{signcl_loss.item():.3f}",
                ctc=f"{ctc_loss.item():.3f}",
            )

    denom = max(steps, 1)
    return {
        "train_total_loss": total / denom,
        "train_seq2seq_loss": total_seq / denom,
        "train_signcl_loss": total_signcl / denom,
        "train_ctc_loss": total_ctc / denom,
    }


@torch.no_grad()
def evaluate_loss(model: KeypointT5Model, loader: DataLoader, device: torch.device, show_progress: bool) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0

    val_iter = tqdm(
        loader,
        total=len(loader),
        desc="ValLoss",
        leave=False,
        disable=not show_progress,
    )
    for steps, batch in enumerate(val_iter, start=1):
        batch = move_to_device(batch, device)
        out = model(src=batch["src"], src_mask=batch["src_mask"], labels=batch["labels"])
        total_loss += float(out["seq2seq"].loss.item())

        if show_progress and steps % 20 == 0:
            val_iter.set_postfix(loss=f"{(total_loss / steps):.3f}")

    return total_loss / max(steps, 1)


@torch.no_grad()
def generate_predictions(
    model: KeypointT5Model,
    loader: DataLoader,
    tokenizer: T5TokenizerFast,
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
    max_batches: int,
    show_progress: bool,
) -> tuple[list[str], list[str]]:
    model.eval()
    preds: list[str] = []
    refs: list[str] = []

    if max_batches > 0:
        total_batches = min(len(loader), max_batches)
        batch_iterable = itertools.islice(loader, max_batches)
    else:
        total_batches = len(loader)
        batch_iterable = loader

    gen_iter = tqdm(
        batch_iterable,
        total=total_batches,
        desc="Generate",
        leave=False,
        disable=not show_progress,
    )

    for batch in gen_iter:

        batch = move_to_device(batch, device)
        generated = model.generate(
            src=batch["src"],
            src_mask=batch["src_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        ref_texts = [str(t) for t in batch["ref_texts"]]

        preds.extend([p.strip() for p in pred_texts])
        refs.extend([r.strip() for r in ref_texts])

    return preds, refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train How2Sign keypoint-to-text with T5 + SignCL")

    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-tsv", type=str, default="how2sign_train.csv")
    parser.add_argument("--val-tsv", type=str, default="how2sign_val.csv")
    parser.add_argument("--test-tsv", type=str, default="how2sign_test.csv")

    parser.add_argument("--train-keypoints", type=str, default="train_2D_keypoints/openpose_output/json")
    parser.add_argument("--val-keypoints", type=str, default="val_2D_keypoints/openpose_output/json")
    parser.add_argument("--test-keypoints", type=str, default="test_2D_keypoints/openpose_output/json")

    parser.add_argument("--pretrained-model", type=str, default="t5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--max-target-tokens", type=int, default=64)
    parser.add_argument("--max-gen-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--use-face", action="store_true")

    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--interpolation-gap", type=int, default=3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--scale-jitter", type=float, default=0.1)

    parser.add_argument("--signcl-weight", type=float, default=0.01)
    parser.add_argument("--signcl-temperature", type=float, default=0.07)
    parser.add_argument("--signcl-neg-distance", type=int, default=20)
    parser.add_argument("--signcl-max-anchors", type=int, default=32)
    parser.add_argument("--signcl-max-negatives", type=int, default=64)

    parser.add_argument("--ctc-weight", type=float, default=0.0)

    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    parser.add_argument("--gap-audit-max-clips", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--eval-test-on-best", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--save-dir", type=Path, default=Path("runs/how2sign_t5"))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--print-pred-samples", type=int, default=3)
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars")

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a last.pt checkpoint to resume training from. "
             "Model weights, optimizer, scheduler, and best-BLEU are all restored. "
             "Epochs already recorded in metrics.csv are skipped.",
    )
    parser.add_argument(
        "--reset-lr",
        action="store_true",
        help="When resuming, discard the saved (possibly decayed) scheduler and start "
             "a fresh cosine LR schedule for the remaining epochs. "
             "Model weights and optimizer momentum are fully preserved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = args.data_root.resolve()
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_tsv = root / args.train_tsv
    val_tsv = root / args.val_tsv
    test_tsv = root / args.test_tsv
    train_kp = root / args.train_keypoints
    val_kp = root / args.val_keypoints
    test_kp = root / args.test_keypoints

    for p in [train_tsv, val_tsv, train_kp, val_kp]:
        if not p.exists():
            raise FileNotFoundError(f"Required path missing: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU not detected in this environment. Use WSL CUDA run for full training, or pass --allow-cpu for debug only."
        )

    amp_enabled = bool(args.amp and device.type == "cuda")
    effective_batch = args.batch_size * max(1, args.grad_accum_steps)

    print(f"Device: {device}")
    print(f"Effective batch size: {effective_batch}")
    show_progress = not args.disable_tqdm
    print("Loading tokenizer/model...")
    tokenizer = T5TokenizerFast.from_pretrained(args.pretrained_model)

    print("Reading train/val metadata with explicit skip logging...")
    train_records, train_skipped = read_split_records(train_tsv, train_kp, max_samples=args.max_train_samples)
    val_records, val_skipped = read_split_records(val_tsv, val_kp, max_samples=args.max_val_samples)

    if not train_records:
        raise RuntimeError("No usable train samples found")
    if not val_records:
        raise RuntimeError("No usable val samples found")

    write_skip_log(save_dir / "train_skipped.csv", train_skipped)
    write_skip_log(save_dir / "val_skipped.csv", val_skipped)

    train_total = len(train_records) + len(train_skipped)
    val_total = len(val_records) + len(val_skipped)
    print(
        f"Train kept={len(train_records)} skipped={len(train_skipped)} "
        f"({100.0 * len(train_skipped) / max(train_total, 1):.2f}%) | reasons={reason_counts(train_skipped)}"
    )
    print(
        f"Val   kept={len(val_records)} skipped={len(val_skipped)} "
        f"({100.0 * len(val_skipped) / max(val_total, 1):.2f}%) | reasons={reason_counts(val_skipped)}"
    )

    if args.gap_audit_max_clips != 0:
        print("Running missing-gap audit...")
        train_gap = collect_gap_audit(
            records=train_records,
            use_face=args.use_face,
            min_conf=args.min_conf,
            max_clips=args.gap_audit_max_clips,
            show_progress=show_progress,
        )
        val_gap = collect_gap_audit(
            records=val_records,
            use_face=args.use_face,
            min_conf=args.min_conf,
            max_clips=min(args.gap_audit_max_clips, len(val_records)),
            show_progress=show_progress,
        )
        with (save_dir / "train_gap_audit.json").open("w", encoding="utf-8") as f:
            json.dump(train_gap, f, indent=2)
        with (save_dir / "val_gap_audit.json").open("w", encoding="utf-8") as f:
            json.dump(val_gap, f, indent=2)

    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    train_ds = How2SignT5Dataset(
        records=train_records,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_target_tokens=args.max_target_tokens,
        use_face=args.use_face,
        min_conf=args.min_conf,
        interpolation_gap=args.interpolation_gap,
        training=True,
        augment=args.augment,
        flip_prob=args.flip_prob,
        scale_jitter=args.scale_jitter,
        cache_size=args.cache_size,
    )
    val_ds = How2SignT5Dataset(
        records=val_records,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_target_tokens=args.max_target_tokens,
        use_face=args.use_face,
        min_conf=args.min_conf,
        interpolation_gap=args.interpolation_gap,
        training=False,
        augment=False,
        flip_prob=0.0,
        scale_jitter=0.0,
        cache_size=args.cache_size,
    )

    collate = BatchCollator(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    model = KeypointT5Model(
        pretrained_name=args.pretrained_model,
        input_dim=feature_dim(args.use_face),
        dropout=args.dropout,
        temporal_stride=args.temporal_stride,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use len(train_records) directly — avoids relying on DataLoader internals
    # and is guaranteed to be set before the DataLoader is ever iterated.
    effective_batch_size = args.batch_size * max(1, args.grad_accum_steps)
    optim_steps_per_epoch = math.ceil(len(train_records) / effective_batch_size)
    total_steps = max(1, optim_steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Scheduler | steps/epoch={optim_steps_per_epoch} | total_steps={total_steps} | warmup_steps={warmup_steps} | lr={args.lr}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    metrics_path = save_dir / "metrics.csv"
    METRICS_HEADER = [
        "epoch",
        "train_total_loss",
        "train_seq2seq_loss",
        "train_signcl_loss",
        "train_ctc_loss",
        "val_seq2seq_loss",
        "val_bleu_sacrebleu",
        "val_bleu1_sacrebleu",
        "val_bleu2_sacrebleu",
        "val_bleu3_sacrebleu",
        "val_bleu4_sacrebleu",
        "val_reduced_bleu_sacrebleu",
        "val_meteor",
        "val_rouge_l",
        "seconds",
        "lr",
    ]

    # ── Resume-from-checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_reduced_bleu = -1.0

    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")

        print(f"Resuming from checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])

        last_epoch = int(resume_ckpt["epoch"])
        start_epoch = last_epoch + 1

        # Recover best BLEU seen so far from the saved val_metrics
        saved_metrics = resume_ckpt.get("val_metrics", {})
        best_reduced_bleu = float(saved_metrics.get("reduced_bleu", -1.0))

        if args.reset_lr:
            # Discard the stale decayed scheduler; build a fresh one for remaining epochs.
            # Reset every param group LR to the target value first so Adam starts clean.
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            remaining_epochs = args.epochs - last_epoch
            remaining_steps = max(1, optim_steps_per_epoch * remaining_epochs)
            # Short cosine warmup (2% of remaining) then cosine decay to lr/10
            cosine_warmup = max(1, int(remaining_steps * 0.02))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cosine_warmup,
                num_training_steps=remaining_steps,
            )
            print(
                f"  [reset-lr] Fresh cosine schedule | remaining_steps={remaining_steps} "
                f"| warmup={cosine_warmup} | lr={args.lr} | eta_min={args.lr*0.05:.2e}"
            )
        else:
            # Restore the original scheduler state (may be decayed — use --reset-lr to fix)
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])

        print(
            f"  Resumed: last completed epoch={last_epoch}, "
            f"best reduced-BLEU so far={best_reduced_bleu:.4f}, "
            f"continuing from epoch {start_epoch}/{args.epochs}"
        )

        if start_epoch > args.epochs:
            print("All epochs already completed. Nothing to do.")
            return

        # Append to existing metrics.csv; write header only if file is new/empty
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            with metrics_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(METRICS_HEADER)
        # (existing rows stay intact — we only append new epochs below)

    else:
        # Fresh run: write a new metrics.csv with header
        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp=amp_enabled,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            signcl_weight=args.signcl_weight,
            signcl_temperature=args.signcl_temperature,
            signcl_neg_distance=args.signcl_neg_distance,
            signcl_max_anchors=args.signcl_max_anchors,
            signcl_max_negatives=args.signcl_max_negatives,
            ctc_weight=args.ctc_weight,
            ctc_blank_id=tokenizer.unk_token_id,
            log_every=args.log_every,
            show_progress=show_progress,
        )

        val_loss = evaluate_loss(model=model, loader=val_loader, device=device, show_progress=show_progress)
        val_preds, val_refs = generate_predictions(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_gen_tokens,
            num_beams=args.num_beams,
            max_batches=args.eval_max_batches,
            show_progress=show_progress,
        )
        val_metrics = evaluate_text_metrics(val_preds, val_refs)

        elapsed = time.time() - start
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:02d} | train_total={train_stats['train_total_loss']:.4f} | "
            f"val_loss={val_loss:.4f} | bleu={val_metrics['bleu']:.2f} | "
            f"reduced_bleu={val_metrics['reduced_bleu']:.2f} | "
            f"meteor={val_metrics['meteor']:.2f} | rouge_l={val_metrics['rouge_l']:.2f} | "
            f"time={elapsed:.1f}s"
        )

        with metrics_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_stats["train_total_loss"],
                    train_stats["train_seq2seq_loss"],
                    train_stats["train_signcl_loss"],
                    train_stats["train_ctc_loss"],
                    val_loss,
                    val_metrics["bleu"],
                    val_metrics["bleu1"],
                    val_metrics["bleu2"],
                    val_metrics["bleu3"],
                    val_metrics["bleu4"],
                    val_metrics["reduced_bleu"],
                    val_metrics["meteor"],
                    val_metrics["rouge_l"],
                    elapsed,
                    lr,
                ]
            )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_seq2seq_loss": val_loss,
            "val_metrics": val_metrics,
            "train_stats": train_stats,
            "args": vars(args),
            "tokenizer_name": args.pretrained_model,
        }
        torch.save(ckpt, save_dir / "last.pt")

        if val_metrics["reduced_bleu"] > best_reduced_bleu:
            best_reduced_bleu = val_metrics["reduced_bleu"]
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  Saved best checkpoint on reduced-BLEU to: {save_dir / 'best.pt'}")

            sample_count = min(args.print_pred_samples, len(val_preds), len(val_refs))
            samples = [
                {"reference": val_refs[i], "prediction": val_preds[i]}
                for i in range(sample_count)
            ]
            with (save_dir / "best_val_samples.json").open("w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)

    print("Training complete.")
    print(f"Best validation reduced-BLEU: {best_reduced_bleu:.2f}")

    if args.eval_test_on_best:
        if not test_tsv.exists() or not test_kp.exists():
            print("Skipping test evaluation because test TSV/keypoint path is missing.")
            return

        test_records, test_skipped = read_split_records(test_tsv, test_kp, max_samples=args.max_test_samples)
        write_skip_log(save_dir / "test_skipped.csv", test_skipped)
        if not test_records:
            print("Skipping test evaluation because no usable test samples were found.")
            return

        best_ckpt = torch.load(save_dir / "best.pt", map_location=device, weights_only=False)  # safe: saved by this script
        model.load_state_dict(best_ckpt["model_state"])
        model.eval()

        test_ds = How2SignT5Dataset(
            records=test_records,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            max_target_tokens=args.max_target_tokens,
            use_face=args.use_face,
            min_conf=args.min_conf,
            interpolation_gap=args.interpolation_gap,
            training=False,
            augment=False,
            flip_prob=0.0,
            scale_jitter=0.0,
            cache_size=args.cache_size,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )

        test_preds, test_refs = generate_predictions(
            model=model,
            loader=test_loader,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_gen_tokens,
            num_beams=args.num_beams,
            max_batches=0,
            show_progress=show_progress,
        )
        test_metrics = evaluate_text_metrics(test_preds, test_refs)
        test_metrics["num_test_samples"] = len(test_records)
        test_metrics["num_test_skipped"] = len(test_skipped)
        test_metrics["skip_reasons"] = reason_counts(test_skipped)

        with (save_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        with (save_dir / "test_predictions.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["reference", "prediction"])
            for ref, pred in zip(test_refs, test_preds):
                writer.writerow([ref, pred])

        print(
            f"Test BLEU={test_metrics['bleu']:.2f} | reduced-BLEU={test_metrics['reduced_bleu']:.2f} | "
            f"METEOR={test_metrics['meteor']:.2f} | ROUGE-L={test_metrics['rouge_l']:.2f}"
        )


if __name__ == "__main__":
    main()
