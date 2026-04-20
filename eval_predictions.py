#!/usr/bin/env python3
"""
Standalone evaluation script — loads a checkpoint and generates predictions
on a subset of the validation set.

Usage:
    python eval_predictions.py --checkpoint runs/how2sign_t5_full/best.pt --num-samples 20
    python eval_predictions.py --checkpoint runs/how2sign_t5_full/last.pt --num-samples 20
"""
import argparse
import sys
from pathlib import Path

import torch

# Import everything we need from the training script
sys.path.insert(0, str(Path(__file__).parent))
from train_how2sign_t5 import (
    How2SignT5Dataset,
    KeypointT5Model,
    BatchCollator,
    read_split_records,
    generate_predictions,
    evaluate_text_metrics,
    feature_dim,
)
from transformers import T5TokenizerFast
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and print predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20, help="Number of val samples to run")
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", "?")
    val_metrics = ckpt.get("val_metrics", {})
    saved_args = argparse.Namespace(**ckpt["args"])
    tokenizer_name = ckpt.get("tokenizer_name", saved_args.pretrained_model)

    print(f"Checkpoint epoch : {epoch}")
    print(f"Val loss         : {ckpt.get('val_seq2seq_loss', 'N/A'):.4f}")
    print(f"Val BLEU-4       : {val_metrics.get('bleu', 'N/A'):.4f}")
    print(f"Val METEOR       : {val_metrics.get('meteor', 'N/A'):.4f}")
    print(f"Val ROUGE-L      : {val_metrics.get('rouge_l', 'N/A'):.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)

    # Val dataset
    val_records, skipped = read_split_records(
        Path(saved_args.val_tsv),
        Path(saved_args.val_keypoints),
        max_samples=args.num_samples,
    )
    print(f"Val samples loaded: {len(val_records)} (skipped: {len(skipped)})\n")

    val_ds = How2SignT5Dataset(
        records=val_records,
        tokenizer=tokenizer,
        max_frames=saved_args.max_frames,
        max_target_tokens=saved_args.max_target_tokens,
        use_face=saved_args.use_face,
        min_conf=saved_args.min_conf,
        interpolation_gap=saved_args.interpolation_gap,
        training=False,
        augment=False,
        flip_prob=0.0,
        scale_jitter=0.0,
        cache_size=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=saved_args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=BatchCollator(tokenizer),
    )

    # Build model and load weights
    model = KeypointT5Model(
        pretrained_name=tokenizer_name,
        input_dim=feature_dim(saved_args.use_face),
        temporal_stride=saved_args.temporal_stride,
        dropout=saved_args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Generate predictions
    preds, refs = generate_predictions(
        model=model,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=saved_args.max_gen_tokens,
        num_beams=args.num_beams,
        max_batches=0,
        show_progress=True,
    )

    metrics = evaluate_text_metrics(preds, refs)
    print(f"\n=== Metrics on {len(preds)} samples ===")
    print(f"  BLEU-4    : {metrics['bleu']:.4f}")
    print(f"  BLEU-1    : {metrics['bleu1']:.4f}")
    print(f"  BLEU-2    : {metrics['bleu2']:.4f}")
    print(f"  BLEU-3    : {metrics['bleu3']:.4f}")
    print(f"  METEOR    : {metrics['meteor']:.4f}")
    print(f"  ROUGE-L   : {metrics['rouge_l']:.4f}")

    empty = sum(1 for p in preds if not p.strip())
    print(f"  Empty preds: {empty}/{len(preds)}")

    print(f"\n=== Predictions (epoch {epoch}) ===\n")
    for i, (ref, pred) in enumerate(zip(refs, preds)):
        status = " [EMPTY]" if not pred.strip() else ""
        print(f"[{i+1:02d}] REF : {ref}")
        print(f"      PRED: {pred}{status}")
        print()


if __name__ == "__main__":
    main()
