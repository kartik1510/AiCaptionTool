#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/jnami/Downloads/Compressed/kartik

source .venv_wsl/bin/activate

mkdir -p runs/how2sign_t5_wsl_full

python -u train_how2sign_t5.py \
  --data-root . \
  --pretrained-model t5-small \
  --epochs 20 \
  --batch-size 2 \
  --grad-accum-steps 16 \
  --num-workers 2 \
  --max-frames 96 \
  --temporal-stride 2 \
  --max-target-tokens 64 \
  --max-gen-tokens 64 \
  --num-beams 4 \
  --lr 3e-5 \
  --weight-decay 1e-4 \
  --warmup-ratio 0.06 \
  --grad-clip 1.0 \
  --min-conf 0.05 \
  --interpolation-gap 3 \
  --augment \
  --flip-prob 0.5 \
  --scale-jitter 0.1 \
  --signcl-weight 0.01 \
  --signcl-temperature 0.07 \
  --signcl-neg-distance 20 \
  --signcl-max-anchors 32 \
  --signcl-max-negatives 64 \
  --ctc-weight 0.05 \
  --gap-audit-max-clips 1000 \
  --eval-test-on-best \
  --save-dir runs/how2sign_t5_wsl_full \
  --amp \
  --log-every 200 \
  2>&1 | tee runs/how2sign_t5_wsl_full/train.log