#!/usr/bin/env bash
# launch_local.sh — runs training from YOUR machine's dataset paths
# Works in WSL, Git Bash, or any bash-compatible shell.
#
# On Windows/WSL, mount E: as /mnt/e — adjust if using a different letter.
set -euo pipefail

# ── Paths (Windows drive letters converted to WSL mounts) ─────────────────────
PROJECT_ROOT="/mnt/e/CaptioningTool"
TRANSLATIONS_ROOT="$PROJECT_ROOT/DATASET/English translation"
KEYPOINTS_ROOT="$PROJECT_ROOT/DATASET/2d_keypoints"

TRAIN_TSV="$TRANSLATIONS_ROOT/how2sign_train.csv"
VAL_TSV="$TRANSLATIONS_ROOT/how2sign_val.csv"
TEST_TSV="$TRANSLATIONS_ROOT/how2sign_test.csv"

TRAIN_KP="$KEYPOINTS_ROOT/train_2D_keypoints/openpose_output/json"
VAL_KP="$KEYPOINTS_ROOT/val_2D_keypoints/openpose_output/json"
TEST_KP="$KEYPOINTS_ROOT/test_2D_keypoints/openpose_output/json"

SAVE_DIR="$PROJECT_ROOT/runs/how2sign_t5_local"

# ── Virtual-env activation (adjust path if yours is different) ────────────────
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/.venv_wsl/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv_wsl/bin/activate"
fi

mkdir -p "$SAVE_DIR"

# ── Training invocation ───────────────────────────────────────────────────────
python -u "$PROJECT_ROOT/train_how2sign_t5.py" \
  --data-root       "$PROJECT_ROOT" \
  --train-tsv       "$TRAIN_TSV" \
  --val-tsv         "$VAL_TSV" \
  --test-tsv        "$TEST_TSV" \
  --train-keypoints "$TRAIN_KP" \
  --val-keypoints   "$VAL_KP" \
  --test-keypoints  "$TEST_KP" \
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
  --save-dir "$SAVE_DIR" \
  --amp \
  --log-every 200 \
  2>&1 | tee "$SAVE_DIR/train.log"
