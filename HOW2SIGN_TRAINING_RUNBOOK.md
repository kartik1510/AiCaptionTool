# How2Sign Training Runbook

This document records exactly what was implemented, what was understood about your data, and the training approach used.

Note: a corrected methodology and implementation is documented in [HOW2SIGN_TRAINING_RUNBOOK_V2.md](HOW2SIGN_TRAINING_RUNBOOK_V2.md).

## 1) Objective

Train a keypoint-to-text model on the How2Sign split files using OpenPose 2D keypoints as input and the English sentence as target output.

## 2) Dataset Understanding

The training code assumes:

- Annotation files are TSV (tab-delimited), not comma-delimited.
- Required columns are `SENTENCE_NAME` and `SENTENCE`.
- `SENTENCE_NAME` is the clip key used to locate keypoint folders.
- Keypoint frames are OpenPose JSON files named like `*_keypoints.json`.

Observed split counts from the current dataset:

- Train usable samples: 31047 (118 missing keypoint folders)
- Val usable samples: 1739 (2 missing keypoint folders)

The trainer automatically skips rows that are missing keypoint folders.

## 3) What Was Implemented

### 3.1 Main Trainer Script

Added [train_how2sign.py](train_how2sign.py), an end-to-end training pipeline including:

- TSV parsing and filtering
- OpenPose JSON loading
- Keypoint feature extraction and normalization
- Tokenization and vocabulary building
- Batched dynamic padding for source and target sequences
- Transformer encoder-decoder model
- Training loop, validation loop, checkpointing, and metrics logging

### 3.2 Dependency Setup

Installed into the workspace virtual environment:

- `torch`
- `numpy`

### 3.3 Compatibility Fixes

Adjusted mixed-precision scaler construction to support current PyTorch API behavior while remaining backward compatible:

- Tries `torch.amp.GradScaler("cuda", enabled=amp)` first
- Falls back to `torch.cuda.amp.GradScaler(enabled=amp)`

### 3.4 Validation Runs Performed

Smoke run command executed successfully on a tiny subset:

```powershell
c:/Users/jnami/Downloads/Compressed/kartik/.venv/Scripts/python.exe train_how2sign.py --data-root . --epochs 1 --batch-size 2 --max-train-samples 8 --max-val-samples 4 --max-frames 16 --num-workers 0 --d-model 128 --nhead 4 --enc-layers 2 --dec-layers 2 --ffn-dim 256 --log-every 1 --print-samples 1 --save-dir runs/smoke_run
```

Smoke artifacts:

- [runs/smoke_run/best.pt](runs/smoke_run/best.pt)
- [runs/smoke_run/last.pt](runs/smoke_run/last.pt)
- [runs/smoke_run/metrics.csv](runs/smoke_run/metrics.csv)
- [runs/smoke_run/vocab.json](runs/smoke_run/vocab.json)

Then launched full training:

```powershell
c:/Users/jnami/Downloads/Compressed/kartik/.venv/Scripts/python.exe train_how2sign.py --data-root . --epochs 20 --batch-size 8 --max-frames 96 --max-target-tokens 64 --num-workers 2 --d-model 256 --nhead 8 --enc-layers 4 --dec-layers 4 --ffn-dim 1024 --lr 2e-4 --weight-decay 1e-4 --cache-size 512 --log-every 200 --print-samples 1 --save-dir runs/how2sign_full_v1
```

Full-run artifacts (created at launch):

- [runs/how2sign_full_v1/metrics.csv](runs/how2sign_full_v1/metrics.csv)
- [runs/how2sign_full_v1/vocab.json](runs/how2sign_full_v1/vocab.json)

## 4) Approach Used

### 4.1 Input Representation

Per frame, features are built from OpenPose 2D blocks:

- body pose: 25 keypoints
- left hand: 21 keypoints
- right hand: 21 keypoints
- optional face: 70 keypoints

Each keypoint contributes `(x, y, confidence)`.

Feature dimension:

- without face: `(25 + 21 + 21) * 3 = 201`
- with face: `(25 + 21 + 21 + 70) * 3 = 411`

If multiple people exist in a frame, the selected person is the one with highest total keypoint confidence.

### 4.2 Spatial Normalization

For valid keypoints (`confidence > 0`), coordinates are standardized within the frame:

$$
x' = \frac{x - \mu_x}{\sigma_x}, \quad y' = \frac{y - \mu_y}{\sigma_y}
$$

Invalid points are zeroed in `(x, y)` while confidence is kept.

### 4.3 Temporal Handling

- Frame files are sorted by filename.
- If the clip has more than `max_frames`, uniform temporal subsampling is applied with evenly spaced indices.
- If a clip has no valid keypoint frame, a single zero frame is used.

### 4.4 Text Processing

- Lowercasing and regex tokenization.
- Vocabulary built from train split only.
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- Target training format:
  - decoder input: `[BOS] + tokens`
  - decoder output: `tokens + [EOS]`

### 4.5 Model Architecture

Sequence-to-sequence Transformer with:

- source projection `Linear(input_dim -> d_model)`
- sinusoidal positional encoding for source and target
- Transformer encoder stack
- Transformer decoder stack with causal mask
- output projection to vocabulary logits

### 4.6 Optimization and Metrics

- Optimizer: `AdamW`
- Loss: token-level cross entropy with `ignore_index=pad_id`
- Gradient clipping: configurable (`--grad-clip`)
- Scheduler: `ReduceLROnPlateau` on validation loss
- Logged metrics:
  - train loss
  - train token accuracy
  - val loss
  - val token accuracy
  - epoch seconds

### 4.7 Checkpointing

Each epoch writes:

- `last.pt` (latest state)
- `best.pt` (whenever validation loss improves)

Checkpoints include:

- model state
- optimizer state
- epoch and val loss
- vocabulary
- argument set

## 5) How To Run

Run from dataset root `C:\Users\jnami\Downloads\Compressed\kartik`.

### 5.1 Quick Sanity Run

```powershell
c:/Users/jnami/Downloads/Compressed/kartik/.venv/Scripts/python.exe train_how2sign.py --data-root . --epochs 1 --batch-size 2 --max-train-samples 16 --max-val-samples 8 --max-frames 16 --num-workers 0 --d-model 128 --nhead 4 --enc-layers 2 --dec-layers 2 --ffn-dim 256 --save-dir runs/quick_check
```

### 5.2 Full Run

```powershell
c:/Users/jnami/Downloads/Compressed/kartik/.venv/Scripts/python.exe train_how2sign.py --data-root . --epochs 20 --batch-size 8 --max-frames 96 --max-target-tokens 64 --num-workers 2 --d-model 256 --nhead 8 --enc-layers 4 --dec-layers 4 --ffn-dim 1024 --lr 2e-4 --weight-decay 1e-4 --cache-size 512 --log-every 200 --print-samples 1 --save-dir runs/how2sign_full_v1
```

## 6) Monitoring and Artifacts

During training, check:

- [runs/how2sign_full_v1/metrics.csv](runs/how2sign_full_v1/metrics.csv)
- console output for per-epoch summary and sample predictions

After the first epoch finishes, checkpoint files are written in `runs/how2sign_full_v1` as `best.pt` and `last.pt`.

## 7) Current Limitations and Next Improvements

Current implementation is intentionally straightforward and practical, but not yet state of the art.

Recommended next upgrades:

1. Add evaluation metrics beyond token accuracy (BLEU, ROUGE, WER-like scoring).
2. Add beam search decoding for better validation predictions.
3. Add resume-from-checkpoint and early stopping flags.
4. Add distributed/multi-GPU support.
5. Add richer keypoint augmentations (temporal jitter, coordinate noise, confidence dropout).
6. Add optional use of side-view keypoints for multi-view modeling.

## 8) File Map

- Trainer: [train_how2sign.py](train_how2sign.py)
- Audit script: [audit_name_alignment.py](audit_name_alignment.py)
- Full run dir: [runs/how2sign_full_v1](runs/how2sign_full_v1)
- Smoke run dir: [runs/smoke_run](runs/smoke_run)
