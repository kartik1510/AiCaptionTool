# How2Sign Training Runbook V2 (Current)

This runbook documents the current corrected pipeline in [train_how2sign_t5.py](train_how2sign_t5.py), including what is implemented and what was validated in WSL CUDA.

## 1) Design Goals

The V2 pipeline is built around four goals:

1. Use a pretrained text prior (T5 + T5 tokenizer).
2. Normalize and repair keypoint sequences before translation training.
3. Improve representation separability and temporal alignment with auxiliary objectives.
4. Log metrics and data-quality signals that are comparable and actionable.

## 2) Implemented Architecture

### 2.1 Visual Representation

- Input modality: OpenPose 2D keypoints from clip frame JSON files.
- Joint groups:
  - pose: 25
  - left hand: 21
  - right hand: 21
  - optional face: 70

Per-joint feature channels:

- normalized x
- normalized y
- confidence
- missing flag

### 2.2 Normalization and Missing Data

Implemented in [train_how2sign_t5.py](train_how2sign_t5.py):

- shoulder-anchored signing-space normalization (center and scale)
- fallback normalization when shoulders are unreliable
- linear interpolation for missing runs with gap <= 3
- explicit missing-flag channel retained for larger gaps

### 2.3 Temporal Strategy

- train: stochastic temporal bin sampling
- eval: deterministic uniform sampling
- structural compression: Conv1D temporal compression before T5 encoder (`--temporal-stride`)

### 2.4 Model Core

- backbone: pretrained T5 (`t5-small` default)
- bridge: linear projection from keypoint feature vector to T5 hidden size
- regularization: layer norm + dropout

## 3) Auxiliary Objectives

### 3.1 SignCL (implemented)

- adjacent frames as positive pairs
- negatives sampled from far temporal distance (`--signcl-neg-distance`, default 20)
- weighted loss term (`--signcl-weight`, default 0.01)

### 3.2 CTC Auxiliary Loss (implemented, optional)

- CTC head on encoder sequence states
- token-level CTC over text tokens without gloss labels
- controlled by `--ctc-weight` (default 0.0; set >0 to enable)

## 4) Data Health and Auditing

### 4.1 Skipped-sample logging

For each split, skipped rows are saved with reason to CSV.

### 4.2 Missing-gap distribution

Gap-size histograms are computed and saved as JSON:

- `train_gap_audit.json`
- `val_gap_audit.json`

Audit scope is controlled by `--gap-audit-max-clips`.

## 5) Evaluation Metrics

Validation metrics now include:

- BLEU (sacrebleu)
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- reduced-BLEU (sacrebleu)
- METEOR (NLTK if available, fallback approximation otherwise)
- ROUGE-L

Checkpoint selection uses reduced-BLEU.

## 6) Training Controls

Key controls in [train_how2sign_t5.py](train_how2sign_t5.py):

- gradient accumulation: `--grad-accum-steps`
- AMP on CUDA: `--amp`
- scheduler warmup ratio: `--warmup-ratio`
- GPU guardrail: CPU mode requires explicit `--allow-cpu`

## 7) Validated Runs

### 7.1 Local CPU debug smoke (validated)

- output: [runs/t5_smoke_v2](runs/t5_smoke_v2)

### 7.2 WSL CUDA smoke (validated)

- output: [runs/t5_smoke_wsl](runs/t5_smoke_wsl)

### 7.3 WSL CUDA smoke after scheduler fix (validated)

- output: [runs/t5_smoke_wsl_v2](runs/t5_smoke_wsl_v2)

## 8) WSL CUDA Operations

Use [WSL_CUDA_TRAINING.md](WSL_CUDA_TRAINING.md) for:

- environment bootstrap in WSL
- CUDA verification commands
- full-run launch command tuned for this project

## 9) Known Remaining Gaps

1. PoseStitch synthetic pretraining stage is not yet implemented in code.
2. VLP-style intermediate contrastive pretraining stage is not yet implemented.
3. Keypoint extractor remains dataset-provided OpenPose (no MediaPipe/MMPose re-extraction yet).
