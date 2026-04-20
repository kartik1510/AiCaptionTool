# How2Sign Keypoint-to-Text Training (WSL CUDA)

This workspace trains gloss-free sign-language translation models from OpenPose keypoints to English text on the How2Sign splits.

The current recommended pipeline is implemented in [train_how2sign_t5.py](train_how2sign_t5.py) and is designed to run on CUDA through WSL.

## Current Status

- Upgraded trainer implemented and validated on WSL GPU.
- Full training is launched through [launch_wsl_full.sh](launch_wsl_full.sh).
- Detailed chronology is in [WORKLOG.md](WORKLOG.md).

## Dataset Layout Expected

The trainer expects these TSV files at project root:

- [how2sign_train.csv](how2sign_train.csv)
- [how2sign_val.csv](how2sign_val.csv)
- [how2sign_test.csv](how2sign_test.csv)

And keypoint roots:

- `train_2D_keypoints/openpose_output/json`
- `val_2D_keypoints/openpose_output/json`
- `test_2D_keypoints/openpose_output/json`

## Recommended Trainer

Main script:

- [train_how2sign_t5.py](train_how2sign_t5.py)

Core features currently implemented:

- shoulder-anchored signing-space normalization with fallback geometry
- linear interpolation for short missing-keypoint gaps
- explicit missing-joint feature channel
- stochastic temporal sampling (train) + deterministic sampling (eval)
- temporal Conv1D compression before T5 encoder
- pretrained T5 tokenizer and model bridge
- SignCL auxiliary loss
- optional CTC auxiliary loss
- metrics: BLEU, BLEU-1/2/3/4, reduced-BLEU, METEOR, ROUGE-L
- skipped-row logs and gap-distribution audits

## WSL CUDA Quick Start

For full setup and operations, see [WSL_CUDA_TRAINING.md](WSL_CUDA_TRAINING.md).

### 1) Verify CUDA in WSL

```powershell
wsl --cd /mnt/c/Users/jnami/Downloads/Compressed/kartik --exec ./.venv_wsl/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2) Launch Full Training (Background)

```powershell
wsl -e bash /mnt/c/Users/jnami/Downloads/Compressed/kartik/launch_wsl_full.sh
```

### 3) Monitor Logs

```powershell
wsl -e bash -lc 'tail -n 60 /mnt/c/Users/jnami/Downloads/Compressed/kartik/runs/how2sign_t5_wsl_full/train.log'
```

## Output Files

Typical files written under your chosen save directory:

- `metrics.csv`
- `best.pt`
- `last.pt`
- `train_skipped.csv`
- `val_skipped.csv`
- `train_gap_audit.json`
- `val_gap_audit.json`
- `best_val_samples.json`

For the current full run:

- [runs/how2sign_t5_wsl_full/train.log](runs/how2sign_t5_wsl_full/train.log)
- [runs/how2sign_t5_wsl_full/train.pid](runs/how2sign_t5_wsl_full/train.pid)

## Documentation

- Baseline runbook: [HOW2SIGN_TRAINING_RUNBOOK.md](HOW2SIGN_TRAINING_RUNBOOK.md)
- Corrected runbook (current): [HOW2SIGN_TRAINING_RUNBOOK_V2.md](HOW2SIGN_TRAINING_RUNBOOK_V2.md)
- WSL operations: [WSL_CUDA_TRAINING.md](WSL_CUDA_TRAINING.md)
- Work history: [WORKLOG.md](WORKLOG.md)

## Important Notes

- The baseline script [train_how2sign.py](train_how2sign.py) is retained for reference, but the recommended path is [train_how2sign_t5.py](train_how2sign_t5.py).
- PoseStitch synthetic pretraining and VLP pretraining stages are not yet implemented.
- Re-extraction with MediaPipe/MMPose is not yet implemented; this pipeline currently uses provided OpenPose keypoints.
