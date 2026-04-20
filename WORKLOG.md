# Work Log

## 2026-04-10

### Scope

Build and run a trainable How2Sign keypoint-to-text pipeline from this dataset root.

### Actions Completed

1. Confirmed no existing training script in workspace.
2. Added [train_how2sign.py](train_how2sign.py) with:
   - TSV ingestion
   - keypoint loading from OpenPose JSON
   - vocabulary and tokenization
   - Transformer encoder-decoder
   - train/val loop
   - checkpointing and metrics logging
3. Configured workspace Python environment.
4. Installed missing runtime dependencies:
   - torch
   - numpy
5. Ran smoke training and verified success using `runs/smoke_run`.
6. Fixed GradScaler compatibility in trainer for current PyTorch behavior.
7. Launched full training run to `runs/how2sign_full_v1`.
8. Added complete technical documentation in [HOW2SIGN_TRAINING_RUNBOOK.md](HOW2SIGN_TRAINING_RUNBOOK.md).
9. Reviewed critical methodology flaws and switched to corrective implementation work.
10. Stopped the baseline CPU full run to avoid wasting compute on a non-comparable setup.
11. Added corrected trainer [train_how2sign_t5.py](train_how2sign_t5.py) with:
   - signing-space normalization (shoulder-anchored with fallback)
   - linear interpolation for missing keypoints (gap <= 3)
   - explicit missing-joint feature channel
   - stochastic temporal frame sampling for train, deterministic for eval
   - keypoint augmentation (horizontal flip with L/R swap, scale jitter)
   - pretrained T5 encoder-decoder with projection bridge
   - sacrebleu metrics and reduced-BLEU checkpointing
   - explicit skipped-sample CSV logs per split
12. Installed additional dependencies:
   - transformers
   - sacrebleu
   - sentencepiece
13. Ran corrected smoke training successfully using `runs/t5_smoke`.
14. Verified CUDA availability: `False` on this machine, so full corrected training was not launched locally.
15. Upgraded [train_how2sign_t5.py](train_how2sign_t5.py) with deeper literature-driven changes:
   - SignCL auxiliary contrastive loss (adjacent positives, long-range negatives)
   - optional CTC auxiliary alignment loss on text tokens
   - temporal Conv1D compression before T5 encoder
   - BLEU-1/2/3/4 + reduced-BLEU + METEOR + ROUGE-L logging
   - gradient accumulation support for larger effective batch size
   - missing-gap distribution audit JSONs (`train_gap_audit.json`, `val_gap_audit.json`)
16. Verified upgraded script locally via CPU debug smoke run at `runs/t5_smoke_v2`.
17. Verified WSL GPU visibility (`nvidia-smi`) and installed WSL dependencies in `.venv_wsl`.
18. Validated CUDA-backed WSL smoke run at `runs/t5_smoke_wsl`.
19. Patched scheduler stepping behavior to avoid false warning when optimizer step is skipped by AMP scaler.
20. Re-validated patched trainer on WSL GPU at `runs/t5_smoke_wsl_v2`.
21. Added WSL operational guide [WSL_CUDA_TRAINING.md](WSL_CUDA_TRAINING.md).
22. Added launch helper [launch_wsl_full.sh](launch_wsl_full.sh) to run full WSL training in background with persisted logs.
23. Launched full WSL CUDA training in background (PID saved in `runs/how2sign_t5_wsl_full/train.pid`).
24. Confirmed live startup logs in `runs/how2sign_t5_wsl_full/train.log`:
   - Device: cuda
   - Effective batch size: 32
   - Metadata loading started
25. Added project-level [README.md](README.md) summarizing current pipeline, WSL CUDA workflow, and monitoring commands.
26. Added tqdm progress bars to [train_how2sign_t5.py](train_how2sign_t5.py) for:
   - gap-audit loop
   - training loop
   - validation-loss loop
   - generation loop
27. Added CLI toggle `--disable-tqdm` to suppress progress bars when desired.
28. Validated tqdm-enabled trainer with WSL CUDA smoke run at [runs/t5_tqdm_check](runs/t5_tqdm_check).

### Current Runtime State

- Baseline CPU full run was terminated.
- Latest corrected pipeline has passed CPU and WSL CUDA smoke tests, and full WSL training is running in background.
- Current corrected output files in [runs/t5_smoke_wsl_v2](runs/t5_smoke_wsl_v2):
   - best.pt
   - last.pt
   - metrics.csv
   - config.json
   - train_skipped.csv
   - val_skipped.csv
   - best_val_samples.json
   - train_gap_audit.json
   - val_gap_audit.json

### Notes

- Dataset rows with missing keypoint folders are skipped by design at load time.
- Smoke run artifacts are in [runs/smoke_run](runs/smoke_run).
- Corrected smoke artifacts are in [runs/t5_smoke](runs/t5_smoke).
- WSL CUDA smoke artifacts are in [runs/t5_smoke_wsl](runs/t5_smoke_wsl) and [runs/t5_smoke_wsl_v2](runs/t5_smoke_wsl_v2).
- Full WSL run artifacts are being written to [runs/how2sign_t5_wsl_full](runs/how2sign_t5_wsl_full).
