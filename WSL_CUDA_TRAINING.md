# WSL CUDA Training Guide

This guide runs the corrected trainer on GPU through WSL.

## 1) Verified Environment

Confirmed in this workspace:

- WSL GPU visibility: `nvidia-smi` works
- WSL PyTorch CUDA check: `True`
- Device name: `NVIDIA GeForce RTX 3050 Laptop GPU`

## 2) WSL Environment (already created here)

Linux venv path:

- `.venv_wsl`

Installed packages in WSL venv:

- torch (CUDA build)
- transformers
- sacrebleu
- sentencepiece
- nltk

## 3) Quick GPU Validation Command

```powershell
wsl --cd /mnt/c/Users/jnami/Downloads/Compressed/kartik --exec ./.venv_wsl/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 4) GPU Smoke Test (already validated)

```powershell
wsl --cd /mnt/c/Users/jnami/Downloads/Compressed/kartik --exec ./.venv_wsl/bin/python train_how2sign_t5.py --data-root . --epochs 1 --batch-size 2 --max-train-samples 8 --max-val-samples 4 --max-frames 16 --num-workers 0 --eval-max-batches 2 --save-dir runs/t5_smoke_wsl --augment --ctc-weight 0.1 --signcl-weight 0.01 --gap-audit-max-clips 4 --amp
```

Artifacts from the validated run:

- [runs/t5_smoke_wsl/best.pt](runs/t5_smoke_wsl/best.pt)
- [runs/t5_smoke_wsl/last.pt](runs/t5_smoke_wsl/last.pt)
- [runs/t5_smoke_wsl/metrics.csv](runs/t5_smoke_wsl/metrics.csv)
- [runs/t5_smoke_wsl/train_gap_audit.json](runs/t5_smoke_wsl/train_gap_audit.json)
- [runs/t5_smoke_wsl/val_gap_audit.json](runs/t5_smoke_wsl/val_gap_audit.json)

## 5) Recommended Full Run (WSL CUDA)

This workspace now includes a background launcher script:

- [launch_wsl_full.sh](launch_wsl_full.sh)

Run it from Windows PowerShell:

```powershell
wsl -e bash /mnt/c/Users/jnami/Downloads/Compressed/kartik/launch_wsl_full.sh
```

If you want a direct one-shot command instead of the launcher script:

```powershell
wsl --cd /mnt/c/Users/jnami/Downloads/Compressed/kartik --exec ./.venv_wsl/bin/python train_how2sign_t5.py --data-root . --pretrained-model t5-small --epochs 20 --batch-size 2 --grad-accum-steps 16 --num-workers 2 --max-frames 96 --temporal-stride 2 --max-target-tokens 64 --max-gen-tokens 64 --num-beams 4 --lr 3e-5 --weight-decay 1e-4 --warmup-ratio 0.06 --grad-clip 1.0 --min-conf 0.05 --interpolation-gap 3 --augment --flip-prob 0.5 --scale-jitter 0.1 --signcl-weight 0.01 --signcl-temperature 0.07 --signcl-neg-distance 20 --signcl-max-anchors 32 --signcl-max-negatives 64 --ctc-weight 0.05 --gap-audit-max-clips 1000 --eval-test-on-best --save-dir runs/how2sign_t5_wsl_full --amp --log-every 200
```

Notes:

- Effective batch size is `batch_size * grad_accum_steps`.
- With the command above: effective batch size = `2 * 16 = 32`.
- If VRAM allows, increase `batch-size` and/or `grad-accum-steps` toward larger effective batches for stronger SignCL behavior.

## 6) Monitoring

Live full-run files in this workspace:

- [runs/how2sign_t5_wsl_full/train.log](runs/how2sign_t5_wsl_full/train.log)
- [runs/how2sign_t5_wsl_full/train.pid](runs/how2sign_t5_wsl_full/train.pid)

Tail logs from PowerShell:

```powershell
wsl -e bash -lc 'tail -n 60 /mnt/c/Users/jnami/Downloads/Compressed/kartik/runs/how2sign_t5_wsl_full/train.log'
```
