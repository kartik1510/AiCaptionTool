@echo off
setlocal enabledelayedexpansion
:: ============================================================
:: launch_windows.bat  --  How2Sign full training on Windows
:: GPU: CUDA 12.8 via E:\conda3\python.exe (torch 2.11+cu128)
:: No WSL required.  launch_local.sh kept as WSL backup.
:: Resumes from last.pt with a FRESH cosine LR (--reset-lr).
:: ============================================================

:: ---- Python interpreter (system conda, no venv needed) -----
set PYTHON=E:\conda3\python.exe

:: ---- Project root -------------------------------------------
set PROJECT=E:\CaptioningTool

:: ---- Dataset paths ------------------------------------------
set TRANSLATIONS=%PROJECT%\DATASET\English translation
set KEYPOINTS=%PROJECT%\DATASET\2d_keypoints

set TRAIN_TSV=%TRANSLATIONS%\how2sign_train.csv
set VAL_TSV=%TRANSLATIONS%\how2sign_val.csv
set TEST_TSV=%TRANSLATIONS%\how2sign_test.csv

set TRAIN_KP=%KEYPOINTS%\train_2D_keypoints\openpose_output\json
set VAL_KP=%KEYPOINTS%\val_2D_keypoints\openpose_output\json
set TEST_KP=%KEYPOINTS%\test_2D_keypoints\openpose_output\json

:: ---- Output directory ---------------------------------------
set SAVE_DIR=%PROJECT%\runs\how2sign_t5_full
mkdir "%SAVE_DIR%" 2>nul

:: ---- Auto-resume with fresh LR reset -----------------------
:: Resumes from last.pt if it exists, but resets the LR scheduler
:: to a fresh cosine schedule (--reset-lr) so LR is healthy from
:: epoch 3 onward. Remove --reset-lr after the first resume.
set RESUME_CKPT=%SAVE_DIR%\last.pt
set RESUME_ARG=
if exist "%RESUME_CKPT%" (
    set RESUME_ARG=--resume "%RESUME_CKPT%" --reset-lr
    echo [RESUME] Found checkpoint: %RESUME_CKPT% -- will reset LR schedule.
) else (
    echo [FRESH ] No checkpoint found -- starting from epoch 1.
)

:: ---- Verify key paths exist before launching ----------------
if not exist "%TRAIN_TSV%" ( echo [ERROR] TRAIN_TSV not found: %TRAIN_TSV% & exit /b 1 )
if not exist "%VAL_TSV%"   ( echo [ERROR] VAL_TSV not found:   %VAL_TSV%   & exit /b 1 )
if not exist "%TRAIN_KP%"  ( echo [ERROR] TRAIN_KP not found:  %TRAIN_KP%  & exit /b 1 )
if not exist "%VAL_KP%"    ( echo [ERROR] VAL_KP not found:    %VAL_KP%    & exit /b 1 )

echo [OK] All required paths found.
echo [OK] Python: %PYTHON%
echo [OK] Save dir: %SAVE_DIR%
echo.

:: ---- Launch training ----------------------------------------
"%PYTHON%" -u "%PROJECT%\train_how2sign_t5.py" ^
  --train-tsv       "%TRAIN_TSV%"                    ^
  --val-tsv         "%VAL_TSV%"                      ^
  --test-tsv        "%TEST_TSV%"                     ^
  --train-keypoints "%TRAIN_KP%"                     ^
  --val-keypoints   "%VAL_KP%"                       ^
  --test-keypoints  "%TEST_KP%"                      ^
  --pretrained-model t5-small                        ^
  --epochs 20                                        ^
  --batch-size 2                                     ^
  --grad-accum-steps 16                              ^
  --num-workers 2                                    ^
  --max-frames 96                                    ^
  --temporal-stride 2                                ^
  --max-target-tokens 64                             ^
  --max-gen-tokens 64                                ^
  --num-beams 4                                      ^
  --lr 5e-5                                          ^
  --weight-decay 1e-4                                ^
  --warmup-ratio 0.05                                ^
  --grad-clip 1.0                                    ^
  --min-conf 0.05                                    ^
  --interpolation-gap 3                              ^
  --augment                                          ^
  --flip-prob 0.5                                    ^
  --scale-jitter 0.1                                 ^
  --signcl-weight 0.01                               ^
  --signcl-temperature 0.07                          ^
  --signcl-neg-distance 20                           ^
  --signcl-max-anchors 32                            ^
  --signcl-max-negatives 64                          ^
  --ctc-weight 0.05                                  ^
  --gap-audit-max-clips 0                         ^
  --eval-test-on-best                                ^
  --amp                                              ^
  --save-dir "%SAVE_DIR%"                            ^
  --log-every 200                                    ^
  %RESUME_ARG%                                       ^
  2>&1 | "%PYTHON%" -u -c "import sys, pathlib; log=pathlib.Path(r'%SAVE_DIR%\train.log').open('a',encoding='utf-8'); [sys.stdout.write(l) or sys.stdout.flush() or log.write(l) or log.flush() for l in sys.stdin]"

endlocal
