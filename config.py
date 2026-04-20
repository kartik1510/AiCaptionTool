"""
config.py — How2Sign project path configuration
================================================
All paths on THIS machine.  Import this anywhere you need a path,
or pass the variables directly to the training CLI (see launch_local.sh).

Usage:
    from config import TRAIN_TSV, TRAIN_KP, VAL_TSV, VAL_KP, TEST_TSV, TEST_KP
"""

from pathlib import Path

# ── Root folders ─────────────────────────────────────────────────────────────
# Absolute path to the project checkout
PROJECT_ROOT = Path(r"E:\CaptioningTool")

# Where the How2Sign dataset lives
DATASET_ROOT = PROJECT_ROOT / "DATASET"

# ── CSV annotation files ──────────────────────────────────────────────────────
TRANSLATIONS_ROOT = DATASET_ROOT / "English translation"
TRAIN_TSV = TRANSLATIONS_ROOT / "how2sign_train.csv"
VAL_TSV   = TRANSLATIONS_ROOT / "how2sign_val.csv"
TEST_TSV  = TRANSLATIONS_ROOT / "how2sign_test.csv"

# ── Keypoint directories (the leaf 'json' folders that contain per-clip dirs) ─
KEYPOINTS_ROOT = DATASET_ROOT / "2d_keypoints"
TRAIN_KP = KEYPOINTS_ROOT / "train_2D_keypoints" / "openpose_output" / "json"
VAL_KP   = KEYPOINTS_ROOT / "val_2D_keypoints"   / "openpose_output" / "json"
TEST_KP  = KEYPOINTS_ROOT / "test_2D_keypoints"  / "openpose_output" / "json"

# ── Training output directory ─────────────────────────────────────────────────
SAVE_DIR = PROJECT_ROOT / "runs" / "how2sign_t5_full"


# ── Quick sanity-check (run this file directly: python config.py) ─────────────
if __name__ == "__main__":
    paths = {
        "TRAIN_TSV": TRAIN_TSV,
        "VAL_TSV":   VAL_TSV,
        "TEST_TSV":  TEST_TSV,
        "TRAIN_KP":  TRAIN_KP,
        "VAL_KP":    VAL_KP,
        "TEST_KP":   TEST_KP,
    }
    all_ok = True
    for name, p in paths.items():
        status = "[OK]     " if p.exists() else "[MISSING]"
        print(f"  {status}  {name}: {p}")
        if not p.exists():
            all_ok = False
    print()
    if all_ok:
        print("All paths verified -- ready to train!")
    else:
        print("Some paths are MISSING. Check the dataset installation.")
