#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SPLITS = {
    "train": {
        "csv": "how2sign_train.csv",
        "keypoints": "train_2D_keypoints/openpose_output/json",
        "videos": "train_rgb_front_clips/raw_videos",
    },
    "val": {
        "csv": "how2sign_val.csv",
        "keypoints": "val_2D_keypoints/openpose_output/json",
        "videos": "val_rgb_front_clips/raw_videos",
    },
    "test": {
        "csv": "how2sign_test.csv",
        "keypoints": "test_2D_keypoints/openpose_output/json",
        "videos": "test_rgb_front_clips/raw_videos",
    },
}


def read_sentence_names(tsv_path: Path) -> tuple[int, set[str], int]:
    rows = 0
    names: list[str] = []

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "SENTENCE_NAME" not in reader.fieldnames:
            raise ValueError(f"Missing SENTENCE_NAME in TSV header: {tsv_path}")

        for row in reader:
            rows += 1
            key = (row.get("SENTENCE_NAME") or "").strip()
            if key:
                names.append(key)

    unique = set(names)
    duplicate_count = len(names) - len(unique)
    return rows, unique, duplicate_count


def list_subdirs(path: Path) -> set[str]:
    return {p.name for p in path.iterdir() if p.is_dir()}


def list_mp4_stems(path: Path) -> set[str]:
    return {p.stem for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}


def sorted_sample(items: set[str], n: int) -> list[str]:
    return sorted(items)[:n]


def audit_split(root: Path, split: str, sample_size: int, skip_video: bool = False) -> dict:
    config = SPLITS[split]

    csv_path = root / config["csv"]
    kp_path = root / config["keypoints"]
    vid_path = root / config["videos"]

    for required_path in [csv_path, kp_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")
    if not skip_video and not vid_path.exists():
        raise FileNotFoundError(f"Required path not found: {vid_path}")

    row_count, csv_names, csv_duplicate_count = read_sentence_names(csv_path)
    kp_names = list_subdirs(kp_path)
    vid_names = list_mp4_stems(vid_path) if not skip_video else set()

    csv_missing_kp = csv_names - kp_names
    kp_orphans = kp_names - csv_names

    csv_missing_vid = csv_names - vid_names
    vid_orphans = vid_names - csv_names

    kp_missing_vid = kp_names - vid_names
    vid_missing_kp = vid_names - kp_names

    return {
        "split": split,
        "paths": {
            "csv": str(csv_path),
            "keypoints": str(kp_path),
            "videos": str(vid_path),
        },
        "counts": {
            "rows": row_count,
            "csv_unique": len(csv_names),
            "csv_duplicates": csv_duplicate_count,
            "kp_unique": len(kp_names),
            "vid_unique": len(vid_names),
            "csv_missing_kp": len(csv_missing_kp),
            "kp_orphans": len(kp_orphans),
            "csv_missing_vid": len(csv_missing_vid),
            "vid_orphans": len(vid_orphans),
            "kp_missing_vid": len(kp_missing_vid),
            "vid_missing_kp": len(vid_missing_kp),
        },
        "samples": {
            "csv_missing_kp": sorted_sample(csv_missing_kp, sample_size),
            "kp_orphans": sorted_sample(kp_orphans, sample_size),
            "csv_missing_vid": sorted_sample(csv_missing_vid, sample_size),
            "vid_orphans": sorted_sample(vid_orphans, sample_size),
            "kp_missing_vid": sorted_sample(kp_missing_vid, sample_size),
            "vid_missing_kp": sorted_sample(vid_missing_kp, sample_size),
        },
        "all_modalities_aligned": (
            len(csv_missing_kp) == 0
            and len(kp_orphans) == 0
            and len(csv_missing_vid) == 0
            and len(vid_orphans) == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit name alignment across annotation TSV, keypoint folders, and RGB-front videos."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root path")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size for mismatch examples")
    parser.add_argument("--skip-video", action="store_true", help="Skip video directory existence check and video-related statistics")
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Optional path to save the full JSON report",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    report = {split: audit_split(root, split, args.sample_size, skip_video=args.skip_video) for split in SPLITS}

    text = json.dumps(report, indent=2)
    print(text)

    if args.write_json is not None:
        output_path = args.write_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()