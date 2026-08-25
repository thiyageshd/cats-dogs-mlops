"""
preprocess.py
-------------
M1 Task 1/2 prep: turns the raw Kaggle "Cats and Dogs" folder (data/raw/PetImages/
{Cat,Dog}/*.jpg) into a clean, resized, split dataset under data/processed/
{train,val,test}/{cat,dog}/.

The raw Kaggle download is known to contain a handful of corrupt/zero-byte/
non-JPEG files (famously Cat/666.jpg and a few others) that crash a naive
decoder. `is_valid_image` filters those out before anything is resized.

The same `load_image_array` function is imported by both train.py (building the
training dataset) and api.py (preprocessing an uploaded image at inference time),
so there is zero train/serve skew in how an image becomes a model input.

Run:
    python src/preprocess.py
"""

from __future__ import annotations

import io
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Kaggle's own dataset has a few truncated JPEGs; without this, PIL raises on
# them instead of decoding what bytes are present. We still filter truncated
# files out via is_valid_image, this just keeps Pillow from being needlessly
# strict on borderline-but-usable files during that check.
ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "PetImages"
PROCESSED_DIR = ROOT / "data" / "processed"
IMG_SIZE = 224
CLASSES = {"Cat": "cat", "Dog": "dog"}
RANDOM_STATE = 42


@dataclass
class CleanReport:
    total_seen: int = 0
    corrupt_skipped: int = 0
    per_class: dict = field(default_factory=dict)


def is_valid_image(path: Path) -> bool:
    """True if `path` decodes as a real image PIL can open and read pixels from."""
    try:
        with Image.open(path) as im:
            im.verify()
        # verify() can pass on some truncated files; re-open and force a full
        # decode to be sure.
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def load_image_array(source: bytes | str | Path, size: int = IMG_SIZE) -> np.ndarray:
    """Decode + resize + scale an image to a (size, size, 3) float32 array in [0, 1].

    Shared by training (reading from disk) and the API (reading an upload's bytes),
    so preprocessing is identical in both places.
    """
    if isinstance(source, (bytes, bytearray)):
        im = Image.open(io.BytesIO(source))
    else:
        im = Image.open(source)
    with im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def _split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val : n_val + n_test]
    train_idx = idx[n_val + n_test :]
    return train_idx, val_idx, test_idx


def build_dataset(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    size: int = IMG_SIZE,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = RANDOM_STATE,
    sample_size: int | None = None,
) -> CleanReport:
    """Filter, resize, split, and write the dataset. Returns a CleanReport.

    `sample_size`, if given, caps how many *valid* images per class are used
    (for fast CI/local smoke runs) — applied after shuffling, so it's a random
    subset, not just the first N files on disk.
    """
    report = CleanReport()
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for raw_name, label in CLASSES.items():
        class_dir = raw_dir / raw_name
        files = sorted(class_dir.glob("*.jpg"))
        report.total_seen += len(files)

        valid = [f for f in files if is_valid_image(f)]
        report.corrupt_skipped += len(files) - len(valid)

        rng = random.Random(seed)
        rng.shuffle(valid)
        if sample_size is not None:
            valid = valid[:sample_size]

        train_idx, val_idx, test_idx = _split_indices(len(valid), val_frac, test_frac, seed)
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        report.per_class[label] = {split: len(ids) for split, ids in splits.items()}

        for split, ids in splits.items():
            split_dir = out_dir / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in ids:
                src = valid[i]
                im = Image.open(src).convert("RGB").resize((size, size), Image.BILINEAR)
                im.save(split_dir / f"{label}_{i}.jpg", quality=90)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--size", type=int, default=IMG_SIZE)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Cap valid images used per class (fast CI smoke runs); default uses all.",
    )
    args = parser.parse_args()

    rep = build_dataset(
        raw_dir=args.raw, out_dir=args.out, size=args.size,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
        sample_size=args.sample_size,
    )
    print(f"Scanned {rep.total_seen} raw files, skipped {rep.corrupt_skipped} corrupt.")
    for label, counts in rep.per_class.items():
        print(f"  {label}: {counts}")
    print(f"Wrote processed dataset -> {args.out}")
