"""Unit tests for the data pre-processing functions (M3 Task 1)."""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import build_dataset, is_valid_image, load_image_array  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "sample_images"


def test_is_valid_image_accepts_real_jpeg():
    sample = next((FIXTURES / "Cat").glob("*.jpg"))
    assert is_valid_image(sample) is True


def test_is_valid_image_rejects_corrupt_file(tmp_path):
    bad = tmp_path / "not_an_image.jpg"
    bad.write_bytes(b"this is not image data")
    assert is_valid_image(bad) is False


def test_load_image_array_shape_and_range():
    sample = next((FIXTURES / "Dog").glob("*.jpg"))
    arr = load_image_array(sample, size=224)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_load_image_array_from_bytes_matches_from_path():
    sample = next((FIXTURES / "Cat").glob("*.jpg"))
    from_path = load_image_array(sample, size=64)
    from_bytes = load_image_array(sample.read_bytes(), size=64)
    assert np.allclose(from_path, from_bytes)


def test_build_dataset_splits_and_resizes(tmp_path):
    out_dir = tmp_path / "processed"
    report = build_dataset(raw_dir=FIXTURES, out_dir=out_dir, size=32, val_frac=0.2, test_frac=0.2, seed=1)

    assert report.total_seen == 40  # 20 cat + 20 dog fixtures
    assert report.corrupt_skipped == 0

    for split in ("train", "val", "test"):
        for label in ("cat", "dog"):
            split_dir = out_dir / split / label
            assert split_dir.exists()
            files = list(split_dir.glob("*.jpg"))
            assert len(files) > 0
            from PIL import Image
            with Image.open(files[0]) as im:
                assert im.size == (32, 32)

    shutil.rmtree(out_dir, ignore_errors=True)
