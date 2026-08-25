"""Tests for the model-utility/inference function and the FastAPI serving layer (M3 Task 1)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "cats_dogs_cnn.keras"
FIXTURES = ROOT / "tests" / "fixtures" / "sample_images"


def test_model_artifact_exists():
    assert MODEL.exists(), "Run `python src/preprocess.py && python src/train.py` to produce the model."


# ------------------------------------------------------------- inference-utility test
def test_predict_array_returns_valid_prediction():
    """Unit test for the model utility/inference function `_predict_array`."""
    import api

    api.load_model()
    dummy = np.random.rand(api._meta.get("img_size", 224), api._meta.get("img_size", 224), 3).astype(np.float32)

    result = api._predict_array(dummy)

    assert result.prediction in ("cat", "dog")
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6


# ------------------------------------------------------------------- api tests
@pytest.fixture(scope="module")
def client():
    from api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_image(client):
    sample = next((FIXTURES / "Cat").glob("*.jpg"))
    with open(sample, "rb") as f:
        r = client.post("/predict", files={"file": ("cat.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("cat", "dog")
    assert 0 <= body["confidence"] <= 1
    assert set(body["probabilities"].keys()) == {"cat", "dog"}


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert r.status_code == 400


def test_predict_rejects_corrupt_image_bytes(client):
    r = client.post("/predict", files={"file": ("bad.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400
