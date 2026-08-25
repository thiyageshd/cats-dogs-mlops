"""
api.py
------
M2 Task 1: model-serving API.

Loads the CNN saved by src/train.py and exposes:
    GET  /health   -> liveness + model metadata
    POST /predict  -> single-image cat/dog classification (multipart file upload)
    GET  /          -> redirect to interactive docs

Uses the same `load_image_array` preprocessing function as training (see
preprocess.py), so there is no train/serve skew in how an image becomes a
model input.

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    # interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Make `from preprocess import ...` resolve regardless of how this module is
# invoked (uvicorn CLI, Docker, pytest) rather than relying on cwd being on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_image_array  # noqa: E402

# ------------------------------------------------------------------ logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cats-dogs-api")

# ------------------------------------------------------------------ artifacts
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cats_dogs_cnn.keras"
META_PATH = ROOT / "models" / "model_metadata.json"

_model = None
_meta: dict = {}


def load_model():
    """Load the trained CNN and metadata once at startup."""
    global _model, _meta
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python src/preprocess.py` then `python src/train.py` first."
        )
    # Imported lazily: keeps `python src/preprocess.py` (used by CI's data step)
    # from requiring tensorflow to be importable in every context.
    import tensorflow as tf

    _model = tf.keras.models.load_model(MODEL_PATH)
    if META_PATH.exists():
        _meta = json.loads(META_PATH.read_text())
    logger.info("Model loaded: classes=%s", _meta.get("class_names", "unknown"))
    return _model


# ------------------------------------------------------------------ schema
class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class label: 'cat' or 'dog'")
    probabilities: dict[str, float] = Field(..., description="P(cat), P(dog)")
    confidence: float = Field(..., description="Probability of the predicted class")


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="Predicts whether an uploaded image is a cat or a dog. "
                "Serves the CNN trained in the MLOps pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Prometheus metrics: exposes /metrics with request counts, latency, etc.
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus instrumentation enabled at /metrics")
except ImportError:  # pragma: no cover
    logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Liveness probe + model info (used by the deploy smoke test)."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "class_names": _meta.get("class_names", []),
        "metrics": _meta.get("metrics", {}),
    }


def _predict_array(arr: np.ndarray) -> PredictionResponse:
    class_names = _meta.get("class_names", ["cat", "dog"])
    batch = np.expand_dims(arr, axis=0)
    p_dog = float(_model.predict(batch, verbose=0)[0, 0])
    p_cat = 1.0 - p_dog
    label = class_names[1] if p_dog >= 0.5 else class_names[0]
    confidence = p_dog if p_dog >= 0.5 else p_cat
    return PredictionResponse(
        prediction=label,
        probabilities={class_names[0]: round(p_cat, 4), class_names[1]: round(p_dog, 4)},
        confidence=round(confidence, 4),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image upload, got {file.content_type!r}")
    try:
        img_size = _meta.get("img_size", 224)
        raw = await file.read()
        arr = load_image_array(raw, size=img_size)
        start = time.perf_counter()
        result = _predict_array(arr)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict filename=%s -> %s (confidence=%.3f, %.1fms)",
            file.filename, result.prediction, result.confidence, elapsed_ms,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
