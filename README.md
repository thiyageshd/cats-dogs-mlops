# Cats vs Dogs — MLOps Pipeline

End-to-end MLOps solution for **MLOps Assignment 02 (AIMLCZG523)**: a binary
image classifier (cat vs dog) for a pet-adoption platform, packaged, containerized,
and deployed with automated CI/CD.

**Dataset:** Kaggle "Cats and Dogs" binary classification dataset (Microsoft's
Cats vs Dogs, 25,000 images) — preprocessed to 224×224 RGB, split 80/10/10
train/val/test, with corrupt files filtered out (a handful of known-bad files
in this exact Kaggle set).

---

## Repository structure

```
cats-dogs-mlops/
├── data/
│   ├── raw/PetImages/{Cat,Dog}/        # raw Kaggle download (gitignored)
│   └── processed/{train,val,test}/     # resized + split (DVC-tracked, see below)
├── src/
│   ├── preprocess.py    # M1: filter, resize, split — shared preprocessing used by training + serving
│   ├── train.py         # M1: baseline CNN + MLflow experiment tracking
│   └── api.py            # M2: FastAPI /health + /predict
├── tests/                 # M3: pytest unit tests (+ committed fixture images for CI)
├── scripts/
│   ├── smoke_test.sh                 # M4: post-deploy health + predict check
│   └── collect_performance_sample.py # M5: live-prediction accuracy sample
├── models/                # trained model.keras + metadata (committed — small artifact)
├── monitoring/            # Prometheus scrape config + Grafana provisioning
├── .github/workflows/     # CI (build/test/push) + CD (self-hosted deploy/smoke-test)
├── Dockerfile, docker-compose.yml
├── requirements.txt, requirements-serve.txt
└── README.md, memory.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

Download the Kaggle "Cats and Dogs" dataset and place it at
`data/raw/PetImages/{Cat,Dog}/*.jpg` (25,000 JPEGs, ~858MB — too large to
commit; this step is manual/local).

## Preprocess

```bash
python src/preprocess.py
```

Filters corrupt files, resizes to 224×224, and writes an 80/10/10 train/val/test
split to `data/processed/`.

## Data versioning (DVC)

`data/processed/` is tracked with [DVC](https://dvc.org) against a local remote
(a plain directory outside the repo — no cloud account needed):

```bash
dvc remote add -d local-storage ~/dvc-storage/cats-dogs-mlops   # one-time
dvc add data/processed
git add data/processed.dvc .gitignore
dvc push
```

To reproduce the dataset on another machine (after `git clone` + `dvc pull`):

```bash
dvc pull   # restores data/processed/ from the local remote
```

## Train

```bash
python src/train.py                 # full processed dataset, real artifact
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs -> http://127.0.0.1:5000
```

Saves `models/cats_dogs_cnn.keras` + `models/model_metadata.json`.

## Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
# docs: http://localhost:8000/docs
curl -F "file=@data/processed/test/cat/<some_file>.jpg" http://localhost:8000/predict
```

## Containerize

```bash
docker build -t cats-dogs-mlops:latest .
docker run -p 8000:8000 cats-dogs-mlops:latest
curl -F "file=@sample.jpg" http://localhost:8000/predict
```

## Full local stack (API + Prometheus + Grafana)

```bash
docker compose up --build
```

- API → http://localhost:8000/docs
- Metrics → http://localhost:8000/metrics
- Prometheus → http://localhost:9090
- Grafana → http://localhost:3000 (admin / admin)

## CI/CD

- **CI** (`.github/workflows/ci-cd.yml`, GitHub-hosted runner): lint, pytest,
  a fast smoke-train against committed fixture images, Docker build, and (on
  push to `main` only) push to GHCR at `ghcr.io/thiyageshd/cats-dogs-mlops`.
- **CD** (same workflow, self-hosted runner, `push`-to-`main` only — never on
  pull requests, so a stranger's fork PR can never run on this machine): pulls
  the new image, restarts the local `docker compose` stack, then runs
  `scripts/smoke_test.sh` against the live `/health` and `/predict` endpoints.
  The pipeline fails if the smoke test fails.

## Post-deploy performance tracking

```bash
python scripts/collect_performance_sample.py --host http://localhost:8000 --n 25
```

Sends held-out test images with known labels to the live endpoint and reports
live accuracy + latency to `monitoring/performance_sample.json`.

---

## Build phases

| Module | Task | Status |
|--------|------|--------|
| M1 | Data/code versioning (Git + DVC), baseline CNN, MLflow tracking | ✅ done |
| M2 | FastAPI inference service, requirements pinning, Dockerfile | ✅ done |
| M3 | Unit tests, CI (lint/test/build/push to GHCR) | ✅ done |
| M4 | Docker Compose deployment, self-hosted CD, smoke tests | ✅ done |
| M5 | Prometheus/Grafana monitoring, post-deploy performance sample | ✅ done |
