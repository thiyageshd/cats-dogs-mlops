# Project memory — Cats vs Dogs MLOps (Assignment 2)

## 1. What this is

MLOps Assignment 02 (AIMLCZG523, 50 marks): end-to-end pipeline for a binary
image classifier (cat vs dog) — model dev + tracking (M1), packaging +
containerization (M2), CI (M3), CD (M4), monitoring + final submission (M5).
Brief: `../Assignment 2.pdf`.

Sibling project [[assignment-1-heart-disease-mlops]] (if that memory exists)
covers the same M1–M5 shape on tabular data; this one is image classification
and adds two things Assignment 1 didn't need: DVC dataset versioning and a
CI/CD pipeline that actually auto-redeploys on push to `main`.

## 2. Dataset

Full Kaggle "Cats and Dogs" (Microsoft) dataset, user-downloaded to
`data/raw/PetImages/{Cat,Dog}/` — 12,499 images per class, ~858MB total.
**Not committed to git** (too large, gitignored) — regenerate by downloading
from Kaggle and placing at that path.

This exact dataset is known to contain a handful of corrupt/truncated JPEGs
(e.g. the infamous `Cat/666.jpg`). `src/preprocess.py::is_valid_image` filters
these out via `PIL.Image.verify()` + a forced full decode before anything is
resized — don't skip this check if touching preprocessing.

`data/processed/{train,val,test}/{cat,dog}/` (resized 224×224 JPEGs, 80/10/10
split) is DVC-tracked with a **local** remote at `~/dvc-storage/cats-dogs-mlops`
(plain directory, no cloud account). `data/processed.dvc` is the only
git-tracked pointer.

## 3. Two-tier training (important — don't "fix" this)

`src/train.py` takes `--sample-size`/`SAMPLE_SIZE` and `--epochs`/`EPOCHS`, and
`--model-out`/`--meta-out` to redirect where the artifact is saved.

- **CI** runs a 1-epoch smoke pass against `tests/fixtures/sample_images/`
  (~40 tiny committed images, not the real 858MB dataset) and writes to
  `/tmp/ci_model.keras` — this is only there to prove the pipeline runs
  end-to-end; it must never overwrite `models/cats_dogs_cnn.keras`.
- **The real artifact** (`models/cats_dogs_cnn.keras`, committed to git — it's
  small, a few MB) comes from a local run against the full/larger
  `data/processed/` set. If accuracy looks suspiciously bad, check which one
  is actually loaded before assuming the model is broken.

## 4. Registry, deployment, CD

- Registry: GHCR, `ghcr.io/thiyageshd/cats-dogs-mlops` — pushed via
  `GITHUB_TOKEN` in Actions, no extra secret.
- Deployment target: Docker Compose only (not K8s — Assignment 1 already
  covered K8s; Compose is simpler for the automated pull-and-restart CD loop).
- CD automation: a **self-hosted GitHub Actions runner on this Mac**
  (per-repo, background launchd service) runs the `deploy-and-smoke-test` job
  in `.github/workflows/ci-cd.yml`. That job is gated to
  `github.event_name == 'push' && github.ref == 'refs/heads/main'` — **never**
  `pull_request`/`pull_request_target`. This is a deliberate security
  boundary: self-hosted runners on a public repo must never run code
  triggered by someone else's PR. Do not relax this gate.

## 5. Status & remaining work

**Done (code):** M1–M5 scaffolded — preprocessing, CNN training + MLflow,
FastAPI service, Dockerfile, docker-compose (+ Prometheus/Grafana), CI/CD
workflow, smoke test, post-deploy performance script, tests.

**Remaining (human-only, cannot be automated):**
- Record a screen recording under 5 minutes: code change → CI → image in
  GHCR → self-hosted CD redeploy → smoke test → live prediction.
- Zip source + config (DVC, CI/CD, Docker, deployment manifests) + trained
  model artifacts for submission (no written report required this time,
  unlike Assignment 1).

## 6. Gotchas

- `tensorflow-cpu` is **not** published for macOS — use plain `tensorflow`
  (works cross-platform; confirmed on this Mac). Don't "fix" requirements.txt
  by switching to `tensorflow-cpu`.
- `src/api.py` explicitly does `sys.path.insert(0, .../src)` before
  `from preprocess import ...` rather than relying on uvicorn's implicit cwd
  insertion (which Assignment 1's `api.py` quietly depended on) — keep this
  explicit insert if refactoring imports.
- Local dev venv lives at `cats-dogs-mlops/.venv` (project-local, gitignored),
  separate from the shared `bits/.venv` playground environment one level up.
