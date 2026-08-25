"""
collect_performance_sample.py
------------------------------
M5 Task 2: post-deployment model performance tracking.

Sends a small batch of held-out test images (with known true labels) to a
running deployment's /predict endpoint, compares predictions to ground truth,
and writes a small report (accuracy + per-sample results) so drift/regression
can be spot-checked without re-running the full test suite.

Run (against a locally deployed service):
    python scripts/collect_performance_sample.py \
        --host http://localhost:8000 \
        --data data/processed/test \
        --n 50
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def sample_images(data_dir: Path, n_per_class: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    samples = []
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        files = list(label_dir.glob("*.jpg"))
        rng.shuffle(files)
        for f in files[:n_per_class]:
            samples.append((f, label_dir.name))
    rng.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "test")
    parser.add_argument("--n", type=int, default=25, help="Images per class to sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "monitoring" / "performance_sample.json")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"No processed test data at {args.data}. Run src/preprocess.py first.")

    samples = sample_images(args.data, args.n, args.seed)
    if not samples:
        raise SystemExit(f"No images found under {args.data}")

    results = []
    correct = 0
    latencies_ms = []

    for path, true_label in samples:
        with open(path, "rb") as f:
            start = time.perf_counter()
            resp = requests.post(f"{args.host}/predict", files={"file": (path.name, f, "image/jpeg")}, timeout=30)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        resp.raise_for_status()
        body = resp.json()
        is_correct = body["prediction"] == true_label
        correct += is_correct
        results.append({
            "file": path.name,
            "true_label": true_label,
            "predicted": body["prediction"],
            "confidence": body["confidence"],
            "correct": is_correct,
        })

    accuracy = correct / len(results)
    report = {
        "host": args.host,
        "n_samples": len(results),
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"Sampled {len(results)} live predictions against {args.host}")
    print(f"Live accuracy: {accuracy:.2%}  |  avg latency: {report['avg_latency_ms']}ms")
    print(f"Wrote report -> {args.out}")


if __name__ == "__main__":
    main()
