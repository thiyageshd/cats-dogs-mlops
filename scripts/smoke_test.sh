#!/usr/bin/env bash
# M4 Task 3: post-deploy smoke test. Hits /health, then a real /predict call.
# Exits non-zero (failing the CD job) if either check fails.
set -euo pipefail

HOST="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-$(dirname "$0")/../tests/fixtures/sample_images/Cat/$(ls "$(dirname "$0")/../tests/fixtures/sample_images/Cat" | head -1)}"
MAX_WAIT="${MAX_WAIT:-60}"

echo "== Smoke test: waiting for $HOST/health =="
elapsed=0
until curl -sf "$HOST/health" > /tmp/smoke_health.json; do
  if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "FAIL: /health did not become ready within ${MAX_WAIT}s"
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

if ! grep -q '"status":"ok"' /tmp/smoke_health.json && ! grep -q '"status": "ok"' /tmp/smoke_health.json; then
  echo "FAIL: /health did not report status=ok"
  cat /tmp/smoke_health.json
  exit 1
fi
echo "OK: /health ready after ${elapsed}s"

echo "== Smoke test: POST /predict with $SAMPLE_IMAGE =="
http_code=$(curl -s -o /tmp/smoke_predict.json -w "%{http_code}" \
  -F "file=@${SAMPLE_IMAGE};type=image/jpeg" "$HOST/predict")

if [ "$http_code" != "200" ]; then
  echo "FAIL: /predict returned HTTP $http_code"
  cat /tmp/smoke_predict.json
  exit 1
fi

echo "OK: /predict -> $(cat /tmp/smoke_predict.json)"
echo "== Smoke test passed =="
