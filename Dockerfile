FROM python:3.11-slim

# --- runtime hygiene ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- install only serving dependencies (smaller image than full requirements) ---
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# --- copy source and the trained artifact ---
COPY src/api.py src/preprocess.py ./src/
COPY models/cats_dogs_cnn.keras models/model_metadata.json ./models/

# --- run as non-root ---
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# --- container-level health check hits the API's /health probe ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
