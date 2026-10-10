# The app in a container, for a host that has no GPU.
#   docker build -t voice-agent .
#   docker run -p 7860:7860 -e DATABASE_URL=... -e GROQ_API_KEY=... -e SARVAM_API_KEY=... -e ACCESS_CODE=... voice-agent
# The database is not in this image: DATABASE_URL points at a PostgreSQL that has pgvector.
FROM python:3.12-slim

# ffmpeg decodes what the browser records. Tesseract reads scanned pages (Hindi and Marathi packs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar \
    && rm -rf /var/lib/apt/lists/*

# Not root: the app reads files that strangers upload.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

# The CPU build of PyTorch first: the default one carries GPU libraries several gigabytes in size.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

RUN chown app /app
USER app

# The embedding model is fetched while the image is built, so the first question does not wait for a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Only the files .dockerignore lets through: the app itself, never keys, uploads or logs.
COPY --chown=app . .

# Inside a container the app listens on every address, and the host's proxy is the one hop in front of it.
ENV HOST=0.0.0.0 \
    PORT=7860 \
    PROXY_HOPS=1
EXPOSE 7860
CMD ["python", "server.py"]
