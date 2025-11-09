FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install deps (ffmpeg required for Whisper)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY backend files first (skip notebooks, audio, frontend, etc)
COPY requirements.txt .

# Install dependencies without caching
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Then copy your backend code
COPY . .

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
