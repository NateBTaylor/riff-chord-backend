# ChordMini Backend - Chord Recognition + Beat Detection
FROM python:3.10-slim

WORKDIR /app

# Install system deps + build tools for madmom
RUN apt-get update && apt-get install -y \
    curl build-essential libsndfile1-dev libsndfile1 ffmpeg git pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install base Python deps
RUN pip install --no-cache-dir --upgrade pip "setuptools==79.0.1" wheel
RUN pip install --no-cache-dir Cython>=0.29.0 numpy==1.26.4

# Install PyTorch + torchaudio CPU-only (~250MB vs ~900MB for full)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install madmom (beat detection) from git
RUN pip install --no-cache-dir git+https://github.com/CPJKU/madmom

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to save space
RUN apt-get purge -y build-essential git pkg-config && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy application code
COPY app.py app_factory.py config.py extensions.py error_handlers.py ./
COPY config/ config/
COPY services/ services/
COPY blueprints/ blueprints/
COPY models/ models/
COPY utils/ utils/
COPY compat/ compat/
RUN rm -f /app/scipy_patch.py || true

# Non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 app \
    && chown -R app:app /app
USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

ENV FLASK_ENV=production
ENV FLASK_DEBUG=False
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "600", "--worker-class", "sync", "--preload", "app:app"]
