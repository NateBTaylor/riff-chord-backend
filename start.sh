#!/usr/bin/env bash
# Container entrypoint: launch the bgutil PoT server in the background,
# then start Gunicorn in the foreground. yt-dlp's bgutil HTTP plugin
# talks to the PoT server on 127.0.0.1:4416 to mint YouTube Proof-of-
# Origin tokens — without those, YouTube returns only thumbnail formats
# regardless of cookies / player_client choice.
set -euo pipefail

# Find the server binary installed by `npm install -g bgutil-ytdlp-pot-provider`.
# Different npm versions / install paths put it in different places, so try
# both PATH and the npm-global bin dir.
POT_BIN="$(command -v bgutil-pot-server || true)"
if [[ -z "$POT_BIN" ]]; then
    POT_BIN="$(npm prefix -g 2>/dev/null)/bin/bgutil-pot-server"
fi

if [[ -x "$POT_BIN" ]]; then
    echo "[start] Launching PoT server: $POT_BIN --port 4416"
    # Background, redirect output so it doesn't drown the gunicorn logs.
    "$POT_BIN" --port 4416 > /tmp/bgutil-pot.log 2>&1 &
    POT_PID=$!

    # Give it a couple seconds to bind the port. yt-dlp's first ping
    # otherwise races and fails on the very first request.
    for _ in 1 2 3 4 5; do
        if curl -fsS http://127.0.0.1:4416/ping > /dev/null 2>&1; then
            echo "[start] PoT server ready (pid $POT_PID)"
            break
        fi
        sleep 0.5
    done
else
    echo "[start] WARNING: bgutil-pot-server binary not found. YouTube downloads"
    echo "[start]          will fail until it's installed via 'npm install -g"
    echo "[start]          bgutil-ytdlp-pot-provider' in the Dockerfile."
fi

echo "[start] Launching Gunicorn..."
exec gunicorn \
    --bind 0.0.0.0:8080 \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --worker-class gthread \
    --preload \
    app:app
