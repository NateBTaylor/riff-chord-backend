#!/usr/bin/env bash
# Container entrypoint: launch the bgutil PoT server in the background,
# then start Gunicorn in the foreground. yt-dlp's bgutil HTTP plugin
# talks to the PoT server on 127.0.0.1:4416 to mint YouTube Proof-of-
# Origin tokens — without those, YouTube returns only thumbnail formats
# regardless of cookies / player_client choice.
set -euo pipefail

# bgutil PoT server is built from source into /opt/bgutil-ytdlp-pot-provider/server/build/main.js
# (see Dockerfile). Launch it on 127.0.0.1:4416 — yt-dlp's bgutil HTTP
# plugin auto-discovers it there.
POT_SCRIPT="/opt/bgutil-ytdlp-pot-provider/server/build/main.js"

if [[ -f "$POT_SCRIPT" ]]; then
    echo "[start] Launching PoT server: node $POT_SCRIPT --port 4416"
    node "$POT_SCRIPT" --port 4416 > /tmp/bgutil-pot.log 2>&1 &
    POT_PID=$!

    # Give it a couple seconds to bind the port. yt-dlp's first ping
    # otherwise races and fails on the very first request.
    for _ in 1 2 3 4 5 6 7 8; do
        if curl -fsS http://127.0.0.1:4416/ping > /dev/null 2>&1; then
            echo "[start] PoT server ready (pid $POT_PID)"
            break
        fi
        sleep 0.5
    done
else
    echo "[start] WARNING: PoT server script not found at $POT_SCRIPT —"
    echo "[start]          YouTube downloads will return only thumbnails."
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
