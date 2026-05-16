"""
YouTube audio extraction via public Piped / Invidious instances.

Why: YouTube's bot detection blocks Railway's data-center IPs regardless
of cookies, PoT tokens, or player client choice — yt-dlp from the backend
just doesn't work for YouTube anymore. Piped and Invidious are
community-run YouTube proxy networks; instead of hitting YouTube
directly, we hit one of these and they hand us a valid stream URL.

Strategy:
  - Iterate through a list of public instances
  - For each: GET /streams/{video_id} with a short timeout
  - First instance that returns audio streams wins
  - Pick the best audio-only stream, download it, return the local file path

Reliability: any single Piped instance is unreliable (they go down,
their YouTube extractor occasionally breaks). The combined fallback
chain is much more reliable than any individual instance.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests

from utils.logging import log_info, log_error


# Ordered list of Piped instances. The Piped project keeps a list of
# active community instances at https://github.com/TeamPiped/Piped/wiki/Instances —
# this set is a curated subset known to expose the /streams endpoint
# publicly. We try them in order, failing fast on each.
PIPED_INSTANCES = [
    "https://pipedapi.kavin.rocks",
    "https://pipedapi.tokhmi.xyz",
    "https://pipedapi.r4fo.com",
    "https://api.piped.private.coffee",
    "https://pipedapi-libre.kavin.rocks",
    "https://api.piped.yt",
    "https://pipedapi.adminforge.de",
    "https://pipedapi.smnz.de",
    "https://pipedapi.in.projectsegfau.lt",
    "https://pipedapi.us.projectsegfau.lt",
]


def _video_id(url: str) -> Optional[str]:
    """Pull the 11-char YouTube video ID out of any common URL shape."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "youtu.be" in host:
        # https://youtu.be/<id>
        return parsed.path.lstrip("/").split("/")[0] or None
    if "youtube.com" in host:
        # /watch?v=<id>
        q = parse_qs(parsed.query)
        if "v" in q and q["v"]:
            return q["v"][0]
        # /shorts/<id>, /embed/<id>, /v/<id>
        m = re.match(r"^/(?:shorts|embed|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(1)
    return None


def _pick_best_audio(streams: list[dict]) -> Optional[dict]:
    """Pick the highest-bitrate audio-only stream. Prefer m4a/aac so the
    downstream ffmpeg pipeline doesn't have to transcode."""
    audio_only = [
        s for s in streams
        if (s.get("mimeType") or "").startswith("audio/")
    ]
    if not audio_only:
        return None
    # Prefer m4a-ish formats first, then by bitrate.
    def score(s):
        mime = (s.get("mimeType") or "").lower()
        is_m4a = "mp4" in mime or "aac" in mime or "m4a" in mime
        return (is_m4a, s.get("bitrate") or 0)
    return max(audio_only, key=score)


def download_audio(youtube_url: str, output_dir: str, per_request_timeout: int = 8) -> Optional[str]:
    """Try each Piped instance in turn until one returns a working audio
    stream. Downloads the chosen stream to `output_dir` and returns the
    local file path on success, or None if every instance fails.
    """
    vid = _video_id(youtube_url)
    if not vid:
        log_error(f"[Piped] Couldn't parse video ID from: {youtube_url[:120]}")
        return None

    for instance in PIPED_INSTANCES:
        endpoint = f"{instance}/streams/{vid}"
        t0 = time.time()
        try:
            resp = requests.get(endpoint, timeout=per_request_timeout)
        except Exception as e:
            log_info(f"[Piped] {instance}: {type(e).__name__}")
            continue

        if resp.status_code != 200:
            log_info(f"[Piped] {instance}: HTTP {resp.status_code}")
            continue

        try:
            data = resp.json()
        except Exception:
            log_info(f"[Piped] {instance}: non-JSON response")
            continue

        # Some instances return error markers in 200 responses.
        if data.get("error"):
            log_info(f"[Piped] {instance}: \"{data.get('error')}\"")
            continue

        audio_streams = data.get("audioStreams") or []
        best = _pick_best_audio(audio_streams)
        if not best or not best.get("url"):
            log_info(f"[Piped] {instance}: no usable audio streams "
                     f"({len(audio_streams)} listed)")
            continue

        stream_url = best["url"]
        log_info(f"[Piped] {instance}: stream picked in {time.time() - t0:.1f}s "
                 f"({best.get('quality', '?')}, {best.get('bitrate', 0)} bps)")

        # Download the audio bytes. The URL is a googlevideo.com signed CDN
        # link valid for several hours. Some instances return a URL that
        # requires going through their proxy — we follow redirects and use
        # a generic UA which both paths accept.
        try:
            stream_resp = requests.get(
                stream_url,
                timeout=60,
                stream=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                        "Version/17.0 Safari/605.1.15"
                    ),
                },
            )
        except Exception as e:
            log_info(f"[Piped] stream fetch failed via {instance}: {e}")
            continue

        if stream_resp.status_code != 200:
            log_info(f"[Piped] stream fetch HTTP {stream_resp.status_code} via {instance}")
            continue

        # Decide extension from MIME, default to m4a.
        mime = (best.get("mimeType") or "").lower()
        if "webm" in mime:
            ext = "webm"
        elif "opus" in mime:
            ext = "opus"
        else:
            ext = "m4a"

        out_path = os.path.join(output_dir, f"piped_{vid}.{ext}")
        try:
            total = 0
            with open(out_path, "wb") as f:
                for chunk in stream_resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            log_info(f"[Piped] Downloaded {total / 1024:.0f}KB to "
                     f"{os.path.basename(out_path)}")
            return out_path
        except Exception as e:
            log_error(f"[Piped] disk write failed: {e}")
            try:
                os.unlink(out_path)
            except OSError:
                pass
            continue

    log_error(f"[Piped] All {len(PIPED_INSTANCES)} instances failed for {youtube_url[:100]}")
    return None
