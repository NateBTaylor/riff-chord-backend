"""
yt-dlp subprocess runner mirroring the Keys project's approach.

Why subprocess instead of `import yt_dlp`:
- bgutil-ytdlp-pot-provider auto-registers when yt_dlp imports.
  Subprocess invocations get a fresh import each call which has been
  more reliable than long-lived library state on the same Gunicorn
  worker.
- yt-dlp itself supports a richer set of CLI args that aren't always
  cleanly mapped onto YoutubeDL() options.
- Matches what's already proven working in production (Keys server).

Strategy:
1. Iterate through a list of player_client values.
2. For each, run yt-dlp with --print-json so we can extract title/artist
   metadata from stdout.
3. First client that successfully downloads wins; return immediately.
4. Output is written to a temp directory under output_template = audio.%(ext)s
   so we can read it back by listing the directory.

Each invocation gets:
  - --cookies <path>  (when YOUTUBE_COOKIES_TXT env var is set)
  - --user-agent <iPhone Safari>
  - -f bestaudio[ext=m4a]/bestaudio/best
  - --print-json (metadata)
  - --no-progress --no-playlist --no-check-certificate
  - --retries 3
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from utils.logging import log_info, log_error


IPHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"
)

# Per-host retry sequences. Each attempt's extra_args are appended to the
# base yt-dlp invocation.
_YOUTUBE_ATTEMPTS = [
    ("default",      []),
    ("ios",          ["--extractor-args", "youtube:player_client=ios"]),
    ("android",      ["--extractor-args", "youtube:player_client=android"]),
    ("tv_embedded",  ["--extractor-args", "youtube:player_client=tv_embedded"]),
    ("mediaconnect", ["--extractor-args", "youtube:player_client=mediaconnect"]),
    ("web_safari",   ["--extractor-args", "youtube:player_client=web_safari"]),
]

_TIKTOK_ATTEMPTS = [
    ("tiktok:default",       []),
    ("tiktok:app_trill",     ["--extractor-args", "tiktok:app_name=trill"]),
    ("tiktok:app_musically", ["--extractor-args", "tiktok:app_name=musical_ly"]),
    ("tiktok:api22",         ["--extractor-args",
                              "tiktok:api_hostname=api22-normal-c-useast1a.tiktokv.com"]),
    ("tiktok:api16",         ["--extractor-args",
                              "tiktok:api_hostname=api16-normal-c-useast1a.tiktokv.com"]),
]


@dataclass
class DownloadResult:
    file_path: str          # absolute path to the downloaded audio file
    extension: str          # e.g. "m4a", "mp3", "webm"
    title: Optional[str]
    artist: Optional[str]
    thumbnail_url: Optional[str]
    canonical_url: Optional[str]
    client_used: str        # which player_client succeeded


class YtDlpError(Exception):
    """Raised when every retry attempt for a URL has failed."""


def _ensure_cookies_path() -> Optional[str]:
    """Materialize YOUTUBE_COOKIES_TXT env var as /tmp/youtube_cookies.txt.
    Returns the path or None if env var unset / content looks bogus."""
    content = os.environ.get("YOUTUBE_COOKIES_TXT")
    if not content:
        return None
    content = content.replace("\\n", "\n")

    # Sanity check: must contain at least one Netscape-format row.
    looks_valid = False
    for line in content.splitlines()[:60]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols = s.split("\t")
        if len(cols) >= 7 and (cols[0].startswith(".") or "." in cols[0]):
            looks_valid = True
            break
    if not looks_valid:
        log_error("[yt-dlp] YOUTUBE_COOKIES_TXT doesn't look like Netscape cookies. "
                  f"First 160 chars: {content[:160]!r}")
        return None

    path = "/tmp/youtube_cookies.txt"
    try:
        with open(path, "w") as f:
            f.write(content)
        return path
    except Exception as e:
        log_error(f"[yt-dlp] failed to write cookies file: {e}")
        return None


def _is_retryable_error(stderr: str) -> bool:
    """yt-dlp stderr indicates a problem that switching player_client
    might fix (auth, bot, format unavailable, unable to extract)."""
    s = stderr.lower()
    return any(token in s for token in (
        "sign in", "not a bot", "confirm you", "cookies",
        "forbidden", "http error 403", "video not available",
        "status code", "unable to extract",
        "requested format is not available",
        "no video formats found",
    ))


def _strip_hashtags_and_mentions(title: str) -> str:
    """Clean a title — drop hashtags, @mentions, URLs, collapse whitespace."""
    cleaned = re.sub(r"#\w+", "", title)
    cleaned = re.sub(r"@[\w.]+", "", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_metadata(stdout: str) -> dict:
    """yt-dlp --print-json emits one JSON object per video. Take the last
    full JSON line and extract title, artist, thumbnail."""
    info = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                info = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    raw_title = info.get("title") or info.get("description") or ""
    title = _strip_hashtags_and_mentions(raw_title) if raw_title else None

    artist = (info.get("creator") or info.get("artist") or info.get("uploader")
              or info.get("channel") or info.get("uploader_id"))
    if isinstance(artist, str):
        artist = artist.strip() or None

    thumbnail_url = info.get("thumbnail")
    # Some videos give thumbnails as a list of {url, width, height}
    if not thumbnail_url and isinstance(info.get("thumbnails"), list):
        candidates = [t for t in info["thumbnails"]
                      if isinstance(t, dict) and t.get("url")]
        # Prefer ones <= 1080px wide
        candidates.sort(
            key=lambda t: (t.get("width") or 0) if (t.get("width") or 0) <= 1080
                          else -(t.get("width") or 0),
            reverse=True,
        )
        if candidates:
            thumbnail_url = candidates[0].get("url")

    return {
        "title": title,
        "artist": artist,
        "thumbnail_url": thumbnail_url,
        "canonical_url": info.get("webpage_url"),
    }


def _run_yt_dlp(base_args: list[str], extra_args: list[str],
                source_url: str, timeout: int = 120) -> str:
    """Spawn `python3 -m yt_dlp ARGS source_url` and return stdout.
    Raises YtDlpError on non-zero exit with a trimmed stderr summary."""
    cmd = ["python3", "-m", "yt_dlp", *base_args, *extra_args, source_url]
    log_info(f"[yt-dlp] running {' '.join(cmd[:8])}...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise YtDlpError(f"yt-dlp timed out after {timeout}s")

    if proc.returncode != 0:
        # Echo full stderr to backend logs so we can see plugin output;
        # the raised exception carries just the trimmed tail.
        log_info(f"[yt-dlp stderr] {proc.stderr[-1500:]}")
        tail = " | ".join(
            line for line in proc.stderr.splitlines()[-4:] if line.strip()
        )[:400]
        raise YtDlpError(f"yt-dlp exit {proc.returncode}: {tail or 'no stderr'}")
    return proc.stdout


def _youtube_video_id(url: str) -> Optional[str]:
    """Extract the 11-char YouTube video ID from any common URL shape."""
    from urllib.parse import urlparse, parse_qs
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if "youtu.be" in host:
        return p.path.lstrip("/").split("/")[0] or None
    if "youtube.com" in host:
        q = parse_qs(p.query)
        if "v" in q and q["v"]:
            return q["v"][0]
        m = re.match(r"^/(?:shorts|embed|v)/([A-Za-z0-9_-]{11})", p.path)
        if m:
            return m.group(1)
    return None


def _download_youtube_via_rapidapi(source_url: str, output_dir: str) -> Optional[DownloadResult]:
    """RapidAPI YouTube-to-MP3 fast path. Requires RAPIDAPI_KEY env var.
    Returns None if the key isn't set or the API call fails — caller
    falls through to yt-dlp / proxy instances.

    Default host: youtube-mp36.p.rapidapi.com (a "YouTube to MP3"
    service on RapidAPI, generous free tier ~500 calls/month).
    Override by setting RAPIDAPI_YT_HOST.
    """
    import requests
    api_key = os.environ.get("RAPIDAPI_KEY")
    if not api_key:
        return None

    host = os.environ.get("RAPIDAPI_YT_HOST", "youtube-mp36.p.rapidapi.com")
    video_id = _youtube_video_id(source_url)
    if not video_id:
        log_info("[rapidapi] couldn't parse YouTube video ID")
        return None

    api_url = f"https://{host}/dl?id={video_id}"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
    }
    try:
        resp = requests.get(api_url, headers=headers, timeout=15)
    except Exception as e:
        log_info(f"[rapidapi] request failed: {e}")
        return None

    if resp.status_code != 200:
        log_info(f"[rapidapi] HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    try:
        body = resp.json()
    except Exception:
        return None

    # Most YouTube-MP3 RapidAPIs return one of:
    #   { "status": "ok",  "link": "https://...", "title": "..." }
    #   { "status": "processing", ... }  — they're transcoding, retry in a few sec
    #   { "status": "fail", "msg": "..." }
    status = (body.get("status") or "").lower()
    if status != "ok":
        # Some endpoints take a few seconds to transcode — try one short retry.
        if status == "processing":
            import time as _t
            _t.sleep(3)
            try:
                resp = requests.get(api_url, headers=headers, timeout=15)
                body = resp.json()
                status = (body.get("status") or "").lower()
            except Exception:
                pass
        if status != "ok":
            log_info(f"[rapidapi] status={status!r} msg={body.get('msg')!r}")
            return None

    media_url = body.get("link") or body.get("dlink") or body.get("url")
    if not media_url:
        log_info(f"[rapidapi] no media URL in response: {list(body.keys())}")
        return None

    out_path = os.path.join(output_dir, "audio.mp3")
    try:
        stream = requests.get(media_url, timeout=60, stream=True)
        if stream.status_code != 200:
            log_info(f"[rapidapi] media HTTP {stream.status_code}")
            return None
        with open(out_path, "wb") as f:
            for chunk in stream.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        log_info(f"[rapidapi] media download failed: {e}")
        return None

    raw_title = (body.get("title") or "").strip()
    title = _strip_hashtags_and_mentions(raw_title) if raw_title else None

    log_info(f"[rapidapi] success — mp3 {os.path.getsize(out_path) // 1024}KB, "
             f"host={host}, title={title!r}")
    return DownloadResult(
        file_path=out_path,
        extension="mp3",
        title=title,
        artist=None,  # most YT-MP3 APIs don't return uploader; oEmbed handles this
        thumbnail_url=None,
        canonical_url=source_url,
        client_used=f"rapidapi:{host.split('.')[0]}",
    )


def _download_tiktok_via_tikwm(source_url: str, output_dir: str) -> Optional[DownloadResult]:
    """Public TikTok downloader API. Returns a direct audio URL in one
    request, avoiding yt-dlp's broken TikTok extractor entirely (which
    requires `curl-cffi` impersonation we don't have installed).

    Returns None on any failure so the caller can fall through to yt-dlp.
    """
    import requests
    api_url = f"https://www.tikwm.com/api/?url={requests.utils.quote(source_url, safe='')}&hd=1"
    headers = {
        "User-Agent": IPHONE_UA,
        "Accept": "application/json",
    }
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
    except Exception as e:
        log_info(f"[tikwm] request failed: {e}")
        return None
    if resp.status_code != 200:
        log_info(f"[tikwm] HTTP {resp.status_code}")
        return None
    try:
        body = resp.json()
    except Exception:
        return None
    if body.get("code") != 0 or not body.get("data"):
        log_info(f"[tikwm] error: {body.get('msg', 'unknown')}")
        return None

    data = body["data"]
    # Prefer audio-only `music` URL, fall back to no-watermark video `play`
    media_url = data.get("music") or data.get("play")
    if not media_url:
        return None
    if not media_url.startswith("http"):
        media_url = "https://www.tikwm.com" + media_url
    is_audio = (data.get("music") and media_url == data["music"])
    ext = "mp3" if is_audio else "mp4"

    # Stream the audio bytes to a temp file.
    out_path = os.path.join(output_dir, f"audio.{ext}")
    try:
        stream = requests.get(media_url, headers=headers, timeout=60, stream=True)
        if stream.status_code != 200:
            log_info(f"[tikwm] media HTTP {stream.status_code}")
            return None
        with open(out_path, "wb") as f:
            for chunk in stream.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        log_info(f"[tikwm] media download failed: {e}")
        return None

    title_raw = (data.get("title") or "").strip()
    title = _strip_hashtags_and_mentions(title_raw) if title_raw else None
    author = data.get("author") or {}
    artist = author.get("unique_id") or author.get("nickname")
    thumbnail = data.get("cover") or data.get("origin_cover")

    log_info(f"[tikwm] success — {ext} {os.path.getsize(out_path) // 1024}KB, "
             f"title={title!r}")
    return DownloadResult(
        file_path=out_path,
        extension=ext,
        title=title,
        artist=artist,
        thumbnail_url=thumbnail,
        canonical_url=source_url,
        client_used="tikwm",
    )


def download(source_url: str, output_dir: str, timeout: int = 120) -> DownloadResult:
    """Download the best audio stream from any supported source URL.
    Tries multiple player_client values until one succeeds.

    Returns a DownloadResult with the local file path + metadata.
    Raises YtDlpError if every attempt fails.
    """
    host = (urlparse(source_url).hostname or "").lower()
    is_youtube = host in {"youtube.com", "www.youtube.com",
                          "m.youtube.com", "youtu.be"}
    is_tiktok = "tiktok.com" in host

    # Fast path: hit tikwm.com first for TikTok URLs. It's been reliable
    # while yt-dlp's TikTok extractor has been broken (needs curl-cffi
    # for TLS impersonation we don't have).
    if is_tiktok:
        result = _download_tiktok_via_tikwm(source_url, output_dir)
        if result is not None:
            return result
        log_info("[tikwm] fell through, trying yt-dlp...")

    # Fast path: RapidAPI YouTube-to-MP3 service first for YouTube URLs.
    # When RAPIDAPI_KEY is set, this skips the entire yt-dlp/proxy chain
    # and returns audio in 3-8s instead of the 30-90s the free path takes.
    # Falls through to yt-dlp + Invidious chain if the key isn't set or
    # the API call fails.
    if is_youtube:
        result = _download_youtube_via_rapidapi(source_url, output_dir)
        if result is not None:
            return result
        log_info("[rapidapi] fell through, trying yt-dlp...")

    if is_youtube:
        attempts = _YOUTUBE_ATTEMPTS
    elif is_tiktok:
        attempts = _TIKTOK_ATTEMPTS
    else:
        attempts = [("default", [])]

    output_template = os.path.join(output_dir, "audio.%(ext)s")
    base_args = [
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "--no-playlist",
        "--no-check-certificate",
        "--no-progress",
        "--print-json",
        "--retries", "3",
        "--user-agent", IPHONE_UA,
        "-o", output_template,
    ]

    cookies_path = _ensure_cookies_path()
    if cookies_path:
        base_args.extend(["--cookies", cookies_path])
        log_info("[yt-dlp] cookies file attached")

    last_error: Optional[Exception] = None
    for label, extra_args in attempts:
        t0 = time.time()
        try:
            stdout = _run_yt_dlp(base_args, extra_args, source_url, timeout=timeout)
            meta = _parse_metadata(stdout)
            # Find the produced file
            audio_files = [f for f in os.listdir(output_dir)
                           if f.startswith("audio.") and not f.endswith(".part")]
            if not audio_files:
                raise YtDlpError("yt-dlp succeeded but produced no output file")
            audio_path = os.path.join(output_dir, audio_files[0])
            ext = Path(audio_files[0]).suffix.lstrip(".").lower() or "audio"

            log_info(f"[yt-dlp] {label} succeeded in {time.time() - t0:.1f}s — "
                     f"{Path(audio_path).name}, "
                     f"{os.path.getsize(audio_path) // 1024}KB, "
                     f"title={meta.get('title')!r}")
            return DownloadResult(
                file_path=audio_path,
                extension=ext,
                title=meta.get("title"),
                artist=meta.get("artist"),
                thumbnail_url=meta.get("thumbnail_url"),
                canonical_url=meta.get("canonical_url"),
                client_used=label,
            )
        except YtDlpError as e:
            last_error = e
            log_info(f"[yt-dlp] {label} failed in {time.time() - t0:.1f}s: {str(e)[:200]}")
            # If error doesn't look like a per-client issue, other clients
            # won't help — bail early.
            if not _is_retryable_error(str(e)):
                break
            continue
        except Exception as e:
            last_error = e
            log_error(f"[yt-dlp] {label} unexpected error: {e}")
            continue

    raise YtDlpError(
        f"All {len(attempts)} attempts failed for {source_url[:120]}. "
        f"Last error: {last_error}"
    )
