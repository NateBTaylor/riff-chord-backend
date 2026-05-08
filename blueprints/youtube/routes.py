"""
Audio extraction routes.

Provides an endpoint that accepts a YouTube, TikTok, Instagram, or SoundCloud
URL, extracts audio via yt-dlp, and returns the audio file to the client.

For SoundCloud, uses a direct extractor (fresh client_id + HLS via ffmpeg)
since yt-dlp's built-in SoundCloud client_id expires frequently.

The bgutil-ytdlp-pot-provider pip package auto-registers as a yt-dlp plugin
to generate Proof-of-Origin tokens for YouTube (requires Node.js at runtime).
"""

import json
import os
import re
import subprocess
import tempfile
import uuid
import requests as http_requests
from flask import Blueprint, request, jsonify, send_file
from extensions import limiter
from config import get_config
from utils.logging import log_info, log_debug, log_error

youtube_bp = Blueprint('youtube', __name__, url_prefix='/api/youtube')

config = get_config()


_SUPPORTED_URL_PATTERNS = [
    r'^https?://(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/',
    r'^https?://(www\.|vm\.|vt\.|m\.)?tiktok\.com/',
    r'^https?://(www\.)?(instagram\.com|instagr\.am)/',
    r'^https?://(www\.|m\.)?(soundcloud\.com|snd\.sc)/',
]

_SC_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/131.0.0.0 Safari/537.36'
)


def _is_supported_url(url: str) -> bool:
    """Validate that the URL is from a supported platform."""
    return any(re.match(p, url) for p in _SUPPORTED_URL_PATTERNS)


def _is_soundcloud_url(url: str) -> bool:
    return 'soundcloud.com' in url or 'snd.sc' in url


# ------------------------------------------------------------------
# SoundCloud direct extraction (bypasses yt-dlp)
# ------------------------------------------------------------------

_sc_client_id_cache = None


def _extract_soundcloud_client_id(html: str) -> str:
    """Extract a fresh client_id from SoundCloud's JS assets."""
    global _sc_client_id_cache
    if _sc_client_id_cache:
        return _sc_client_id_cache

    # Find JS asset URLs in the page
    script_urls = re.findall(
        r'src="(https://a-v2\.sndcdn\.com/assets/[^"]+\.js)"', html
    )
    log_info(f"[SoundCloud] Found {len(script_urls)} JS assets to search for client_id")

    headers = {'User-Agent': _SC_USER_AGENT}
    for script_url in script_urls[-3:]:  # Check last few (client_id usually in later bundles)
        try:
            resp = http_requests.get(script_url, headers=headers, timeout=10)
            # Look for client_id pattern: client_id:"<32 hex chars>"
            match = re.search(r'client_id[:=]"([0-9a-zA-Z]{32})"', resp.text)
            if match:
                _sc_client_id_cache = match.group(1)
                log_info(f"[SoundCloud] Found client_id: {_sc_client_id_cache[:8]}...")
                return _sc_client_id_cache
        except Exception as e:
            log_error(f"[SoundCloud] Failed to fetch JS asset: {e}")
            continue

    raise ValueError("Could not extract SoundCloud client_id from page assets")


def _download_soundcloud(url: str, tmpdir: str) -> dict:
    """Download audio from SoundCloud by extracting HLS stream URL directly.

    Returns dict with 'output_file', 'title', 'thumbnail_url'.
    """
    headers = {'User-Agent': _SC_USER_AGENT}

    # 1. Fetch page HTML
    log_info(f"[SoundCloud] Fetching page: {url[:80]}")
    resp = http_requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text

    # 2. Extract hydration data
    hydration_match = re.search(
        r'<script>window\.__sc_hydration\s*=\s*(\[.+?\]);</script>',
        html, re.DOTALL
    )
    if not hydration_match:
        raise ValueError("Could not find SoundCloud hydration data")

    hydration = json.loads(hydration_match.group(1))

    # 3. Find track data
    track_data = None
    for entry in hydration:
        if entry.get('hydratable') == 'sound':
            track_data = entry.get('data', {})
            break

    if not track_data:
        raise ValueError("Could not find track data in hydration")

    title = track_data.get('title', 'audio')
    thumbnail_url = track_data.get('artwork_url', '') or ''
    # Get high-res thumbnail
    if thumbnail_url:
        thumbnail_url = thumbnail_url.replace('-large.', '-t500x500.')

    log_info(f"[SoundCloud] Track: {title}")

    # 4. Find HLS transcoding URL
    transcodings = track_data.get('media', {}).get('transcodings', [])
    transcoding_url = None

    # Prefer HLS AAC
    for t in transcodings:
        fmt = t.get('format', {})
        if fmt.get('protocol') == 'hls' and 'mp4' in fmt.get('mime_type', ''):
            transcoding_url = t.get('url')
            log_info(f"[SoundCloud] Using HLS AAC transcoding")
            break

    # Fall back to any HLS
    if not transcoding_url:
        for t in transcodings:
            fmt = t.get('format', {})
            if fmt.get('protocol') == 'hls':
                transcoding_url = t.get('url')
                log_info(f"[SoundCloud] Using HLS fallback transcoding")
                break

    # Fall back to any progressive
    if not transcoding_url:
        for t in transcodings:
            fmt = t.get('format', {})
            if fmt.get('protocol') == 'progressive':
                transcoding_url = t.get('url')
                log_info(f"[SoundCloud] Using progressive transcoding")
                break

    if not transcoding_url:
        raise ValueError(f"No suitable transcoding found. Available: "
                         f"{[t.get('format') for t in transcodings]}")

    # 5. Get client_id
    client_id = _extract_soundcloud_client_id(html)

    # 6. Fetch the actual stream URL
    separator = '&' if '?' in transcoding_url else '?'
    stream_api_url = f"{transcoding_url}{separator}client_id={client_id}"
    log_info(f"[SoundCloud] Fetching stream URL...")

    stream_resp = http_requests.get(stream_api_url, headers=headers, timeout=10)
    stream_resp.raise_for_status()
    stream_data = stream_resp.json()
    media_url = stream_data.get('url')

    if not media_url:
        raise ValueError("SoundCloud API did not return a stream URL")

    log_info(f"[SoundCloud] Got media URL: {media_url[:80]}...")

    # 7. Download the stream using ffmpeg (handles HLS m3u8 natively)
    output_path = os.path.join(tmpdir, f'{uuid.uuid4().hex}.m4a')
    cmd = [
        'ffmpeg', '-y',
        '-i', media_url,
        '-c:a', 'aac',
        '-b:a', '128k',
        '-vn',  # no video
        output_path,
    ]
    log_info(f"[SoundCloud] Downloading via ffmpeg...")
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=120
    )
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace')[-500:]
        raise ValueError(f"ffmpeg failed: {stderr}")

    file_size = os.path.getsize(output_path)
    log_info(f"[SoundCloud] Downloaded: {title} ({file_size // 1024}KB)")

    if file_size < 1000:
        raise ValueError(f"Downloaded file too small ({file_size} bytes)")

    return {
        'output_file': output_path,
        'title': title,
        'thumbnail_url': thumbnail_url,
    }


# ------------------------------------------------------------------
# Main extraction endpoint
# ------------------------------------------------------------------

@youtube_bp.route('/audio', methods=['POST'])
@limiter.limit(config.get_rate_limit('heavy_processing'))
def extract_audio():
    """
    Extract audio from a YouTube, TikTok, Instagram, or SoundCloud URL.

    Request JSON:
        { "url": "https://..." }

    Returns:
        Audio file (m4a/mp3) as binary response.
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400

    if not _is_supported_url(url):
        return jsonify({'error': 'URL must be from YouTube, TikTok, Instagram, or SoundCloud'}), 400

    log_info(f"[AudioExtract] Extraction requested for: {url[:80]}")

    tmpdir = tempfile.mkdtemp(prefix='riff_yt_')

    try:
        # SoundCloud: use direct extraction (yt-dlp's client_id is often stale)
        if _is_soundcloud_url(url):
            sc_result = _download_soundcloud(url, tmpdir)
            output_file = sc_result['output_file']
            ext = os.path.splitext(output_file)[1].lstrip('.')
            mimetype = 'audio/mp4' if ext in ('m4a', 'mp4') else 'audio/mpeg'

            response = send_file(
                output_file,
                mimetype=mimetype,
                as_attachment=True,
                download_name=f'audio.{ext}',
            )
            if sc_result.get('thumbnail_url'):
                response.headers['X-Thumbnail-URL'] = sc_result['thumbnail_url']
            return response

        # All other platforms: use yt-dlp
        import yt_dlp

        output_template = os.path.join(tmpdir, f'{uuid.uuid4().hex}.%(ext)s')

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'noplaylist': True,
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': _SC_USER_AGENT,
                'Accept-Language': 'en-US,en;q=0.9',
            },
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '128',
            }],
        }

        thumbnail_url = ''
        canonical_url = ''
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio')
            thumbnail_url = info.get('thumbnail', '')
            canonical_url = info.get('webpage_url', '')
            log_info(f"[YouTube] Downloaded: {title}")

        # Find the output file
        output_file = None
        for f in os.listdir(tmpdir):
            filepath = os.path.join(tmpdir, f)
            if os.path.isfile(filepath) and os.path.getsize(filepath) > 1000:
                output_file = filepath
                break

        if not output_file:
            log_info("[YouTube] No output file found after yt-dlp download")
            return jsonify({'error': 'Audio extraction failed'}), 500

        ext = os.path.splitext(output_file)[1].lstrip('.')
        mimetype = 'audio/mp4' if ext in ('m4a', 'mp4') else 'audio/mpeg'

        log_info(f"[YouTube] Sending {ext} file ({os.path.getsize(output_file) // 1024}KB)")

        response = send_file(
            output_file,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f'audio.{ext}',
        )
        if thumbnail_url:
            response.headers['X-Thumbnail-URL'] = thumbnail_url
        if canonical_url:
            response.headers['X-Canonical-URL'] = canonical_url
        return response

    except Exception as e:
        log_info(f"[AudioExtract] Extraction failed: {e}")
        return jsonify({'error': f'Audio extraction failed: {str(e)}'}), 500

    finally:
        import threading
        import time

        def cleanup():
            time.sleep(10)
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

        threading.Thread(target=cleanup, daemon=True).start()
