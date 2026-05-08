"""
Audio extraction routes.

Provides an endpoint that accepts a YouTube, TikTok, or Instagram URL,
extracts audio via yt-dlp, and returns the audio file to the client.

The bgutil-ytdlp-pot-provider pip package auto-registers as a yt-dlp plugin
to generate Proof-of-Origin tokens for YouTube (requires Node.js at runtime).
"""

import os
import re
import tempfile
import uuid
from flask import Blueprint, request, jsonify, send_file
from extensions import limiter
from config import get_config
from utils.logging import log_info, log_debug

youtube_bp = Blueprint('youtube', __name__, url_prefix='/api/youtube')

config = get_config()


_SUPPORTED_URL_PATTERNS = [
    r'^https?://(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/',
    r'^https?://(www\.|vm\.|vt\.|m\.)?tiktok\.com/',
    r'^https?://(www\.)?(instagram\.com|instagr\.am)/',
]


def _is_supported_url(url: str) -> bool:
    """Validate that the URL is from a supported platform."""
    return any(re.match(p, url) for p in _SUPPORTED_URL_PATTERNS)


@youtube_bp.route('/audio', methods=['POST'])
@limiter.limit(config.get_rate_limit('heavy_processing'))
def extract_audio():
    """
    Extract audio from a YouTube, TikTok, or Instagram URL using yt-dlp.

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
        return jsonify({'error': 'URL must be from YouTube, TikTok, or Instagram'}), 400

    log_info(f"[AudioExtract] Extraction requested for: {url[:80]}")

    tmpdir = tempfile.mkdtemp(prefix='riff_yt_')
    output_template = os.path.join(tmpdir, f'{uuid.uuid4().hex}.%(ext)s')

    try:
        import yt_dlp

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'noplaylist': True,
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
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
        log_info(f"[YouTube] Extraction failed: {e}")
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
