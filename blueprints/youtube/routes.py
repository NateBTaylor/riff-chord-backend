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
from urllib.parse import urlparse
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


def _youtube_cookies_path():
    """Materialize the YOUTUBE_COOKIES_TXT env var into a Netscape-format
    cookies.txt file at /tmp/youtube_cookies.txt and return the path.

    YouTube's bot detection is bypassed when yt-dlp sends a signed-in
    user's cookies — those requests look like normal browser traffic.
    Cookies typically last 2-8 weeks before YouTube invalidates them;
    when that happens, re-export from your browser and update the env
    var in Railway.

    Returns None if no env var is set. Callers should silently skip
    the cookies path when None is returned.
    """
    content = os.environ.get('YOUTUBE_COOKIES_TXT')
    if not content:
        return None

    # Normalize literal '\n' some env-var UIs paste in instead of newlines.
    content = content.replace('\\n', '\n')

    # Sanity-check: at least one line must look like a Netscape cookie
    # row (7 tab-separated columns, domain starts with '.' or is bare).
    # If the env var got polluted with a Python traceback or other junk,
    # log a loud error rather than silently writing garbage that yt-dlp
    # will reject one line at a time.
    looks_valid = False
    for line in content.splitlines()[:60]:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        cols = s.split('\t')
        if len(cols) >= 7 and (cols[0].startswith('.') or '.' in cols[0]):
            looks_valid = True
            break

    if not looks_valid:
        log_error(
            "[YouTube] YOUTUBE_COOKIES_TXT doesn't look like Netscape cookies. "
            f"First 160 chars: {content[:160]!r}"
        )
        return None

    path = '/tmp/youtube_cookies.txt'
    try:
        with open(path, 'w') as f:
            f.write(content)
        return path
    except Exception:
        return None


@youtube_bp.route('/audio', methods=['POST'])
@limiter.limit(config.get_rate_limit('heavy_processing'))
def extract_audio():
    """
    Extract audio from a YouTube, TikTok, or Instagram URL.

    Delegates to services.yt_dlp_runner which spawns yt-dlp as a
    subprocess and iterates through multiple player_client values
    (cycling ios → android → tv_embedded → mediaconnect → web_safari
    for YouTube) until one returns playable audio. The PoT plugin
    auto-registers per subprocess invocation.

    Request JSON:
        { "url": "https://..." }

    Returns:
        Audio file (m4a/mp3/webm) as binary response with
        X-Thumbnail-URL and X-Canonical-URL headers.
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400
    if not _is_supported_url(url):
        return jsonify({'error': 'URL must be from YouTube, TikTok, or Instagram'}), 400

    log_info(f"[AudioExtract] Extraction requested for: {url[:80]}")
    tmpdir = tempfile.mkdtemp(prefix='riff_yt_')

    try:
        from services.yt_dlp_runner import download as run_download, YtDlpError
        result = run_download(url, tmpdir, timeout=120)

        ext = result.extension if result.extension in ('m4a', 'mp3', 'webm',
                                                       'mp4', 'opus', 'ogg') else 'm4a'
        mimetype = {
            'm4a': 'audio/mp4',
            'mp4': 'audio/mp4',
            'mp3': 'audio/mpeg',
            'webm': 'audio/webm',
            'opus': 'audio/ogg',
            'ogg': 'audio/ogg',
        }.get(ext, 'audio/mp4')

        log_info(f"[AudioExtract] Sending {ext} file "
                 f"({os.path.getsize(result.file_path) // 1024}KB) "
                 f"via client={result.client_used}")

        response = send_file(
            result.file_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f'audio.{ext}',
        )
        if result.thumbnail_url:
            response.headers['X-Thumbnail-URL'] = result.thumbnail_url
        if result.canonical_url:
            response.headers['X-Canonical-URL'] = result.canonical_url
        return response

    except YtDlpError as e:
        log_info(f"[AudioExtract] yt-dlp failed: {e}")
        return jsonify({'error': f'Audio extraction failed: {str(e)[:300]}'}), 502
    except Exception as e:
        log_info(f"[AudioExtract] unexpected error: {e}")
        return jsonify({'error': f'Audio extraction failed: {str(e)[:300]}'}), 500

    finally:
        import threading as _th, time as _ti
        def _cleanup():
            _ti.sleep(15)
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
        _th.Thread(target=_cleanup, daemon=True).start()




@youtube_bp.route('/metadata', methods=['POST'])
@limiter.limit("30/minute")
def fetch_metadata():
    """
    Fetch lightweight metadata for a URL without downloading audio.

    Used by the iOS import confirmation sheet to prefill title/artist/
    thumbnail while the user picks genre & difficulty. Runs yt-dlp with
    `download=False` so it returns in ~1-3s.

    Request JSON:  { "url": "https://..." }
    Response:      { "title": "...", "artist": "...", "duration": 213,
                     "thumbnail_url": "...", "webpage_url": "..." }
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400
    if not _is_supported_url(url):
        return jsonify({'error': 'URL must be from YouTube, TikTok, or Instagram'}), 400

    # Fast path: YouTube has its own free oEmbed endpoint that doesn't
    # require auth or trigger bot detection. Returns title + author +
    # thumbnail in ~150ms. Skips yt-dlp entirely for YouTube URLs.
    host = (urlparse(url).hostname or '').lower()
    if host in {'youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be'}:
        try:
            import requests as _requests
            oembed_resp = _requests.get(
                'https://www.youtube.com/oembed',
                params={'url': url, 'format': 'json'},
                timeout=8,
            )
            if oembed_resp.status_code == 200:
                data = oembed_resp.json()
                log_info(f"[Metadata] YouTube oEmbed: \"{data.get('title', '')[:50]}\"")
                return jsonify({
                    'title': data.get('title') or '',
                    'artist': data.get('author_name') or '',
                    'duration': 0,  # oEmbed doesn't include duration
                    'thumbnail_url': data.get('thumbnail_url') or '',
                    'webpage_url': url,
                }), 200
            log_info(f"[Metadata] oEmbed HTTP {oembed_resp.status_code} — "
                     f"falling through to yt-dlp")
        except Exception as e:
            log_info(f"[Metadata] oEmbed failed: {e} — falling through to yt-dlp")

    try:
        import yt_dlp

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'noplaylist': True,
            'socket_timeout': 15,
            # PoT-friendly clients for TikTok/Instagram/IG (and the unlikely
            # case where oEmbed fell through for YouTube).
            'extractor_args': {
                'youtube': {
                    'player_client': ['web_safari', 'mweb', 'tv_embedded'],
                },
            },
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            },
        }

        # Same cookies env var as extract_audio. YouTube metadata uses
        # the oEmbed fast-path above so cookies are rarely needed here,
        # but if oEmbed falls through cookies make the yt-dlp retry work.
        cookies_path = _youtube_cookies_path()
        if cookies_path:
            ydl_opts['cookiefile'] = cookies_path

        # Retry transient network errors so the iOS confirmation sheet's
        # title/artist/thumbnail prefill survives a flaky TikTok request.
        # Also retry on bot-detection errors — the PoT plugin sometimes
        # needs a second attempt to mint a fresh token.
        import time as _time
        info = None
        for attempt in range(3):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                break
            except Exception as e:
                error_str = str(e).lower()
                transient = any(token in error_str for token in (
                    'connection aborted',
                    'remotedisconnected',
                    'connection reset',
                    'timed out',
                    'temporary failure',
                    # YouTube bot-detection: the PoT provider sometimes
                    # needs a second attempt to fetch a fresh token.
                    "sign in to confirm",
                    'confirm you',
                    "you're not a bot",
                ))
                if not transient or attempt == 2:
                    raise
                _time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s

        # uploader / creator naming differs by platform — try the most
        # specific fields first.
        artist = (
            info.get('artist')
            or info.get('creator')
            or info.get('uploader')
            or info.get('channel')
            or ''
        )

        return jsonify({
            'title': info.get('title') or '',
            'artist': artist,
            'duration': info.get('duration') or 0,
            'thumbnail_url': info.get('thumbnail') or '',
            'webpage_url': info.get('webpage_url') or url,
        }), 200

    except Exception as e:
        log_info(f"[Metadata] Fetch failed for {url[:80]}: {e}")
        return jsonify({'error': f'Metadata fetch failed: {str(e)}'}), 500
