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

    # For YouTube URLs, only fall back to Piped when we don't have
    # cookies configured. Cookies + yt-dlp is faster and more reliable
    # than the public Piped instance chain (which has been mostly down).
    yt_host = (urlparse(url).hostname or '').lower()
    is_youtube = yt_host in {'youtube.com', 'www.youtube.com',
                              'm.youtube.com', 'youtu.be'}
    cookies_configured = bool(os.environ.get('YOUTUBE_COOKIES_TXT'))
    if is_youtube and not cookies_configured:
        try:
            from services.youtube_piped import download_audio as piped_download
            piped_path = piped_download(url, tmpdir)
        except Exception as e:
            log_info(f"[YouTube] Piped extractor errored ({e}) — falling through to yt-dlp")
            piped_path = None
        if piped_path and os.path.exists(piped_path) and os.path.getsize(piped_path) > 1000:
            ext = os.path.splitext(piped_path)[1].lstrip('.')
            mimetype = 'audio/mp4' if ext in ('m4a', 'mp4') else 'audio/webm' if ext == 'webm' else 'audio/mpeg'
            log_info(f"[YouTube] Piped success: sending {ext} file "
                     f"({os.path.getsize(piped_path) // 1024}KB)")
            response = send_file(piped_path, mimetype=mimetype,
                                 as_attachment=True, download_name=f'audio.{ext}')

            # Schedule cleanup of the temp dir after send_file streams.
            import threading as _th, time as _ti
            def _cleanup():
                _ti.sleep(10)
                try:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
            _th.Thread(target=_cleanup, daemon=True).start()
            return response
        log_info("[YouTube] Piped path exhausted — falling through to yt-dlp")

    try:
        import yt_dlp

        # Strategy: probe formats first across multiple player_client
        # configs, log what we get, then pick the best audio format
        # directly by ID. Bypasses yt-dlp's format selector entirely,
        # which has been opaque and unreliable.
        base_opts = {
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

        cookies_path = _youtube_cookies_path()
        if cookies_path:
            base_opts['cookiefile'] = cookies_path
            log_info("[YouTube] Using YOUTUBE_COOKIES_TXT for yt-dlp auth")

        # Try several player_client combos in sequence. Different clients
        # see different format menus on the same video; first one that
        # returns ANY audio-capable format wins.
        client_configs = (
            ['ios', 'android', 'tv_embedded', 'web', 'mweb', 'web_safari']
            if cookies_path else
            ['web_safari', 'mweb', 'tv_embedded']
        )

        info = None
        chosen_client = None
        for client in client_configs:
            # process=False bypasses yt-dlp's format selector entirely.
            # That selector was raising "Requested format is not available"
            # during the probe phase, masking whether formats actually
            # existed at all. With process=False we get the raw extractor
            # output, no selector applied.
            probe_opts = {**base_opts,
                          'extractor_args': {'youtube': {'player_client': [client]}}}
            try:
                with yt_dlp.YoutubeDL(probe_opts) as ydl:
                    probe = ydl.extract_info(url, download=False, process=False)
                formats = probe.get('formats') or []
                audio_formats = [
                    f for f in formats
                    if (f.get('acodec') and f.get('acodec') != 'none')
                    and f.get('url')
                ]
                # Dump first 3 format IDs + extras so we can see if formats
                # are present but unsuitable, vs totally absent.
                summary = ", ".join(
                    f"{f.get('format_id')}({f.get('ext')},{f.get('acodec') or '-'}/"
                    f"{f.get('vcodec') or '-'})"
                    for f in formats[:5]
                )
                log_info(f"[YouTube] client={client} → "
                         f"{len(formats)} formats ({len(audio_formats)} usable audio): "
                         f"[{summary}]")
                if audio_formats:
                    info = probe
                    chosen_client = client
                    break
            except Exception as e:
                log_info(f"[YouTube] client={client} probe failed: {str(e)[:250]}")
                continue

        if not info:
            return jsonify({'error': 'All yt-dlp player clients failed to return audio formats. '
                                     'YouTube may have invalidated the cookies — re-export them.'}), 500

        # Pick the highest-bitrate audio-only stream (audio_formats sorted desc
        # by abr / bitrate; fall back to first if no abr available).
        audio_formats = [
            f for f in info.get('formats', [])
            if (f.get('acodec') and f.get('acodec') != 'none')
            and f.get('url')
        ]
        # Prefer audio-only (vcodec=none), then by abr/bitrate
        audio_only = [f for f in audio_formats if f.get('vcodec') == 'none']
        target_list = audio_only or audio_formats
        chosen = max(
            target_list,
            key=lambda f: (f.get('abr') or 0, f.get('tbr') or 0, f.get('filesize') or 0)
        )
        log_info(f"[YouTube] picked format {chosen.get('format_id')} "
                 f"({chosen.get('ext')}, {chosen.get('abr')}kbps) via {chosen_client}")

        # Now actually download with that specific format ID.
        ydl_opts = {**base_opts,
                    'format': chosen['format_id'],
                    'extractor_args': {
                        'youtube': {'player_client': [chosen_client]},
                    }}

        thumbnail_url = ''
        canonical_url = ''

        # Retry transient network errors (RemoteDisconnected, connection
        # resets) before failing. TikTok in particular drops connections
        # ~5% of the time; without retry, iOS falls through to its slow
        # ~25s Cobalt fallback chain when a single retry would have
        # succeeded immediately.
        import time as _time
        last_error = None
        for attempt in range(3):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                title = info.get('title', 'audio')
                thumbnail_url = info.get('thumbnail', '')
                canonical_url = info.get('webpage_url', '')
                log_info(f"[YouTube] Downloaded: {title}"
                         + (f" (attempt {attempt + 1})" if attempt else ""))
                last_error = None
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Only retry on transient network failures, not on
                # permanent ones like 404/410 (DRM, removed, region-locked).
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
                wait = 0.5 * (2 ** attempt)  # 0.5s, 1s
                log_info(f"[YouTube] Transient error on attempt {attempt + 1}, "
                         f"retrying in {wait:.1f}s: {e}")
                _time.sleep(wait)

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
