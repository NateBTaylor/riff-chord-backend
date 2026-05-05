"""
YouTube audio extraction blueprint.

Provides an endpoint to extract audio from YouTube URLs via yt-dlp.
"""

from .routes import youtube_bp

__all__ = ['youtube_bp']
