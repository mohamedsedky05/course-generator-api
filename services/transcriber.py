import asyncio
import logging
import re
import time
import unicodedata
import uuid
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx
from youtube_transcript_api import YouTubeTranscriptApi

from config import settings

logger = logging.getLogger("transcriber")

_AR = r'؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿'

_whisper_model = None
_executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# URL type classification
# ---------------------------------------------------------------------------

_YOUTUBE_DOMAINS = frozenset({
    "youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com",
})

# Domains that serve HTML iframe embed pages (not direct video streams).
# The transcriber fetches the HTML and extracts the underlying video URL.
_IFRAME_DOMAINS = frozenset({
    "iframe.mediadelivery.net",   # Bunny.net CDN
    "fast.wistia.net",            # Wistia
    "cdn.jwplayer.com",           # JW Player
    "content.jwplatform.com",     # JW Platform
    "player.vimeo.com",           # Vimeo embed player
    "embed.vhx.tv",               # VHX / Vimeo OTT
    "player.brightcove.net",      # Brightcove
    "iframe.dacast.com",          # Dacast
    "embed.sproutvideo.com",      # SproutVideo
    "app.vidyard.com",            # Vidyard
})


def _detect_url_type(url: str) -> str:
    """
    Classify a video URL into one of three routing categories:

    - "youtube" : YouTube.com / youtu.be  → try captions first, yt-dlp fallback
    - "iframe"  : Known embed-CDN domains → fetch HTML to extract direct video URL
    - "direct"  : Everything else         → pass straight to yt-dlp (Vimeo, Facebook …)
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "direct"

    if host in _YOUTUBE_DOMAINS:
        return "youtube"
    if host in _IFRAME_DOMAINS:
        return "iframe"
    return "direct"


# ---------------------------------------------------------------------------
# Whisper helpers
# ---------------------------------------------------------------------------

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(settings.whisper_model)
    return _whisper_model


def _postprocess_whisper(text: str) -> str:
    """Fix common Whisper output artifacts in mixed Arabic/English transcriptions."""
    text = unicodedata.normalize("NFKC", text)
    # Ensure a single space at every Arabic↔Latin boundary
    text = re.sub(rf'([{_AR}])\s*([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(rf'([A-Za-z0-9])\s*([{_AR}])', r'\1 \2', text)
    # Collapse double-spaces inside English sequences (Whisper sometimes splits acronyms)
    text = re.sub(r'([A-Z])\s{2,}([A-Z])', r'\1\2', text)
    # Remove stray newlines that Whisper inserts mid-sentence
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _transcribe_audio_sync(audio_path: str, language: Optional[str]) -> Tuple[str, str]:
    model = load_whisper_model()
    result = model.transcribe(audio_path, task="transcribe", language=None, fp16=False)
    detected_lang = result.get("language", "unknown")
    text = _postprocess_whisper(result.get("text", "").strip())
    return text, detected_lang


# ---------------------------------------------------------------------------
# YouTube captions
# ---------------------------------------------------------------------------

def _try_youtube_captions(video_url: str) -> Optional[Tuple[str, str]]:
    try:
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", video_url)
        if not match:
            return None
        video_id = match.group(1)
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["ar", "en"])
        except Exception:
            # Fallback: grab whatever language is available
            transcript = next(iter(transcript_list), None)
            if transcript is None:
                return None
        entries = transcript.fetch()
        text = " ".join(e.text for e in entries)
        lang = transcript.language_code
        if len(text.split()) < 20:
            return None
        return text.strip(), lang
    except Exception as e:
        err_name = type(e).__name__
        err_str = str(e).lower()
        # Bot-detection or sign-in wall → fall through to yt-dlp instead of erroring
        if any(kw in err_str for kw in ("bot", "sign in", "confirm", "captcha", "blocked")):
            logger.warning(f"[captions] bot-detection triggered ({err_name}), skipping to yt-dlp")
            return None
        print(f"[captions] failed: {err_name}")
        return None


# ---------------------------------------------------------------------------
# Iframe / embed CDN extraction
# ---------------------------------------------------------------------------

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Ordered list of regex patterns to find a direct video URL inside embed HTML
_VIDEO_URL_PATTERNS = [
    # HTML5 <source src="...video.ext">
    r'<source[^>]+src=["\']([^"\']+\.(?:mp4|m3u8|webm|ogg)[^"\']*)["\']',
    # JW Player / generic JS:  file: "...ext"
    r'file\s*:\s*["\']([^"\']+\.(?:mp4|m3u8|webm|ogg)[^"\']*)["\']',
    # Config object:  src: "...ext"
    r'["\']src["\']\s*:\s*["\']([^"\']+\.(?:mp4|m3u8|webm|ogg)[^"\']*)["\']',
    # Bunny.net CDN assets on b-cdn.net
    r'["\']([^"\']*\.b-cdn\.net[^"\']*\.(?:mp4|m3u8)[^"\']*)["\']',
    # Generic HLS manifest
    r'["\']([^"\']+\.m3u8(?:\?[^"\']*)?)["\']',
    # JSON "url": "...ext"
    r'"url"\s*:\s*"([^"]+\.(?:mp4|m3u8|webm)[^"]*)"',
]


def _extract_video_from_iframe(url: str) -> Optional[str]:
    """
    Fetch an iframe embed page and extract the underlying direct video URL.

    Searches the HTML/JS for known video URL patterns and returns the first
    match normalised to an absolute HTTPS URL.  Returns None on any failure.
    """
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": _UA},
            timeout=15,
            follow_redirects=True,
        )
        html = resp.text
    except Exception as e:
        logger.warning(f"[iframe] HTTP fetch failed for {url[:80]}: {type(e).__name__}")
        return None

    parsed_base = urlparse(url)

    for pattern in _VIDEO_URL_PATTERNS:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            video_url = m.group(1)
            # Make protocol-relative or path-relative URLs absolute
            if video_url.startswith("//"):
                video_url = "https:" + video_url
            elif video_url.startswith("/"):
                video_url = f"{parsed_base.scheme}://{parsed_base.netloc}{video_url}"
            logger.info(f"[iframe] extracted direct URL: {video_url[:80]}")
            return video_url

    logger.warning(f"[iframe] no video URL found in HTML from {url[:80]}")
    return None


# ---------------------------------------------------------------------------
# yt-dlp audio download
# ---------------------------------------------------------------------------

def _download_audio_sync(video_url: str, output_path: str) -> str:
    import yt_dlp
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": "/usr/bin/ffmpeg",
        # Bot-detection bypass — ios client mimics the official YouTube app
        "cookiesfrombrowser": None,
        "extractor_args": {
            "youtube": {"player_client": ["ios", "web_creator", "tv_embedded"]},
        },
        "add_headers": {
            "User-Agent": (
                "com.google.ios.youtube/19.29.1 "
                "(iPhone16,2; U; CPU iOS 17_5_1 like Mac OS X)"
            ),
        },
        "sleep_interval": 2,
        "max_sleep_interval": 5,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    mp3_path = output_path + ".mp3"
    if Path(mp3_path).exists():
        return mp3_path
    for ext in [".m4a", ".webm", ".opus"]:
        candidate = output_path + ext
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Audio file not found after yt-dlp download")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def transcribe_video(video_url: str, language: Optional[str] = None) -> Tuple[str, str]:
    """
    Transcribe any supported video URL.

    Routing strategy
    ----------------
    youtube  → try YouTube captions API first; fall back to yt-dlp + Whisper
    iframe   → fetch embed HTML to extract direct video URL; fall back to
               passing the iframe URL directly to yt-dlp
    direct   → pass URL straight to yt-dlp + Whisper (Vimeo, Facebook, etc.)
    """
    loop = asyncio.get_event_loop()
    url_type = _detect_url_type(video_url)
    logger.info(f"[transcribe] url_type={url_type!r} | {video_url[:80]}")

    download_url = video_url  # may be overridden below

    if url_type == "youtube":
        # ── Step 1: fast caption path ──────────────────────────────────────
        t0 = time.time()
        caption_result = await loop.run_in_executor(
            _executor, _try_youtube_captions, video_url
        )
        if caption_result:
            text, lang = caption_result
            logger.info(
                f"[transcribe] captions fetched in {time.time()-t0:.2f}s | "
                f"lang={lang} words={len(text.split())}"
            )
            return text, lang
        logger.info(
            f"[transcribe] no captions ({time.time()-t0:.2f}s), falling back to yt-dlp"
        )

    elif url_type == "iframe":
        # ── Step 1: extract direct video URL from embed page HTML ──────────
        logger.info("[transcribe] iframe detected — attempting HTML extraction")
        t0 = time.time()
        direct_url = await loop.run_in_executor(
            _executor, _extract_video_from_iframe, video_url
        )
        if direct_url:
            logger.info(
                f"[transcribe] extracted direct URL in {time.time()-t0:.2f}s: "
                f"{direct_url[:80]}"
            )
            download_url = direct_url
        else:
            logger.info(
                f"[transcribe] HTML extraction failed ({time.time()-t0:.2f}s), "
                "passing iframe URL directly to yt-dlp"
            )

    # ── Download audio with yt-dlp, transcribe with Whisper ───────────────
    temp_dir = Path(settings.temp_audio_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_base = str(temp_dir / f"audio_{uuid.uuid4().hex}")
    audio_path = None

    try:
        t1 = time.time()
        audio_path = await loop.run_in_executor(
            _executor, _download_audio_sync, download_url, audio_base
        )
        logger.info(f"[transcribe] audio downloaded in {time.time()-t1:.2f}s")

        t2 = time.time()
        whisper_lang = None if language == "auto" else language
        text, detected_lang = await loop.run_in_executor(
            _executor, _transcribe_audio_sync, audio_path, whisper_lang
        )
        logger.info(
            f"[transcribe] Whisper done in {time.time()-t2:.2f}s | "
            f"lang={detected_lang} words={len(text.split())}"
        )
        return text, detected_lang
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        if audio_path and Path(audio_path).exists():
            try:
                os.remove(audio_path)
            except OSError:
                pass
