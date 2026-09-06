import asyncio
import base64
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

_executor = ThreadPoolExecutor(max_workers=2)
_youtube_cookie_file: Optional[str] = None

# ---------------------------------------------------------------------------
# URL type classification
# ---------------------------------------------------------------------------

_YOUTUBE_DOMAINS = frozenset({
    "youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com",
})

# Domains that serve HTML iframe embed pages (not direct video streams).
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
# Whisper post-processing (applied to speech-to-text output)
# ---------------------------------------------------------------------------

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
        # Bot-detection or sign-in wall → fall through to download instead of erroring
        if any(kw in err_str for kw in ("bot", "sign in", "confirm", "captcha", "blocked")):
            logger.warning(f"[captions] bot-detection triggered ({err_name}), skipping to download")
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
# Audio download — yt-dlp (YouTube fallback when captions unavailable)
# ---------------------------------------------------------------------------

def _download_audio_youtube(video_url: str, output_dir: str, filename: str) -> str:
    """
    Download audio from YouTube using yt-dlp.
    yt-dlp is actively maintained and handles current YouTube clients more reliably than pytubefix.
    Returns the full path to the downloaded file.
    """
    return _download_audio_ytdlp(video_url, str(Path(output_dir) / filename))


# ---------------------------------------------------------------------------
# Audio download — yt-dlp (iframe / direct URLs)
# ---------------------------------------------------------------------------

def _download_audio_ytdlp(video_url: str, output_path: str) -> str:
    """Download audio using yt-dlp (for iframe/direct URLs — requires ffmpeg)."""
    import yt_dlp
    ffmpeg_path = os.environ.get("FFMPEG_BINARY", "/usr/bin/ffmpeg")
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
        "ffmpeg-location": ffmpeg_path,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            },
        },
    }
    cookie_file = _get_youtube_cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
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


def _get_youtube_cookie_file() -> Optional[str]:
    """Materialize encrypted production cookies for yt-dlp without logging them."""
    global _youtube_cookie_file
    encoded = settings.youtube_cookies_b64.strip()
    if not encoded:
        return None
    if _youtube_cookie_file and Path(_youtube_cookie_file).exists():
        return _youtube_cookie_file

    try:
        cookie_data = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise RuntimeError("YOUTUBE_COOKIES_B64 is not valid base64") from exc
    if b"# Netscape HTTP Cookie File" not in cookie_data[:200]:
        raise RuntimeError("YOUTUBE_COOKIES_B64 must contain a Netscape cookies.txt file")

    cookie_path = Path(settings.temp_audio_dir) / "youtube-cookies.txt"
    cookie_path.write_bytes(cookie_data)
    try:
        os.chmod(cookie_path, 0o600)
    except OSError:
        pass
    _youtube_cookie_file = str(cookie_path)
    return _youtube_cookie_file


# ---------------------------------------------------------------------------
# Speech-to-text transcription
# ---------------------------------------------------------------------------

def _transcribe_with_groq_sync(audio_path: str) -> Tuple[str, str]:
    """Transcribe audio with Groq's Whisper-compatible speech-to-text API."""
    api_key = settings.effective_groq_api_key
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for audio transcription; "
            "Anthropic Claude does not accept audio input"
        )

    with open(audio_path, "rb") as audio_file:
        response = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (Path(audio_path).name, audio_file, "audio/mpeg")},
            data={
                "model": settings.groq_transcription_model,
                "response_format": "verbose_json",
            },
            timeout=180,
        )
    if response.is_error:
        raise RuntimeError(f"Speech-to-text provider returned HTTP {response.status_code}")

    payload = response.json()
    text = _postprocess_whisper(str(payload.get("text", "")))
    if not text:
        raise RuntimeError("Speech-to-text provider returned an empty transcript")
    return text, str(payload.get("language", "unknown"))


# Backward-compatible alias for older tests and callers.
_transcribe_with_claude_sync = _transcribe_with_groq_sync


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def transcribe_video(video_url: str, language: Optional[str] = None) -> Tuple[str, str]:
    """
    Transcribe any supported video URL.

    Routing strategy
    ----------------
    youtube  → try YouTube captions API first; fall back to yt-dlp + Whisper
    iframe   → fetch embed HTML to extract direct video URL; yt-dlp + Whisper
    direct   → pass URL straight to yt-dlp + Whisper (Vimeo, Facebook, etc.)
    """
    loop = asyncio.get_event_loop()
    url_type = _detect_url_type(video_url)
    logger.info(f"[transcribe] url_type={url_type!r} | {video_url[:80]}")

    temp_dir = Path(settings.temp_audio_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ── YouTube: try captions first, fall back to yt-dlp + Whisper ──────────
    if url_type == "youtube":
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
            f"[transcribe] no captions ({time.time()-t0:.2f}s), "
            "falling back to yt-dlp + Whisper"
        )

        audio_path = None
        try:
            uid = uuid.uuid4().hex
            t1 = time.time()
            audio_path = await loop.run_in_executor(
                _executor,
                _download_audio_youtube,
                video_url,
                str(temp_dir),
                f"audio_{uid}",
            )
            logger.info(f"[transcribe] yt-dlp download in {time.time()-t1:.2f}s")

            t2 = time.time()
            text, detected_lang = await loop.run_in_executor(
                _executor, _transcribe_with_groq_sync, audio_path
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

    # ── iframe: extract direct URL from embed HTML, then yt-dlp + Whisper ────
    download_url = video_url

    if url_type == "iframe":
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

    # ── yt-dlp download + Whisper transcription (iframe / direct) ──────────
    audio_base = str(temp_dir / f"audio_{uuid.uuid4().hex}")
    audio_path = None

    try:
        t1 = time.time()
        audio_path = await loop.run_in_executor(
            _executor, _download_audio_ytdlp, download_url, audio_base
        )
        logger.info(f"[transcribe] audio downloaded in {time.time()-t1:.2f}s")

        t2 = time.time()
        text, detected_lang = await loop.run_in_executor(
            _executor, _transcribe_with_groq_sync, audio_path
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
