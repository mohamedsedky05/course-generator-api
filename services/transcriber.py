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

from youtube_transcript_api import YouTubeTranscriptApi

from config import settings

logger = logging.getLogger("transcriber")

_AR = r'\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff'

_whisper_model = None
_executor = ThreadPoolExecutor(max_workers=2)


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
        print(f"[captions] failed: {type(e).__name__}")
        return None


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


async def transcribe_video(video_url: str, language: Optional[str] = None) -> Tuple[str, str]:
    loop = asyncio.get_event_loop()

    # Step 1: try captions first
    t0 = time.time()
    caption_result = await loop.run_in_executor(
        _executor, _try_youtube_captions, video_url
    )
    if caption_result:
        text, lang = caption_result
        if text:
            logger.info(f"[transcribe] captions fetched in {time.time()-t0:.2f}s | lang={lang} words={len(text.split())}")
            return text, lang

    logger.info(f"[transcribe] no captions ({time.time()-t0:.2f}s), falling back to Whisper")

    # Step 2: download audio and transcribe with Whisper
    temp_dir = Path(settings.temp_audio_dir)
    audio_base = str(temp_dir / f"audio_{uuid.uuid4().hex}")
    audio_path = None

    try:
        t1 = time.time()
        audio_path = await loop.run_in_executor(
            _executor, _download_audio_sync, video_url, audio_base
        )
        logger.info(f"[transcribe] audio downloaded in {time.time()-t1:.2f}s")

        t2 = time.time()
        whisper_lang = None if language == "auto" else language
        text, detected_lang = await loop.run_in_executor(
            _executor, _transcribe_audio_sync, audio_path, whisper_lang
        )
        logger.info(f"[transcribe] Whisper done in {time.time()-t2:.2f}s | lang={detected_lang} words={len(text.split())}")
        return text, detected_lang
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        if audio_path and Path(audio_path).exists():
            try:
                os.remove(audio_path)
            except OSError:
                pass
