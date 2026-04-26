import logging
import re
import time
from typing import Optional

from fastapi import APIRouter, Form, File, Request, UploadFile
from fastapi.responses import JSONResponse

from services import cache as response_cache
from services.extractor import (
    extract_text_from_file,
    detect_language,
    count_words,
    clean_text,
    SUPPORTED_EXTENSIONS,
)
from services.transcriber import transcribe_video
from services.chunker import chunk_text, merge_course_chunks
from services.llm_service import generate_course
from utils.rate_limit import limiter

router = APIRouter(prefix="/api", tags=["generate"])
logger = logging.getLogger("router")

MIN_WORDS = 50
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

_YOUTUBE_RE = re.compile(
    r'^https?://(www\.)?(youtube\.com/(watch\?.*v=|shorts/|embed/)|youtu\.be/)[A-Za-z0-9_\-]{11}',
    re.IGNORECASE,
)


def _is_valid_video_url(url: str) -> bool:
    return bool(_YOUTUBE_RE.search(url.strip()))


# ---------------------------------------------------------------------------
# Bilingual error helper
# ---------------------------------------------------------------------------

_AR_MESSAGES = {
    "NO_INPUT":             "يرجى تقديم مصدر واحد فقط: نص، ملف، أو رابط فيديو.",
    "MULTIPLE_INPUTS":      "يُسمح بمصدر واحد فقط في كل طلب: نص أو ملف أو رابط فيديو.",
    "INVALID_VIDEO_URL":    "رابط الفيديو غير صالح. يُرجى إدخال رابط يوتيوب صحيح.",
    "FILE_TOO_LARGE":       "حجم الملف يتجاوز الحد المسموح به (50 ميجابايت).",
    "UNSUPPORTED_FILE_TYPE":"نوع الملف غير مدعوم. الأنواع المدعومة: PDF، DOCX، PPTX، TXT.",
    "FILE_EXTRACTION_ERROR":"تعذّر استخراج النص من الملف. تأكد من أن الملف غير تالف.",
    "VIDEO_UNAVAILABLE":    "تعذّر الوصول إلى الفيديو. تأكد من أن الرابط صحيح وأن الفيديو عام.",
    "TRANSCRIPTION_FAILED": "فشل تحويل الصوت إلى نص. حاول مرة أخرى أو جرّب فيديو آخر.",
    "TEXT_TOO_SHORT":       "النص المستخرج قصير جداً. يجب أن يحتوي على 50 كلمة على الأقل.",
    "LLM_QUOTA_EXCEEDED":   "تم استنفاد حصة واجهة برمجة الذكاء الاصطناعي. يرجى المحاولة لاحقاً.",
    "LLM_ERROR":            "حدث خطأ أثناء توليد المحتوى. يرجى المحاولة مرة أخرى.",
}


def _build_error(code: str, message: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error_code": code,
            "message": message,
            "ar_message": _AR_MESSAGES.get(code, "حدث خطأ. يرجى المحاولة مرة أخرى."),
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate")
@limiter.limit("10/minute")
async def generate_endpoint(
    request: Request,
    text: Optional[str] = Form(None),
    video_url: Optional[str] = Form(None),
    num_lectures: int = Form(default=3, ge=2, le=8),
    num_quiz_questions: int = Form(default=10, ge=5, le=20),
    output_language: str = Form(default="auto"),
    file: Optional[UploadFile] = File(None),
):
    start_time = time.time()

    # ── Input source validation ────────────────────────────────────────────
    provided = sum([text is not None, video_url is not None, file is not None])
    if provided == 0:
        return _build_error("NO_INPUT", "Provide exactly one of: text, file, or video_url.")
    if provided > 1:
        return _build_error("MULTIPLE_INPUTS", "Only one input source is allowed per request.")

    if video_url is not None and not _is_valid_video_url(video_url):
        return _build_error(
            "INVALID_VIDEO_URL",
            f"'{video_url}' is not a recognised YouTube URL. "
            "Expected format: https://www.youtube.com/watch?v=... or https://youtu.be/...",
        )

    # ── Cache lookup (video_url only) ──────────────────────────────────────
    cache_key: Optional[str] = None
    if video_url is not None:
        cache_key = response_cache.make_key(video_url)
        cached = response_cache.get(cache_key)
        if cached is not None:
            logger.info(f"[router] Cache HIT for {video_url}")
            return cached

    # ── Text extraction ────────────────────────────────────────────────────
    raw_text = ""
    input_type = ""
    transcription = None
    detected_language = "en"

    if text is not None:
        raw_text = clean_text(text)
        input_type = "plain_text"
        logger.info(f"[router] plain_text input | words={len(raw_text.split())}")

    elif file is not None:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_BYTES:
            return _build_error(
                "FILE_TOO_LARGE",
                f"File size {len(file_bytes) // (1024*1024)} MB exceeds the 50 MB limit.",
            )
        try:
            t0 = time.time()
            raw_text = extract_text_from_file(file.filename or "", file_bytes)
            logger.info(
                f"[router] file extracted in {time.time()-t0:.2f}s | "
                f"name={file.filename} words={len(raw_text.split())}"
            )
        except ValueError as e:
            return _build_error("UNSUPPORTED_FILE_TYPE", str(e))
        except Exception as e:
            return _build_error("FILE_EXTRACTION_ERROR", f"Could not extract text from file: {e}")
        input_type = "file_upload"

    elif video_url is not None:
        try:
            lang_hint = None if output_language == "auto" else output_language
            raw_text, detected_language = await transcribe_video(video_url, lang_hint)
            transcription = raw_text
        except RuntimeError as e:
            msg = str(e).lower()
            if "unavailable" in msg or "private" in msg:
                return _build_error("VIDEO_UNAVAILABLE", "Could not access the provided video URL.")
            return _build_error("TRANSCRIPTION_FAILED", f"Audio transcription failed: {e}", 500)
        except Exception as e:
            return _build_error("VIDEO_UNAVAILABLE", f"Could not access the provided video URL: {e}")
        input_type = "youtube_video"
        logger.info(f"[router] video transcription complete | words={len(raw_text.split())}")

    # ── Word-count gate ────────────────────────────────────────────────────
    word_count = count_words(raw_text)
    if word_count < MIN_WORDS:
        return _build_error(
            "TEXT_TOO_SHORT",
            f"Extracted text has only {word_count} words. Minimum required: {MIN_WORDS}.",
        )

    # ── Language detection ─────────────────────────────────────────────────
    if input_type != "youtube_video" or detected_language == "unknown":
        detected_language = detect_language(raw_text)
        if output_language != "auto":
            detected_language = output_language

    # ── Chunking ───────────────────────────────────────────────────────────
    chunks = chunk_text(raw_text)
    chunks_used = len(chunks)
    logger.info(f"[router] chunked into {chunks_used} chunk(s)")

    # ── LLM generation ────────────────────────────────────────────────────
    try:
        result = await generate_course(raw_text, chunks, num_lectures, num_quiz_questions)
    except Exception as e:
        msg = str(e).lower()
        if "quota" in msg or "resourceexhausted" in msg:
            return _build_error("LLM_QUOTA_EXCEEDED", "Gemini API quota exceeded. Try again later.", 429)
        return _build_error("LLM_ERROR", f"Content generation failed: {e}", 500)

    if isinstance(result, list):
        result = merge_course_chunks(result)

    processing_time = round(time.time() - start_time, 2)
    logger.info(f"[router] request complete in {processing_time}s | input_type={input_type}")

    response_body = {
        "status": "success",
        "input_type": input_type,
        "detected_language": detected_language,
        "transcription": transcription,
        "course": result,
        "metadata": {
            "processing_time_seconds": processing_time,
            "word_count": word_count,
            "chunks_used": chunks_used,
        },
    }

    # ── Cache store (video only) ───────────────────────────────────────────
    if cache_key is not None:
        response_cache.set(cache_key, response_body)
        logger.info(f"[router] result cached for {video_url}")

    return response_body


@router.get("/health")
async def health_check():
    from services.transcriber import _whisper_model
    from config import settings
    return {
        "status": "ok",
        "whisper_model": settings.whisper_model,
        "whisper_loaded": _whisper_model is not None,
        "gemini_configured": bool(settings.gemini_api_key),
        "cache_entries": response_cache.size(),
        "allowed_origins": settings.allowed_origins_list,
    }
