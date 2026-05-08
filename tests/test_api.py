"""API endpoint tests using FastAPI TestClient."""
import io
import pytest
from unittest.mock import AsyncMock

from tests.conftest import ENGLISH_TEXT

# ---------------------------------------------------------------------------
# Shared mock data  (new simplified schema: title + description + quiz only)
# ---------------------------------------------------------------------------

_MOCK_COURSE = {
    "title": "Introduction to Machine Learning",
    "description": "A beginner course covering ML fundamentals.",
    "quiz": [
        {
            "question_number": 1,
            "type": "mcq",
            "question": "What is ML?",
            "options": ["A. AI subset", "B. Database", "C. OS", "D. Network"],
            "correct_answer": 0,
            "explanation": "ML is a subset of AI.",
        },
        {
            "question_number": 2,
            "type": "true_false",
            "question": "ML requires explicit programming.",
            "correct_answer": False,
            "explanation": "ML learns from data without explicit programming.",
        },
    ],
}

_MOCK_TRANSCRIPTION = (ENGLISH_TEXT, "en")


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch generate_content at the router level to return a fake result."""
    mock = AsyncMock(return_value=_MOCK_COURSE)
    monkeypatch.setattr("routers.generate.generate_content", mock)
    return mock


@pytest.fixture
def mock_transcribe(monkeypatch):
    """Patch transcribe_video to skip real network calls."""
    mock = AsyncMock(return_value=_MOCK_TRANSCRIPTION)
    monkeypatch.setattr("routers.generate.transcribe_video", mock)
    return mock


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        assert client.get("/api/health").json()["status"] == "ok"

    def test_contains_required_fields(self, client):
        body = client.get("/api/health").json()
        for field in ("whisper_model", "whisper_loaded", "gemini_configured",
                      "cache_entries", "allowed_origins"):
            assert field in body, f"Missing field: {field}"

    def test_allowed_origins_is_list(self, client):
        assert isinstance(client.get("/api/health").json()["allowed_origins"], list)

    def test_cache_entries_is_int(self, client):
        assert isinstance(client.get("/api/health").json()["cache_entries"], int)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_no_input_returns_400(self, client):
        r = client.post("/api/generate", data={})
        assert r.status_code == 400
        assert r.json()["error_code"] == "NO_INPUT"

    def test_no_input_has_ar_message(self, client):
        body = client.post("/api/generate", data={}).json()
        assert "ar_message" in body
        assert len(body["ar_message"]) > 0

    def test_multiple_inputs_returns_400(self, client):
        r = client.post("/api/generate", data={
            "text": ENGLISH_TEXT,
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        })
        assert r.status_code == 400
        assert r.json()["error_code"] == "MULTIPLE_INPUTS"

    def test_invalid_url_returns_400(self, client):
        """A URL that doesn't start with http/https should be rejected."""
        r = client.post("/api/generate", data={
            "video_url": "not-a-valid-url",
        })
        assert r.status_code == 400
        assert r.json()["error_code"] == "INVALID_URL"

    def test_invalid_url_has_ar_message(self, client):
        r = client.post("/api/generate", data={"video_url": "not-a-url"})
        body = r.json()
        assert "ar_message" in body

    def test_text_too_short_returns_400(self, client):
        r = client.post("/api/generate", data={"text": "too short"})
        assert r.status_code == 400
        assert r.json()["error_code"] == "TEXT_TOO_SHORT"

    def test_file_too_large_returns_400(self, client):
        big = b"a" * (51 * 1024 * 1024)  # 51 MB
        r = client.post(
            "/api/generate",
            files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
        )
        assert r.status_code == 400
        assert r.json()["error_code"] == "FILE_TOO_LARGE"

    def test_unsupported_file_type_returns_400(self, client):
        r = client.post(
            "/api/generate",
            files={"file": ("doc.odt", io.BytesIO(b"content"), "application/octet-stream")},
        )
        assert r.status_code == 400
        assert r.json()["error_code"] == "UNSUPPORTED_FILE_TYPE"

    def test_error_response_always_has_status_error(self, client):
        r = client.post("/api/generate", data={})
        assert r.json()["status"] == "error"

    def test_valid_video_urls_accepted(self, client, mock_llm, mock_transcribe):
        """Any https:// URL should pass validation — YouTube, Vimeo, iframe CDN, etc."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://vimeo.com/123456789",
            "https://www.facebook.com/video/123456789",
            "https://iframe.mediadelivery.net/embed/578375/3fbdf5a1?token=abc",  # Bunny.net
        ]
        for url in valid_urls:
            r = client.post("/api/generate", data={"video_url": url})
            assert r.status_code == 200, (
                f"Expected 200 for {url}, got {r.status_code}: {r.text}"
            )

    def test_non_https_url_rejected(self, client):
        """Only https:// is accepted; http://, ftp://, and bare strings are rejected."""
        bad_urls = [
            "ftp://example.com/video.mp4",
            "http://insecure.com/video.mp4",   # plain HTTP not accepted
            "example.com/video",
            "just-text",
        ]
        for bad_url in bad_urls:
            r = client.post("/api/generate", data={"video_url": bad_url})
            assert r.status_code == 400, f"Expected 400 for '{bad_url}'"
            assert r.json()["error_code"] == "INVALID_URL"


# ---------------------------------------------------------------------------
# Plain text generation
# ---------------------------------------------------------------------------

class TestGeneratePlainText:
    def test_returns_200(self, client, mock_llm):
        r = client.post("/api/generate", data={"text": ENGLISH_TEXT})
        assert r.status_code == 200

    def test_status_is_success(self, client, mock_llm):
        body = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()
        assert body["status"] == "success"

    def test_input_type_is_plain_text(self, client, mock_llm):
        body = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()
        assert body["input_type"] == "plain_text"

    def test_course_key_present(self, client, mock_llm):
        body = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()
        assert "course" in body

    def test_course_has_title(self, client, mock_llm):
        course = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()["course"]
        assert "title" in course

    def test_course_has_description(self, client, mock_llm):
        course = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()["course"]
        assert "description" in course

    def test_course_has_quiz(self, client, mock_llm):
        course = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()["course"]
        assert "quiz" in course
        assert isinstance(course["quiz"], list)

    def test_metadata_key_present(self, client, mock_llm):
        body = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()
        assert "metadata" in body

    def test_metadata_has_required_fields(self, client, mock_llm):
        meta = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()["metadata"]
        for field in ("processing_time_seconds", "word_count"):
            assert field in meta

    def test_word_count_in_metadata(self, client, mock_llm):
        meta = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()["metadata"]
        assert meta["word_count"] > 0

    def test_detected_language_english(self, client, mock_llm):
        body = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()
        assert body["detected_language"] == "en"

    def test_transcript_populated_for_text(self, client, mock_llm):
        """transcript is always present and filled even for plain-text input."""
        body = client.post("/api/generate", data={"text": ENGLISH_TEXT}).json()
        assert "transcript" in body
        assert body["transcript"] is not None
        assert len(body["transcript"]) > 0

    def test_num_quiz_questions_param_passed_to_llm(self, client, mock_llm):
        client.post("/api/generate", data={"text": ENGLISH_TEXT, "num_quiz_questions": 8})
        call_args = mock_llm.call_args[0]
        assert call_args[1] == 8  # second positional arg to generate_content(text, num_quiz_questions)


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

class TestFileUpload:
    def test_txt_upload_returns_200(self, client, mock_llm, txt_bytes):
        r = client.post(
            "/api/generate",
            files={"file": ("doc.txt", io.BytesIO(txt_bytes), "text/plain")},
        )
        assert r.status_code == 200

    def test_txt_input_type(self, client, mock_llm, txt_bytes):
        body = client.post(
            "/api/generate",
            files={"file": ("doc.txt", io.BytesIO(txt_bytes), "text/plain")},
        ).json()
        assert body["input_type"] == "file_upload"

    def test_txt_transcript_populated(self, client, mock_llm, txt_bytes):
        body = client.post(
            "/api/generate",
            files={"file": ("doc.txt", io.BytesIO(txt_bytes), "text/plain")},
        ).json()
        assert body["transcript"] is not None
        assert len(body["transcript"]) > 0

    def test_docx_upload_returns_200(self, client, mock_llm, docx_bytes):
        r = client.post(
            "/api/generate",
            files={"file": ("notes.docx", io.BytesIO(docx_bytes), "application/vnd.openxmlformats")},
        )
        assert r.status_code == 200

    def test_pptx_upload_returns_200(self, client, mock_llm, pptx_bytes):
        r = client.post(
            "/api/generate",
            files={"file": ("slides.pptx", io.BytesIO(pptx_bytes), "application/vnd.openxmlformats")},
        )
        assert r.status_code == 200

    def test_pdf_upload_returns_200(self, client, mock_llm, pdf_bytes):
        r = client.post(
            "/api/generate",
            files={"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Video URL  (YouTube + non-YouTube)
# ---------------------------------------------------------------------------

class TestVideoUrl:
    def test_youtube_url_returns_200(self, client, mock_llm, mock_transcribe):
        r = client.post("/api/generate", data={
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        })
        assert r.status_code == 200

    def test_vimeo_url_returns_200(self, client, mock_llm, mock_transcribe):
        """Non-YouTube URLs must no longer be rejected at validation."""
        r = client.post("/api/generate", data={
            "video_url": "https://vimeo.com/123456789",
        })
        assert r.status_code == 200

    def test_input_type_is_video(self, client, mock_llm, mock_transcribe):
        body = client.post("/api/generate", data={
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }).json()
        assert body["input_type"] == "video"

    def test_transcript_field_populated(self, client, mock_llm, mock_transcribe):
        body = client.post("/api/generate", data={
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }).json()
        assert "transcript" in body
        assert body["transcript"] is not None
        assert len(body["transcript"]) > 0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_second_call_uses_cache(self, client, mock_llm, mock_transcribe):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        r1 = client.post("/api/generate", data={"video_url": url})
        r2 = client.post("/api/generate", data={"video_url": url})
        assert r1.status_code == r2.status_code == 200
        assert mock_llm.call_count == 1  # LLM called only once; second hit from cache

    def test_cache_returns_identical_response(self, client, mock_llm, mock_transcribe):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        r1 = client.post("/api/generate", data={"video_url": url}).json()
        r2 = client.post("/api/generate", data={"video_url": url}).json()
        assert r1["course"]["title"] == r2["course"]["title"]

    def test_different_urls_not_shared(self, client, mock_llm, mock_transcribe):
        url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        url2 = "https://www.youtube.com/watch?v=xxxxxxxxxxx"
        client.post("/api/generate", data={"video_url": url1})
        client.post("/api/generate", data={"video_url": url2})
        assert mock_llm.call_count == 2

    def test_plain_text_not_cached(self, client, mock_llm):
        client.post("/api/generate", data={"text": ENGLISH_TEXT})
        client.post("/api/generate", data={"text": ENGLISH_TEXT})
        assert mock_llm.call_count == 2

    def test_health_shows_cache_entries(self, client, mock_llm, mock_transcribe):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        client.post("/api/generate", data={"video_url": url})
        health = client.get("/api/health").json()
        assert health["cache_entries"] >= 1


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_eleventh_request_is_throttled(self, client, mock_llm, enable_rate_limiting):
        """10 requests must succeed; the 11th from the same IP must be 429."""
        statuses = []
        for _ in range(11):
            r = client.post("/api/generate", data={"text": ENGLISH_TEXT})
            statuses.append(r.status_code)
        assert all(s == 200 for s in statuses[:10]), f"First 10 statuses: {statuses[:10]}"
        assert statuses[10] == 429, f"Expected 429 on 11th, got {statuses[10]}"

    def test_rate_limit_response_is_json(self, client, mock_llm, enable_rate_limiting):
        for _ in range(11):
            r = client.post("/api/generate", data={"text": ENGLISH_TEXT})
        assert r.headers.get("content-type", "").startswith("application/json")


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------

class TestErrorMessages:
    def test_all_errors_have_ar_message(self, client):
        cases = [
            ({"": ""}, "NO_INPUT"),
            ({"text": "short"}, "TEXT_TOO_SHORT"),
            ({"video_url": "not-a-valid-url"}, "INVALID_URL"),
        ]
        for data, expected_code in cases:
            r = client.post("/api/generate", data=data)
            body = r.json()
            assert "ar_message" in body, f"Missing ar_message for {expected_code}"
            assert len(body["ar_message"]) > 5, f"ar_message too short for {expected_code}"

    def test_all_errors_have_error_code(self, client):
        r = client.post("/api/generate", data={})
        assert "error_code" in r.json()

    def test_all_errors_have_message(self, client):
        r = client.post("/api/generate", data={})
        assert "message" in r.json()


# ---------------------------------------------------------------------------
# Integration — real YouTube end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestYouTubeEndToEnd:
    def test_real_video_returns_success(self, client):
        r = client.post("/api/generate", data={
            "video_url": "https://www.youtube.com/watch?v=aircAruvnKk",
            "num_quiz_questions": 5,
        }, timeout=180)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["input_type"] == "video"
        assert body["transcript"] is not None
        assert len(body["transcript"]) > 0
        assert "course" in body
        assert "title" in body["course"]
        assert "description" in body["course"]
        assert isinstance(body["course"]["quiz"], list)
        assert len(body["course"]["quiz"]) >= 1
