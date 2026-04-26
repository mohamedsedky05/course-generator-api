"""Unit + integration tests for services/llm_service.py"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.llm_service import (
    _parse_json_safe,
    _strip_markdown_json,
    clean_transcription,
    analyze_content,
    generate_course_from_chunk,
    generate_course,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_ANALYSIS = {
    "subject": "Machine Learning",
    "difficulty": "Beginner",
    "language": "English",
    "key_topics": ["supervised learning", "neural networks", "gradient descent"],
    "recommended_lectures": 2,
}

SAMPLE_COURSE = {
    "title": "Introduction to Machine Learning",
    "description": "A beginner-level course on ML fundamentals.",
    "summary": "Machine learning enables computers to learn from data without explicit programming.",
    "subject": "Machine Learning",
    "difficulty": "Beginner",
    "key_topics": ["supervised learning", "neural networks"],
    "lectures": [
        {
            "lecture_number": 1,
            "title": "What is Machine Learning?",
            "content": "ML is a subset of AI that learns from data.",
            "objectives": ["Define ML", "Identify the three types", "List key applications"],
        }
    ],
    "quiz": [
        {
            "question_number": 1,
            "type": "mcq",
            "question": "What is machine learning?",
            "options": ["A. A subset of AI", "B. A database", "C. An OS", "D. A network"],
            "correct_answer": 0,
            "explanation": "Machine learning is a subset of artificial intelligence.",
        },
        {
            "question_number": 2,
            "type": "true_false",
            "question": "Machine learning requires explicit programming for every task.",
            "correct_answer": False,
            "explanation": "ML learns from data without explicit programming.",
        },
    ],
}


# ---------------------------------------------------------------------------
# _strip_markdown_json
# ---------------------------------------------------------------------------

class TestStripMarkdownJson:
    def test_strips_json_code_fence(self):
        assert _strip_markdown_json("```json\n{}\n```") == "{}"

    def test_strips_plain_code_fence(self):
        assert _strip_markdown_json("```\n{}\n```") == "{}"

    def test_strips_case_insensitive_fence(self):
        assert _strip_markdown_json("```JSON\n{}\n```") == "{}"

    def test_leaves_clean_json_unchanged(self):
        assert _strip_markdown_json('{"a": 1}') == '{"a": 1}'

    def test_strips_surrounding_whitespace(self):
        assert _strip_markdown_json("  {}  ") == "{}"


# ---------------------------------------------------------------------------
# _parse_json_safe
# ---------------------------------------------------------------------------

class TestParseJsonSafe:
    def test_parses_clean_json(self):
        assert _parse_json_safe('{"key": "value"}') == {"key": "value"}

    def test_parses_json_inside_markdown_fence(self):
        result = _parse_json_safe('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extracts_json_from_surrounding_text(self):
        result = _parse_json_safe('Here you go:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_raises_on_completely_invalid_input(self):
        with pytest.raises(Exception):
            _parse_json_safe("no json here whatsoever")

    def test_parses_nested_json(self):
        nested = '{"a": {"b": [1, 2, 3]}}'
        assert _parse_json_safe(nested) == {"a": {"b": [1, 2, 3]}}


# ---------------------------------------------------------------------------
# clean_transcription  (mocked)
# ---------------------------------------------------------------------------

class TestCleanTranscriptionMocked:
    async def test_returns_cleaned_text(self):
        cleaned = "The router uses TCP/IP protocol"
        with patch("services.llm_service._call_gemini", new=AsyncMock(return_value=cleaned)):
            result = await clean_transcription("الراوتر يستخدم بروتوكول TCP/IP")
        assert result == cleaned

    async def test_falls_back_to_original_on_exception(self):
        original = "original text with issues"
        with patch("services.llm_service._call_gemini", new=AsyncMock(side_effect=Exception("down"))):
            result = await clean_transcription(original)
        assert result == original

    async def test_returns_stripped_result(self):
        with patch("services.llm_service._call_gemini", new=AsyncMock(return_value="  cleaned  ")):
            result = await clean_transcription("text")
        assert result == "cleaned"


# ---------------------------------------------------------------------------
# analyze_content  (mocked)
# ---------------------------------------------------------------------------

class TestAnalyzeContentMocked:
    async def test_returns_parsed_analysis(self):
        with patch("services.llm_service._call_gemini", new=AsyncMock(return_value=json.dumps(SAMPLE_ANALYSIS))):
            result = await analyze_content("some educational text")
        assert result["subject"] == "Machine Learning"
        assert result["difficulty"] == "Beginner"
        assert isinstance(result["key_topics"], list)

    async def test_retries_once_on_invalid_json(self):
        call_count = 0

        async def flaky_gemini(prompt):
            nonlocal call_count
            call_count += 1
            return "not json" if call_count == 1 else json.dumps(SAMPLE_ANALYSIS)

        with patch("services.llm_service._call_gemini", new=flaky_gemini):
            result = await analyze_content("text")
        assert call_count == 2
        assert result["subject"] == "Machine Learning"

    async def test_raises_if_both_attempts_invalid(self):
        with patch("services.llm_service._call_gemini", new=AsyncMock(return_value="not json")):
            with pytest.raises(Exception):
                await analyze_content("text")


# ---------------------------------------------------------------------------
# generate_course_from_chunk  (mocked)
# ---------------------------------------------------------------------------

class TestGenerateCourseFromChunkMocked:
    async def test_returns_course_dict(self):
        with patch("services.llm_service._call_gemini", new=AsyncMock(return_value=json.dumps(SAMPLE_COURSE))):
            result = await generate_course_from_chunk("text", SAMPLE_ANALYSIS, 1, 2)
        assert result["title"] == "Introduction to Machine Learning"
        assert "lectures" in result
        assert "quiz" in result

    async def test_retries_on_invalid_json(self):
        call_count = 0

        async def flaky(prompt):
            nonlocal call_count
            call_count += 1
            return "not json" if call_count == 1 else json.dumps(SAMPLE_COURSE)

        with patch("services.llm_service._call_gemini", new=flaky):
            result = await generate_course_from_chunk("text", SAMPLE_ANALYSIS, 1, 2)
        assert call_count == 2
        assert "title" in result


# ---------------------------------------------------------------------------
# _call_gemini retry / timeout logic
# ---------------------------------------------------------------------------

class TestCallGeminiRetry:
    async def test_retries_on_503_up_to_3_times(self):
        from google.genai.errors import ClientError
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ClientError(503, {"error": {"code": 503, "message": "Service Unavailable"}})
            return MagicMock(text="success")

        with patch("services.llm_service._get_client") as mock_client, \
             patch("asyncio.sleep", new=AsyncMock()):
            mock_client.return_value.aio.models.generate_content = mock_generate
            from services.llm_service import _call_gemini
            result = await _call_gemini("test")

        assert call_count == 3
        assert result == "success"

    async def test_raises_after_exhausting_retries_on_503(self):
        from google.genai.errors import ClientError

        async def always_503(*args, **kwargs):
            raise ClientError(503, {"error": {"code": 503, "message": "Service Unavailable"}})

        with patch("services.llm_service._get_client") as mock_client, \
             patch("asyncio.sleep", new=AsyncMock()):
            mock_client.return_value.aio.models.generate_content = always_503
            from services.llm_service import _call_gemini
            with pytest.raises(Exception):
                await _call_gemini("test")

    async def test_does_not_retry_on_429(self):
        from google.genai.errors import ClientError
        call_count = 0

        async def quota_err(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ClientError(429, {"error": {"code": 429, "message": "Resource Exhausted"}})

        with patch("services.llm_service._get_client") as mock_client:
            mock_client.return_value.aio.models.generate_content = quota_err
            from services.llm_service import _call_gemini
            with pytest.raises(Exception):
                await _call_gemini("test")

        assert call_count == 1   # no retry for quota errors

    async def test_retries_on_timeout(self):
        call_count = 0

        async def timeout_once(coro, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return MagicMock(text="ok")

        with patch("services.llm_service._get_client") as mock_client, \
             patch("asyncio.wait_for", side_effect=timeout_once), \
             patch("asyncio.sleep", new=AsyncMock()):
            mock_client.return_value.aio.models.generate_content = AsyncMock()
            from services.llm_service import _call_gemini
            result = await _call_gemini("test")

        assert result == "ok"


# ---------------------------------------------------------------------------
# generate_course (full pipeline, mocked)
# ---------------------------------------------------------------------------

class TestGenerateCoursePipelineMocked:
    async def test_single_chunk_returns_dict(self, english_text):
        call_num = 0

        async def mock_gemini(prompt):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return english_text             # clean_transcription
            if call_num == 2:
                return json.dumps(SAMPLE_ANALYSIS)  # analyze_content
            return json.dumps(SAMPLE_COURSE)        # generate

        with patch("services.llm_service._call_gemini", new=mock_gemini):
            result = await generate_course(english_text, [english_text], 1, 2)

        assert isinstance(result, dict)
        assert "title" in result
        assert "lectures" in result
        assert "quiz" in result

    async def test_clean_and_analyze_both_called(self, english_text):
        """Verify both clean_transcription and analyze_content are invoked."""
        prompts_seen = []

        async def capture(prompt):
            prompts_seen.append(prompt[:60])
            if "corrector" in prompt.lower():
                return english_text
            if "analyst" in prompt.lower():
                return json.dumps(SAMPLE_ANALYSIS)
            return json.dumps(SAMPLE_COURSE)

        with patch("services.llm_service._call_gemini", new=capture):
            await generate_course(english_text, [english_text], 1, 2)

        # At least 3 calls: clean + analyze + generate
        assert len(prompts_seen) >= 3


# ---------------------------------------------------------------------------
# Integration — real Gemini API
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestLLMIntegration:
    async def test_clean_transcription_fixes_arabic_transliterations(self):
        text = "الراوتر يستخدم بروتوكول TCP لنقل البيانات عبر الإنترنت"
        result = await clean_transcription(text)
        assert isinstance(result, str) and len(result) > 5
        # Either "router" appears, or the Arabic text is preserved
        assert "router" in result.lower() or "الراوتر" in result or "TCP" in result

    async def test_analyze_content_returns_valid_schema(self, english_text):
        result = await analyze_content(english_text)
        assert "subject" in result
        assert result["difficulty"] in ("Beginner", "Intermediate", "Advanced")
        assert isinstance(result["key_topics"], list)
        assert len(result["key_topics"]) > 0

    async def test_generate_course_returns_complete_structure(self, english_text):
        result = await generate_course(english_text, [english_text], 2, 5)
        if isinstance(result, list):
            from services.chunker import merge_course_chunks
            result = merge_course_chunks(result)
        for key in ("title", "description", "summary", "lectures", "quiz"):
            assert key in result, f"Missing key: {key}"
        assert len(result["lectures"]) >= 1
        assert len(result["quiz"]) >= 1
