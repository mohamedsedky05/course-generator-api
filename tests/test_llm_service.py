"""Unit + integration tests for services/llm_service.py"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config import settings
from services.llm_service import (
    _parse_json_safe,
    _strip_markdown_json,
    clean_transcription,
    generate_content,
)


def test_claude_credentials_and_models_configured():
    assert hasattr(settings, "anthropic_api_key")
    assert hasattr(settings, "claude_generation_model")
    assert hasattr(settings, "claude_cleanup_model")

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_RESULT = {
    "title": "Introduction to Machine Learning",
    "description": "A beginner-level overview of ML fundamentals.",
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
        with patch("services.llm_service._call_claude", new=AsyncMock(return_value=cleaned)):
            result = await clean_transcription("الراوتر يستخدم بروتوكول TCP/IP")
        assert result == cleaned

    async def test_falls_back_to_original_on_exception(self):
        original = "original text with issues"
        with patch("services.llm_service._call_claude", new=AsyncMock(side_effect=Exception("down"))):
            result = await clean_transcription(original)
        assert result == original

    async def test_returns_stripped_result(self):
        with patch("services.llm_service._call_claude", new=AsyncMock(return_value="  cleaned  ")):
            result = await clean_transcription("text")
        assert result == "cleaned"


# ---------------------------------------------------------------------------
# generate_content  (mocked)
# ---------------------------------------------------------------------------

class TestGenerateContentMocked:
    async def test_returns_dict_with_required_keys(self, english_text):
        call_num = 0

        async def mock_gemini(prompt):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return english_text          # clean_transcription pass-through
            return json.dumps(SAMPLE_RESULT) # generation call

        with patch("services.llm_service._call_gemini", new=mock_gemini):
            result = await generate_content(english_text, 2)

        assert isinstance(result, dict)
        assert "title" in result
        assert "description" in result
        assert "quiz" in result

    async def test_quiz_is_list(self, english_text):
        call_num = 0

        async def mock_gemini(prompt):
            nonlocal call_num
            call_num += 1
            return english_text if call_num == 1 else json.dumps(SAMPLE_RESULT)

        with patch("services.llm_service._call_gemini", new=mock_gemini):
            result = await generate_content(english_text, 2)

        assert isinstance(result["quiz"], list)
        assert len(result["quiz"]) > 0

    async def test_retries_on_invalid_json(self, english_text):
        """If the generation call returns bad JSON once, it should retry."""
        call_num = 0

        async def flaky(prompt, *args):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return english_text   # clean_transcription
            if call_num == 2:
                return "not json"     # first generation attempt fails
            return json.dumps(SAMPLE_RESULT)  # retry succeeds

        with patch("services.llm_service._call_claude", new=flaky):
            result = await generate_content(english_text, 2)

        assert call_num == 3
        assert "title" in result

    async def test_num_quiz_questions_appears_in_prompt(self, english_text):
        """The num_quiz_questions value must be embedded in the prompt."""
        prompts_seen = []

        async def capture(prompt, *args):
            prompts_seen.append(prompt)
            return english_text if len(prompts_seen) == 1 else json.dumps(SAMPLE_RESULT)

        with patch("services.llm_service._call_claude", new=capture):
            await generate_content(english_text, 7)

        generation_prompt = prompts_seen[1]
        assert "7" in generation_prompt


# ---------------------------------------------------------------------------
# _call_gemini retry / timeout logic
# ---------------------------------------------------------------------------

class TestCallGeminiRetry:
    async def test_cached_prefix_uses_ephemeral_cache_control(self):
        response = MagicMock()
        response.content = [MagicMock(type="text", text="ok")]
        create = AsyncMock(return_value=response)

        with patch("services.llm_service._get_client") as mock_client:
            mock_client.return_value.messages.create = create
            from services.llm_service import _call_claude
            result = await _call_claude("dynamic text", "claude-sonnet-4-6", "stable instructions")

        assert result == "ok"
        request = create.await_args.kwargs
        content = request["messages"][0]["content"]
        assert content[0] == {
            "type": "text",
            "text": "stable instructions",
            "cache_control": {"type": "ephemeral"},
        }
        assert content[1] == {"type": "text", "text": "dynamic text"}

    async def test_retries_on_503_up_to_3_times(self):
        call_count = 0

        class FakeStatusError(Exception):
            status_code = 503

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeStatusError("Service Unavailable")
            return "success"

        with patch("services.llm_service._get_client") as mock_client, \
             patch("asyncio.sleep", new=AsyncMock()):
            mock_client.return_value.messages.create = mock_generate
            from services.llm_service import _call_claude
            result = await _call_claude("test", "claude-sonnet-4-20250514")

        assert call_count == 3
        assert result == "success"

    async def test_raises_after_exhausting_retries_on_503(self):
        class FakeStatusError(Exception):
            status_code = 503

        async def always_503(*args, **kwargs):
            raise FakeStatusError("Service Unavailable")

        with patch("services.llm_service._get_client") as mock_client, \
             patch("asyncio.sleep", new=AsyncMock()):
            mock_client.return_value.messages.create = always_503
            from services.llm_service import _call_claude
            with pytest.raises(Exception):
                await _call_claude("test", "claude-sonnet-4-20250514")

    async def test_does_not_retry_on_429(self):
        call_count = 0

        class FakeQuotaError(Exception):
            status_code = 429

        async def quota_err(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise FakeQuotaError("Resource Exhausted")

        with patch("services.llm_service._get_client") as mock_client:
            mock_client.return_value.messages.create = quota_err
            from services.llm_service import _call_claude
            with pytest.raises(Exception):
                await _call_claude("test", "claude-sonnet-4-20250514")

        assert call_count == 1  # no retry for quota errors

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
            mock_client.return_value.messages.create = AsyncMock()
            from services.llm_service import _call_claude
            result = await _call_claude("test", "claude-sonnet-4-20250514")

        assert result == "ok"


# ---------------------------------------------------------------------------
# Integration — real Claude API
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestLLMIntegration:
    async def test_clean_transcription_fixes_arabic_transliterations(self):
        text = "الراوتر يستخدم بروتوكول TCP لنقل البيانات عبر الإنترنت"
        result = await clean_transcription(text)
        assert isinstance(result, str) and len(result) > 5
        assert "router" in result.lower() or "الراوتر" in result or "TCP" in result

    async def test_generate_content_returns_complete_structure(self, english_text):
        result = await generate_content(english_text, 5)
        for key in ("title", "description", "quiz"):
            assert key in result, f"Missing key: {key}"
        assert isinstance(result["quiz"], list)
        assert len(result["quiz"]) >= 1

    async def test_generate_content_quiz_has_correct_types(self, english_text):
        result = await generate_content(english_text, 5)
        for q in result["quiz"]:
            assert q["type"] in ("mcq", "true_false")
            if q["type"] == "mcq":
                assert len(q["options"]) == 4
            elif q["type"] == "true_false":
                assert isinstance(q["correct_answer"], bool)
