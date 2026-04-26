"""Unit + integration tests for services/transcriber.py"""
import pytest
from unittest.mock import MagicMock, patch

from services.transcriber import _try_youtube_captions, _postprocess_whisper


# ---------------------------------------------------------------------------
# _postprocess_whisper  (pure function — no mocking needed)
# ---------------------------------------------------------------------------

class TestPostprocessWhisper:
    def test_arabic_to_latin_boundary_gets_space(self):
        result = _postprocess_whisper("شبكةTCP")
        assert "شبكة TCP" in result

    def test_latin_to_arabic_boundary_gets_space(self):
        result = _postprocess_whisper("TCPشبكة")
        assert "TCP شبكة" in result

    def test_collapses_multiple_spaces(self):
        result = _postprocess_whisper("word  word")
        assert "  " not in result

    def test_mid_sentence_newline_becomes_space(self):
        # A lone newline inside a sentence should become a space
        result = _postprocess_whisper("first part\nsecond part")
        assert "\nfirst" not in result
        assert "first part second part" in result

    def test_double_newline_paragraph_break_preserved(self):
        result = _postprocess_whisper("para one\n\npara two")
        assert "\n\n" in result

    def test_nfkc_normalisation_applied(self):
        # Arabic ligature U+FEFB should not survive NFKC normalisation
        result = _postprocess_whisper("\ufefb test")
        assert "\ufefb" not in result

    def test_empty_string_returns_empty(self):
        assert _postprocess_whisper("") == ""

    def test_english_only_unchanged_structure(self):
        text = "Hello world. This is a test."
        result = _postprocess_whisper(text)
        assert "Hello world" in result

    def test_does_not_split_acronyms(self):
        # "I P  address" should become "IP address" (double space between caps collapsed)
        result = _postprocess_whisper("I P  address")
        assert "  " not in result


# ---------------------------------------------------------------------------
# _try_youtube_captions  (mocked)
# ---------------------------------------------------------------------------

class TestTryYoutubeCaptionsMocked:
    def _make_mock_api(self, texts: list[str], lang: str = "en"):
        entries = []
        for t in texts:
            m = MagicMock()
            m.text = t
            entries.append(m)

        transcript = MagicMock()
        transcript.fetch.return_value = entries
        transcript.language_code = lang

        tlist = MagicMock()
        tlist.find_transcript.return_value = transcript
        # Make iter(tlist) yield the transcript too (for fallback path)
        tlist.__iter__ = MagicMock(return_value=iter([transcript]))

        api_instance = MagicMock()
        api_instance.list.return_value = tlist
        return api_instance

    def test_successful_fetch_returns_text_and_lang(self):
        words = ["machine learning"] * 30   # 60 words
        api = self._make_mock_api(words, lang="en")
        with patch("services.transcriber.YouTubeTranscriptApi", return_value=api):
            result = _try_youtube_captions("https://youtube.com/watch?v=abcdefghijk")
        assert result is not None
        text, lang = result
        assert lang == "en"
        assert len(text.split()) >= 20

    def test_short_transcript_returns_none(self):
        # Only 5 "hello" entries → 5 words < 20
        api = self._make_mock_api(["hello"] * 5)
        with patch("services.transcriber.YouTubeTranscriptApi", return_value=api):
            result = _try_youtube_captions("https://youtube.com/watch?v=abcdefghijk")
        assert result is None

    def test_api_exception_returns_none(self):
        api = MagicMock()
        api.list.side_effect = Exception("Network error")
        with patch("services.transcriber.YouTubeTranscriptApi", return_value=api):
            result = _try_youtube_captions("https://youtube.com/watch?v=abcdefghijk")
        assert result is None

    def test_find_transcript_falls_back_to_iter(self):
        """If find_transcript raises, should fall back to first available."""
        words = ["deep learning is great"] * 10   # 40 words
        entries = [MagicMock(text=w) for w in words]

        transcript = MagicMock()
        transcript.fetch.return_value = entries
        transcript.language_code = "en"

        tlist = MagicMock()
        tlist.find_transcript.side_effect = Exception("not found")
        tlist.__iter__ = MagicMock(return_value=iter([transcript]))

        api = MagicMock()
        api.list.return_value = tlist

        with patch("services.transcriber.YouTubeTranscriptApi", return_value=api):
            result = _try_youtube_captions("https://youtube.com/watch?v=abcdefghijk")
        assert result is not None

    def test_url_without_video_id_returns_none(self):
        result = _try_youtube_captions("https://youtube.com/")
        assert result is None

    def test_plain_invalid_url_returns_none(self):
        result = _try_youtube_captions("not_a_url")
        assert result is None

    def test_arabic_captions_returned_with_ar_lang(self):
        arabic_words = ["مرحبا بالعالم"] * 25
        api = self._make_mock_api(arabic_words, lang="ar")
        with patch("services.transcriber.YouTubeTranscriptApi", return_value=api):
            result = _try_youtube_captions("https://youtube.com/watch?v=abcdefghijk")
        assert result is not None
        _, lang = result
        assert lang == "ar"


# ---------------------------------------------------------------------------
# Integration — real YouTube network call
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestYoutubeCaptionsIntegration:
    def test_real_arabic_video_returns_sufficient_text(self):
        """3Blue1Brown neural networks (Arabic-subtitled) — needs network."""
        url = "https://www.youtube.com/watch?v=aircAruvnKk"
        result = _try_youtube_captions(url)
        assert result is not None, "Expected captions but got None"
        text, lang = result
        assert len(text.split()) >= 20
        assert isinstance(lang, str)
        assert len(lang) >= 2
