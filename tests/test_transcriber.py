"""Unit + integration tests for services/transcriber.py"""
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from services.transcriber import (
    _try_youtube_captions,
    _postprocess_whisper,
    _detect_url_type,
    _extract_video_from_iframe,
)


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
        result = _postprocess_whisper("first part\nsecond part")
        assert "\nfirst" not in result
        assert "first part second part" in result

    def test_double_newline_paragraph_break_preserved(self):
        result = _postprocess_whisper("para one\n\npara two")
        assert "\n\n" in result

    def test_nfkc_normalisation_applied(self):
        result = _postprocess_whisper("ﻻ test")
        assert "ﻻ" not in result

    def test_empty_string_returns_empty(self):
        assert _postprocess_whisper("") == ""

    def test_english_only_unchanged_structure(self):
        text = "Hello world. This is a test."
        result = _postprocess_whisper(text)
        assert "Hello world" in result

    def test_does_not_split_acronyms(self):
        result = _postprocess_whisper("I P  address")
        assert "  " not in result


# ---------------------------------------------------------------------------
# _detect_url_type
# ---------------------------------------------------------------------------

class TestDetectUrlType:
    # ── YouTube ───────────────────────────────────────────────────────────

    def test_youtube_watch_url(self):
        assert _detect_url_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube"

    def test_youtube_short_url(self):
        assert _detect_url_type("https://youtu.be/dQw4w9WgXcQ") == "youtube"

    def test_youtube_embed_url(self):
        assert _detect_url_type("https://www.youtube.com/embed/dQw4w9WgXcQ") == "youtube"

    def test_youtube_no_www(self):
        assert _detect_url_type("https://youtube.com/watch?v=abc") == "youtube"

    def test_youtube_mobile(self):
        assert _detect_url_type("https://m.youtube.com/watch?v=abc") == "youtube"

    # ── iframe CDN providers ───────────────────────────────────────────────

    def test_bunnynet_cdn(self):
        url = "https://iframe.mediadelivery.net/embed/578375/3fbdf5a1?token=xyz"
        assert _detect_url_type(url) == "iframe"

    def test_wistia_embed(self):
        assert _detect_url_type("https://fast.wistia.net/embed/iframe/abc123") == "iframe"

    def test_jwplayer_cdn(self):
        assert _detect_url_type("https://cdn.jwplayer.com/players/abc-xyz.html") == "iframe"

    def test_vimeo_player_embed(self):
        # player.vimeo.com is the embed player, distinct from vimeo.com (direct)
        assert _detect_url_type("https://player.vimeo.com/video/123456") == "iframe"

    def test_brightcove_player(self):
        assert _detect_url_type("https://player.brightcove.net/abc/default_default/index.html") == "iframe"

    # ── direct / other sites ──────────────────────────────────────────────

    def test_vimeo_direct(self):
        # vimeo.com (not player.vimeo.com) → yt-dlp handles it natively
        assert _detect_url_type("https://vimeo.com/123456789") == "direct"

    def test_facebook_video(self):
        assert _detect_url_type("https://www.facebook.com/video/123") == "direct"

    def test_tiktok_video(self):
        assert _detect_url_type("https://www.tiktok.com/@user/video/123") == "direct"

    def test_dailymotion(self):
        assert _detect_url_type("https://www.dailymotion.com/video/abc") == "direct"

    def test_plain_mp4_url(self):
        assert _detect_url_type("https://cdn.example.com/video.mp4") == "direct"

    def test_ftp_url_is_direct(self):
        # ftp:// isn't a known host — routing falls through to "direct"
        # (the router rejects it before it reaches the transcriber)
        assert _detect_url_type("ftp://example.com/video.mp4") == "direct"

    def test_malformed_url_is_direct(self):
        assert _detect_url_type("not-a-url") == "direct"


# ---------------------------------------------------------------------------
# _extract_video_from_iframe  (mocked HTTP)
# ---------------------------------------------------------------------------

class TestExtractVideoFromIframe:
    """Test the HTML-parsing logic with controlled mock HTTP responses."""

    _BUNNY_HTML = """
    <html><head><title>Video Player</title></head>
    <body>
    <script>
      var player = new bunny.player({
        sources: [{
          src: "https://vz-abc123.b-cdn.net/video-id/play_720p.mp4",
          type: "video/mp4"
        }]
      });
    </script>
    </body></html>
    """

    _SOURCE_TAG_HTML = """
    <html><body>
    <video controls>
      <source src="https://cdn.example.com/lecture.mp4" type="video/mp4">
    </video>
    </body></html>
    """

    _JWPLAYER_HTML = """
    <html><body>
    <script>
      jwplayer("player").setup({
        file: "https://media.example.com/stream/video.m3u8",
        type: "hls"
      });
    </script>
    </body></html>
    """

    _HLS_ONLY_HTML = """
    <html><body>
    <script>
      var streamUrl = 'https://live.example.com/hls/playlist.m3u8?token=abc123';
    </script>
    </body></html>
    """

    _NO_VIDEO_HTML = """
    <html><body><p>This page has no video source.</p></body></html>
    """

    def _mock_resp(self, html: str, status: int = 200):
        m = MagicMock()
        m.text = html
        m.status_code = status
        return m

    def test_extracts_bunnynet_mp4(self):
        with patch("httpx.get", return_value=self._mock_resp(self._BUNNY_HTML)):
            url = _extract_video_from_iframe("https://iframe.mediadelivery.net/embed/1/abc")
        assert url is not None
        assert ".b-cdn.net" in url
        assert url.endswith(".mp4")

    def test_extracts_html5_source_tag(self):
        with patch("httpx.get", return_value=self._mock_resp(self._SOURCE_TAG_HTML)):
            url = _extract_video_from_iframe("https://iframe.mediadelivery.net/embed/1/abc")
        assert url == "https://cdn.example.com/lecture.mp4"

    def test_extracts_jwplayer_hls(self):
        with patch("httpx.get", return_value=self._mock_resp(self._JWPLAYER_HTML)):
            url = _extract_video_from_iframe("https://cdn.jwplayer.com/players/abc.html")
        assert url is not None
        assert ".m3u8" in url

    def test_extracts_generic_hls_stream(self):
        with patch("httpx.get", return_value=self._mock_resp(self._HLS_ONLY_HTML)):
            url = _extract_video_from_iframe("https://iframe.mediadelivery.net/embed/1/abc")
        assert url is not None
        assert ".m3u8" in url

    def test_returns_none_when_no_video_found(self):
        with patch("httpx.get", return_value=self._mock_resp(self._NO_VIDEO_HTML)):
            result = _extract_video_from_iframe("https://iframe.mediadelivery.net/embed/1/abc")
        assert result is None

    def test_returns_none_on_http_exception(self):
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            result = _extract_video_from_iframe("https://iframe.mediadelivery.net/embed/1/abc")
        assert result is None

    def test_protocol_relative_url_made_absolute(self):
        html = '<source src="//cdn.example.com/video.mp4" type="video/mp4">'
        with patch("httpx.get", return_value=self._mock_resp(html)):
            url = _extract_video_from_iframe("https://player.example.com/embed/1")
        assert url is not None
        assert url.startswith("https://cdn.example.com")

    def test_path_relative_url_made_absolute(self):
        html = '<source src="/media/video.mp4" type="video/mp4">'
        with patch("httpx.get", return_value=self._mock_resp(html)):
            url = _extract_video_from_iframe("https://player.example.com/embed/1")
        assert url is not None
        assert url.startswith("https://player.example.com/media/video.mp4")

    def test_timeout_exception_returns_none(self):
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = _extract_video_from_iframe("https://fast.wistia.net/embed/iframe/abc")
        assert result is None


# ---------------------------------------------------------------------------
# _try_youtube_captions  (mocked)
# ---------------------------------------------------------------------------

class TestTryYoutubeCaptionsMocked:
    def _make_mock_api(self, texts: list[str], lang: str = "en"):
        entries = [MagicMock(text=t) for t in texts]

        transcript = MagicMock()
        transcript.fetch.return_value = entries
        transcript.language_code = lang

        tlist = MagicMock()
        tlist.find_transcript.return_value = transcript
        tlist.__iter__ = MagicMock(return_value=iter([transcript]))

        api_instance = MagicMock()
        api_instance.list.return_value = tlist
        return api_instance

    def test_successful_fetch_returns_text_and_lang(self):
        api = self._make_mock_api(["machine learning"] * 30, lang="en")
        with patch("services.transcriber.YouTubeTranscriptApi", return_value=api):
            result = _try_youtube_captions("https://youtube.com/watch?v=abcdefghijk")
        assert result is not None
        text, lang = result
        assert lang == "en"
        assert len(text.split()) >= 20

    def test_short_transcript_returns_none(self):
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
        entries = [MagicMock(text="deep learning is great") for _ in range(10)]
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
        api = self._make_mock_api(["مرحبا بالعالم"] * 25, lang="ar")
        with patch("services.transcriber.YouTubeTranscriptApi", return_value=api):
            result = _try_youtube_captions("https://youtube.com/watch?v=abcdefghijk")
        assert result is not None
        _, lang = result
        assert lang == "ar"


# ---------------------------------------------------------------------------
# transcribe_video routing logic  (mocked internals)
# ---------------------------------------------------------------------------

class TestTranscribeVideoRouting:
    """Verify that transcribe_video dispatches correctly based on URL type."""

    async def test_youtube_url_tries_captions_first(self):
        """For a YouTube URL the caption path must be attempted before pytubefix."""
        from services.transcriber import transcribe_video

        caption_mock = MagicMock(return_value=("transcription text " * 20, "en"))

        with patch("services.transcriber._try_youtube_captions", caption_mock), \
             patch("services.transcriber._download_audio_youtube") as dl_mock:
            result = await transcribe_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        caption_mock.assert_called_once()
        dl_mock.assert_not_called()   # pytubefix must NOT have been called
        assert result[0].startswith("transcription text")

    async def test_youtube_falls_back_to_pytubefix_when_no_captions(self):
        """If captions return None the transcriber must fall through to pytubefix + Groq."""
        from services.transcriber import transcribe_video

        with patch("services.transcriber._try_youtube_captions", return_value=None), \
             patch("services.transcriber._download_audio_youtube",
                   return_value="/tmp/a.mp4") as dl_mock, \
             patch("services.transcriber._transcribe_with_groq_sync",
                   return_value=("groq text " * 10, "en")):
            result = await transcribe_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        dl_mock.assert_called_once()
        assert "groq" in result[0]

    async def test_iframe_url_tries_html_extraction_first(self):
        """For an iframe URL, _extract_video_from_iframe must be called."""
        from services.transcriber import transcribe_video

        extracted_url = "https://vz-abc.b-cdn.net/video.mp4"

        with patch("services.transcriber._extract_video_from_iframe",
                   return_value=extracted_url) as extract_mock, \
             patch("services.transcriber._download_audio_ytdlp", return_value="/tmp/a.mp3"), \
             patch("services.transcriber._transcribe_with_groq_sync",
                   return_value=("audio text " * 10, "en")):
            result = await transcribe_video(
                "https://iframe.mediadelivery.net/embed/578375/abc?token=xyz"
            )

        extract_mock.assert_called_once()
        assert "audio" in result[0]

    async def test_iframe_passes_iframe_url_to_ytdlp_when_extraction_fails(self):
        """If HTML extraction returns None the iframe URL itself goes to yt-dlp."""
        from services.transcriber import transcribe_video

        iframe_url = "https://iframe.mediadelivery.net/embed/578375/abc"

        with patch("services.transcriber._extract_video_from_iframe", return_value=None), \
             patch("services.transcriber._download_audio_ytdlp",
                   return_value="/tmp/a.mp3") as dl_mock, \
             patch("services.transcriber._transcribe_with_groq_sync",
                   return_value=("audio text " * 10, "en")):
            await transcribe_video(iframe_url)

        # yt-dlp must have been called with the original iframe URL
        dl_mock.assert_called_once()
        called_url = dl_mock.call_args[0][0]
        assert called_url == iframe_url

    async def test_direct_url_goes_straight_to_ytdlp(self):
        """Vimeo and other direct URLs skip both captions and iframe extraction."""
        from services.transcriber import transcribe_video

        with patch("services.transcriber._try_youtube_captions") as cap_mock, \
             patch("services.transcriber._extract_video_from_iframe") as iframe_mock, \
             patch("services.transcriber._download_audio_ytdlp", return_value="/tmp/a.mp3"), \
             patch("services.transcriber._transcribe_with_groq_sync",
                   return_value=("vimeo text " * 10, "en")):
            result = await transcribe_video("https://vimeo.com/123456789")

        cap_mock.assert_not_called()
        iframe_mock.assert_not_called()
        assert "vimeo" in result[0]


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
