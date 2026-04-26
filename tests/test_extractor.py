"""Unit tests for services/extractor.py"""
import unicodedata
import pytest

from services.extractor import (
    clean_text,
    detect_language,
    count_words,
    extract_from_docx,
    extract_from_pptx,
    extract_from_txt,
    extract_text_from_file,
    SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_normalises_crlf(self):
        assert "\r\n" not in clean_text("line1\r\nline2")

    def test_normalises_cr(self):
        assert "\r" not in clean_text("line1\rline2")

    def test_collapses_excess_blank_lines(self):
        result = clean_text("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_collapses_spaces_and_tabs(self):
        assert clean_text("word1   \t  word2") == "word1 word2"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("   hello   ") == "hello"

    def test_nfkc_normalisation_applied(self):
        # Arabic ligature ﻻ (U+FEFB) should normalise under NFKC
        text = "\ufefb"
        result = clean_text(text)
        assert result == unicodedata.normalize("NFKC", text).strip()

    def test_arabic_to_latin_boundary_gets_space(self):
        result = clean_text("شبكةTCP")
        assert "شبكة TCP" in result

    def test_latin_to_arabic_boundary_gets_space(self):
        result = clean_text("TCPشبكة")
        assert "TCP شبكة" in result

    def test_ocr_repeated_dots_replaced(self):
        result = clean_text("end......next")
        assert "......" not in result

    def test_ocr_triple_dashes_replaced(self):
        result = clean_text("section---title")
        assert "---" not in result

    def test_punctuation_space_before_removed_for_latin(self):
        # "word ," should become "word,"
        result = clean_text("word ,next")
        assert "word ,next" not in result

    def test_preserves_core_content(self):
        text = "Hello world. This is a test sentence."
        assert "Hello world" in clean_text(text)
        assert "test sentence" in clean_text(text)

    def test_ip_address_preserved(self):
        result = clean_text("server IP is 192.168.1.1 on the network")
        assert "192.168.1.1" in result

    def test_empty_string(self):
        assert clean_text("") == ""


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_detects_english(self, english_text):
        assert detect_language(english_text) == "en"

    def test_detects_arabic(self, arabic_text):
        assert detect_language(arabic_text) == "ar"

    def test_empty_defaults_to_english(self):
        assert detect_language("") == "en"

    def test_numbers_only_defaults_to_english(self):
        assert detect_language("1234 5678 9012") == "en"

    def test_heavy_arabic_mixed(self):
        # Mostly Arabic with a few English words → Arabic
        text = "مرحبا بالعالم في عالم الشبكات والتقنية الحديثة " * 8 + "hello world"
        assert detect_language(text) == "ar"

    def test_heavy_english_mixed(self):
        text = "Hello World machine learning deep learning " * 8 + "مرحبا"
        assert detect_language(text) == "en"


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------

class TestCountWords:
    def test_simple_count(self):
        assert count_words("one two three") == 3

    def test_empty_string(self):
        assert count_words("") == 0

    def test_extra_spaces_ignored(self):
        assert count_words("  one   two  ") == 2

    def test_multiline(self):
        assert count_words("line one\nline two") == 4

    def test_single_word(self):
        assert count_words("hello") == 1


# ---------------------------------------------------------------------------
# File extractors
# ---------------------------------------------------------------------------

class TestExtractFromTxt:
    def test_utf8_content(self, txt_bytes):
        result = extract_from_txt(txt_bytes)
        assert "Machine learning" in result
        assert len(result) > 50

    def test_latin1_fallback(self):
        raw = "Café résumé naïve élève".encode("latin-1")
        result = extract_from_txt(raw)
        assert len(result) > 0

    def test_returns_string(self, txt_bytes):
        assert isinstance(extract_from_txt(txt_bytes), str)


class TestExtractFromDocx:
    def test_extracts_paragraphs(self, docx_bytes):
        result = extract_from_docx(docx_bytes)
        assert "Machine learning" in result
        assert "artificial intelligence" in result

    def test_returns_non_empty(self, docx_bytes):
        result = extract_from_docx(docx_bytes)
        assert len(result.split()) > 10


class TestExtractFromPptx:
    def test_extracts_slide_text(self, pptx_bytes):
        result = extract_from_pptx(pptx_bytes)
        assert "Machine Learning" in result

    def test_returns_non_empty(self, pptx_bytes):
        assert len(extract_from_pptx(pptx_bytes).split()) > 5


class TestExtractFromPdf:
    def test_extracts_text(self, pdf_bytes):
        from services.extractor import extract_from_pdf
        result = extract_from_pdf(pdf_bytes)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_expected_content(self, pdf_bytes):
        from services.extractor import extract_from_pdf
        result = extract_from_pdf(pdf_bytes)
        assert "Machine learning" in result or "machine learning" in result.lower()


# ---------------------------------------------------------------------------
# extract_text_from_file dispatcher
# ---------------------------------------------------------------------------

class TestExtractTextFromFile:
    def test_routes_txt(self, txt_bytes):
        result = extract_text_from_file("document.txt", txt_bytes)
        assert "Machine learning" in result

    def test_routes_docx(self, docx_bytes):
        result = extract_text_from_file("notes.docx", docx_bytes)
        assert "Machine learning" in result

    def test_routes_pptx(self, pptx_bytes):
        result = extract_text_from_file("slides.pptx", pptx_bytes)
        assert "Machine Learning" in result

    def test_case_insensitive_extension(self, txt_bytes):
        result = extract_text_from_file("DOCUMENT.TXT", txt_bytes)
        assert "Machine learning" in result

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_file("file.xyz", b"content")

    def test_unsupported_extension_lists_supported(self):
        with pytest.raises(ValueError) as exc_info:
            extract_text_from_file("file.odt", b"content")
        msg = str(exc_info.value).lower()
        assert "pdf" in msg or "docx" in msg

    def test_supported_extensions_set(self):
        for ext in (".pdf", ".docx", ".pptx", ".txt"):
            assert ext in SUPPORTED_EXTENSIONS
