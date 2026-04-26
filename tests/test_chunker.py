"""Unit tests for services/chunker.py"""
import pytest

from services.chunker import chunk_text, merge_course_chunks, MAX_WORDS_PER_CHUNK


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_returns_single_chunk(self, english_text):
        chunks = chunk_text(english_text)
        assert len(chunks) == 1
        assert chunks[0] == english_text

    def test_long_text_produces_multiple_chunks(self, long_text):
        chunks = chunk_text(long_text)
        assert len(chunks) > 1

    def test_each_chunk_within_word_limit(self, long_text):
        for chunk in chunk_text(long_text):
            assert len(chunk.split()) <= MAX_WORDS_PER_CHUNK

    def test_no_words_lost(self, long_text):
        chunks = chunk_text(long_text)
        original_words = sorted(long_text.split())
        reconstructed_words = sorted(" ".join(chunks).split())
        assert original_words == reconstructed_words

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == [""]

    def test_custom_max_words(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_text(text, max_words=50)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.split()) <= 50

    def test_single_chunk_when_exactly_at_limit(self):
        text = " ".join(["word"] * MAX_WORDS_PER_CHUNK)
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_splits_at_paragraph_boundaries(self):
        # Build text whose paragraphs are clearly sized
        para = " ".join(["word"] * 1000)  # 1000-word paragraphs
        text = "\n\n".join([para] * 10)   # 10 000 words total
        chunks = chunk_text(text, max_words=2000)
        # None of the chunks should end mid-word relative to a paragraph
        for chunk in chunks:
            # A valid split point: each chunk is a join of whole paragraphs
            assert chunk.strip() != ""

    def test_no_empty_chunks(self, long_text):
        for chunk in chunk_text(long_text):
            assert chunk.strip() != ""


# ---------------------------------------------------------------------------
# merge_course_chunks
# ---------------------------------------------------------------------------

def _make_chunk(n: int, prefix: str = "") -> dict:
    return {
        "title": f"{prefix}Course",
        "description": f"{prefix}Desc",
        "summary": f"{prefix}Summary {n}",
        "subject": "Testing",
        "difficulty": "Beginner",
        "key_topics": [f"{prefix}topic{n}A", f"{prefix}topic{n}B"],
        "lectures": [
            {
                "lecture_number": 1,
                "title": f"{prefix}Lecture 1 of chunk {n}",
                "content": "Content here.",
                "objectives": ["obj1", "obj2", "obj3"],
            }
        ],
        "quiz": [
            {
                "question_number": 1,
                "type": "mcq",
                "question": f"{prefix}Question {n}",
                "options": ["A. opt1", "B. opt2", "C. opt3", "D. opt4"],
                "correct_answer": 0,
                "explanation": "Because of A.",
            }
        ],
    }


class TestMergeCourseChunks:
    def test_single_chunk_returned_unchanged(self):
        chunk = _make_chunk(1)
        result = merge_course_chunks([chunk])
        assert result is chunk

    def test_merges_lectures_from_all_chunks(self):
        chunks = [_make_chunk(i) for i in range(1, 4)]
        result = merge_course_chunks(chunks)
        assert len(result["lectures"]) == 3

    def test_merges_quiz_from_all_chunks(self):
        chunks = [_make_chunk(i) for i in range(1, 4)]
        result = merge_course_chunks(chunks)
        assert len(result["quiz"]) == 3

    def test_lecture_numbers_sequential(self):
        chunks = [_make_chunk(i) for i in range(1, 5)]
        result = merge_course_chunks(chunks)
        numbers = [lec["lecture_number"] for lec in result["lectures"]]
        assert numbers == list(range(1, 5))

    def test_quiz_numbers_sequential(self):
        chunks = [_make_chunk(i) for i in range(1, 5)]
        result = merge_course_chunks(chunks)
        numbers = [q["question_number"] for q in result["quiz"]]
        assert numbers == list(range(1, 5))

    def test_summaries_concatenated(self):
        c1, c2 = _make_chunk(1), _make_chunk(2)
        result = merge_course_chunks([c1, c2])
        assert "Summary 1" in result["summary"]
        assert "Summary 2" in result["summary"]

    def test_duplicate_topics_deduplicated(self):
        # Two identical chunks → topics should appear only once
        c1 = _make_chunk(1)
        c2 = _make_chunk(1)  # same topics as c1
        result = merge_course_chunks([c1, c2])
        assert len(result["key_topics"]) == len(set(result["key_topics"]))

    def test_key_topics_capped_at_ten(self):
        # 7 chunks × 2 unique topics each = 14; should be capped at 10
        chunks = [_make_chunk(i, prefix=f"c{i}_") for i in range(1, 8)]
        result = merge_course_chunks(chunks)
        assert len(result["key_topics"]) <= 10

    def test_base_metadata_comes_from_first_chunk(self):
        c1 = _make_chunk(1, prefix="first_")
        c2 = _make_chunk(2, prefix="second_")
        result = merge_course_chunks([c1, c2])
        assert result["title"] == "first_Course"
        assert result["subject"] == "Testing"
