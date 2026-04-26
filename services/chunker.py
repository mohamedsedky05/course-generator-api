from typing import List

MAX_WORDS_PER_CHUNK = 6000


def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_parts: List[str] = []
    current_count = 0

    for para in paragraphs:
        para_word_list = para.split()
        para_word_count = len(para_word_list)

        if para_word_count > max_words:
            # Flush whatever we've accumulated first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_count = 0
            # Split the oversized paragraph into fixed-size word slices
            for i in range(0, para_word_count, max_words):
                chunks.append(" ".join(para_word_list[i: i + max_words]))
        elif current_count + para_word_count > max_words and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_count = para_word_count
        else:
            current_parts.append(para)
            current_count += para_word_count

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def merge_course_chunks(chunk_results: list) -> dict:
    if len(chunk_results) == 1:
        return chunk_results[0]

    base = chunk_results[0]
    all_lectures = list(base.get("lectures", []))
    all_quiz = list(base.get("quiz", []))
    all_topics = list(base.get("key_topics", []))
    summaries = [base.get("summary", "")]

    for result in chunk_results[1:]:
        all_lectures.extend(result.get("lectures", []))
        all_quiz.extend(result.get("quiz", []))
        topics = result.get("key_topics", [])
        for t in topics:
            if t not in all_topics:
                all_topics.append(t)
        summaries.append(result.get("summary", ""))

    # Re-number lectures and quiz questions
    for i, lec in enumerate(all_lectures, 1):
        lec["lecture_number"] = i
    for i, q in enumerate(all_quiz, 1):
        q["question_number"] = i

    base["lectures"] = all_lectures
    base["quiz"] = all_quiz
    base["key_topics"] = all_topics[:10]
    base["summary"] = " ".join(s for s in summaries if s)

    return base
