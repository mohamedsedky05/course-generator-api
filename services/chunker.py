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


def merge_lesson_chunks(chunk_results: list) -> dict:
    if len(chunk_results) == 1:
        return chunk_results[0]

    base = chunk_results[0]
    all_quiz = list(base.get("quiz", []))
    all_content = [base.get("content", "")]
    all_objectives = list(base.get("objectives", []))
    all_key_points = list(base.get("key_points", []))

    for result in chunk_results[1:]:
        all_quiz.extend(result.get("quiz", []))
        if result.get("content"):
            all_content.append(result["content"])
        for objective in result.get("objectives", []):
            if objective not in all_objectives:
                all_objectives.append(objective)
        for point in result.get("key_points", []):
            if point not in all_key_points:
                all_key_points.append(point)

    # Re-number quiz questions after combining chunk results.
    for i, q in enumerate(all_quiz, 1):
        q["question_number"] = i

    base["content"] = "\n\n".join(all_content)
    base["objectives"] = all_objectives[:10]
    base["key_points"] = all_key_points[:10]
    base["quiz"] = all_quiz

    return base


# Backward-compatible name during the course-to-lesson migration.
merge_course_chunks = merge_lesson_chunks
