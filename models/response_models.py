from pydantic import BaseModel
from typing import List, Literal, Union


class MCQQuestion(BaseModel):
    question_number: int
    type: Literal["mcq"]
    question: str
    options: List[str]
    correct_answer: int
    explanation: str


class TrueFalseQuestion(BaseModel):
    question_number: int
    type: Literal["true_false"]
    question: str
    correct_answer: bool
    explanation: str


class LessonResult(BaseModel):
    title: str
    description: str
    content: str
    objectives: List[str]
    key_points: List[str]
    quiz_title: str
    quiz: List[Union[MCQQuestion, TrueFalseQuestion]]


class Metadata(BaseModel):
    processing_time_seconds: float
    word_count: int


class GenerateResponse(BaseModel):
    status: Literal["success"]
    input_type: str
    detected_language: str
    transcript: str          # always populated — raw extracted text
    lesson: LessonResult
    course: LessonResult | None = None
    metadata: Metadata


class ErrorResponse(BaseModel):
    status: Literal["error"]
    error_code: str
    message: str
    ar_message: str
