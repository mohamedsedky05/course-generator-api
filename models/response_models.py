from pydantic import BaseModel
from typing import Optional, Union, List, Literal


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


class Lecture(BaseModel):
    lecture_number: int
    title: str
    content: str
    objectives: List[str]


class Course(BaseModel):
    title: str
    description: str
    summary: str
    subject: str
    difficulty: str
    key_topics: List[str]
    lectures: List[Lecture]
    quiz: List[Union[MCQQuestion, TrueFalseQuestion]]


class Metadata(BaseModel):
    processing_time_seconds: float
    word_count: int
    chunks_used: int


class GenerateResponse(BaseModel):
    status: Literal["success"]
    input_type: str
    detected_language: str
    transcription: Optional[str] = None
    course: Course
    metadata: Metadata


class ErrorResponse(BaseModel):
    status: Literal["error"]
    error_code: str
    message: str
