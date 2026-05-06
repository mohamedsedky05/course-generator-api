from pydantic import BaseModel, Field
from typing import Optional, Literal


class GenerateRequest(BaseModel):
    text: Optional[str] = None
    video_url: Optional[str] = None
    num_quiz_questions: int = Field(default=10, ge=5, le=20)
    output_language: Literal["auto", "ar", "en"] = "auto"
