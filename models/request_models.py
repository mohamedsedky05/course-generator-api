from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal
from fastapi import UploadFile


class GenerateRequest(BaseModel):
    text: Optional[str] = None
    video_url: Optional[str] = None
    num_lectures: int = Field(default=3, ge=2, le=8)
    num_quiz_questions: int = Field(default=10, ge=5, le=20)
    output_language: Literal["auto", "ar", "en"] = "auto"

    @model_validator(mode="after")
    def exactly_one_source(self):
        provided = sum([
            self.text is not None,
            self.video_url is not None,
        ])
        if provided == 0:
            raise ValueError("Provide exactly one of: text, file, or video_url")
        if provided > 1:
            raise ValueError("Only one of text, file, or video_url may be provided at a time")
        return self
