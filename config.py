import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    claude_api_key: str = ""
    claude_generation_model: str = "claude-sonnet-4-6"
    claude_cleanup_model: str = "claude-haiku-4-5-20251001"
    claude_transcription_model: str = "claude-sonnet-4-6"
    groq_api_key: str = ""
    groq_transcription_model: str = "whisper-large-v3-turbo"
    youtube_cookies_b64: str = ""
    youtube_proxy: str = ""
    vimeo_cookies_b64: str = ""
    basic_auth_username: str = ""
    basic_auth_password: str = ""
    max_text_length: int = 50000
    temp_audio_dir: str = "./temp_audio"
    allowed_origins: str = "*"

    @property
    def effective_anthropic_api_key(self) -> str:
        return (
            self.anthropic_api_key
            or self.claude_api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("CLAUDE_API_KEY")
            or ""
        )

    @property
    def effective_groq_api_key(self) -> str:
        return self.groq_api_key or os.getenv("GROQ_API_KEY") or ""

    @property
    def allowed_origins_list(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

Path(settings.temp_audio_dir).mkdir(parents=True, exist_ok=True)
