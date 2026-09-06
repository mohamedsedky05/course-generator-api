# AI Lesson Generator API

A production-ready FastAPI backend that converts text, documents, or YouTube videos into complete lessons using Anthropic Claude for lesson generation and Groq Whisper for audio transcription.

---

## Features

- **3 input types:** plain text, file upload (PDF/DOCX/PPTX/TXT), YouTube URL
- **LLM:** Anthropic Claude Sonnet for structured lesson generation
- **Transcription:** Claude-based processing for audio/video workflows
- **Arabic + English** support with auto language detection
- **Two-stage prompting** for structural analysis then content generation
- **Strict content-faithful pipeline** — LLM only reorganizes, never adds external knowledge

---

## Prerequisites

### 1. Get your Anthropic API key

1. Go to the [Anthropic Console](https://console.anthropic.com/)
2. Create or sign in to your account
3. Generate an API key
4. Copy it into your `.env` file as `ANTHROPIC_API_KEY`

### 2. Install FFmpeg (required by Whisper)

**Windows:**
```bash
winget install ffmpeg
# or via Chocolatey:
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg -y
```

Verify: `ffmpeg -version`

### 3. Python 3.10+

Make sure you have Python 3.10 or newer: `python --version`

---

## Setup

```bash
# Clone / navigate to the project
cd lesson_generator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

---

## Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/api/health

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/api/health
```

---

### Input Type 1 — Plain Text
```bash
curl -X POST http://localhost:8000/api/generate \
  -F "text=Machine learning is a subset of artificial intelligence that enables systems to learn from data. Supervised learning uses labeled datasets to train models. Common algorithms include linear regression, decision trees, and neural networks. Unsupervised learning finds patterns in unlabeled data using clustering and dimensionality reduction techniques." \
  -F "num_quiz_questions=5"
```

---

### Input Type 2 — File Upload (PDF/DOCX/PPTX/TXT)
```bash
curl -X POST http://localhost:8000/api/generate \
  -F "file=@/path/to/your/document.pdf" \
  -F "num_quiz_questions=10"
```

---

### Input Type 3 — YouTube URL
```bash
curl -X POST http://localhost:8000/api/generate \
  -F "video_url=https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  -F "num_quiz_questions=10" \
  -F "output_language=auto"
```

---

## Response Structure

```json
{
  "status": "success",
  "input_type": "youtube_video",
  "detected_language": "ar",
  "transcription": "...",
  "lesson": {
    "title": "...",
    "description": "...",
    "content": "...",
    "objectives": ["...", "..."],
    "key_points": ["...", "..."],
    "quiz": [
      {
        "question_number": 1,
        "type": "mcq",
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "correct_answer": 0,
        "explanation": "..."
      },
      {
        "question_number": 2,
        "type": "true_false",
        "question": "...",
        "correct_answer": true,
        "explanation": "..."
      }
    ]
  },
  "metadata": {
    "processing_time_seconds": 12.4,
    "word_count": 1500,
    "chunks_used": 1
  }
}
```

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CLAUDE_GENERATION_MODEL` | `claude-sonnet-4-6` | Claude model used for lesson generation |
| `CLAUDE_CLEANUP_MODEL` | `claude-haiku-4-5-20251001` | Claude model used for text cleanup |
| `CLAUDE_TRANSCRIPTION_MODEL` | `claude-sonnet-4-6` | Legacy setting; Claude does not receive audio |
| `GROQ_API_KEY` | *(required for audio fallback)* | API key for Whisper-compatible audio transcription |
| `GROQ_TRANSCRIPTION_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model used after video audio is downloaded |
| `YOUTUBE_COOKIES_B64` | *(optional)* | Base64-encoded Netscape `cookies.txt` used by server-side `yt-dlp` when YouTube blocks anonymous downloads |
| `MAX_TEXT_LENGTH` | `50000` | Max characters of text to send to Claude |
| `TEMP_AUDIO_DIR` | `./temp_audio` | Temporary directory for downloaded audio files |

> Recommended default models: `claude-sonnet-4-20250514` for generation and `claude-3-5-haiku-20241022` for cleanup.

### Production YouTube cookies

DigitalOcean App Platform has no browser profile, so `yt-dlp` cannot use
`--cookies-from-browser` there. If YouTube blocks anonymous downloads, export a
Netscape-format `cookies.txt` from a dedicated YouTube account and base64-encode
it locally:

```bash
base64 -w 0 cookies.txt > cookies.txt.b64
```

On Windows PowerShell:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("cookies.txt")) | Set-Content -NoNewline cookies.txt.b64
```

Set the contents of `cookies.txt.b64` as the encrypted App Platform variable
`YOUTUBE_COOKIES_B64`. The application materializes it as a temporary cookie
file only inside the container. Never commit `cookies.txt`, the base64 value,
or either value to chat. If the encoded value is too large for App Platform
environment-variable limits, store the cookie file in a private object store
and download it at startup instead.

Anthropic Claude does not currently accept audio content in the Messages API.
The application therefore uses Groq Whisper for audio transcription and Claude
for cleanup and lesson generation. Set `GROQ_API_KEY` for Bunny, Vimeo, and
other videos that do not provide a usable transcript.

---

## Error Codes

| Code | Meaning |
|---|---|
| `NO_INPUT` | No input provided |
| `MULTIPLE_INPUTS` | More than one input source provided |
| `UNSUPPORTED_FILE_TYPE` | File type not in PDF/DOCX/PPTX/TXT |
| `FILE_EXTRACTION_ERROR` | Could not read/parse the file |
| `VIDEO_UNAVAILABLE` | YouTube video is private, deleted, or invalid |
| `TRANSCRIPTION_FAILED` | Whisper transcription error |
| `TEXT_TOO_SHORT` | Extracted text under 50 words |
| `LLM_QUOTA_EXCEEDED` | Claude API quota hit |
| `LLM_ERROR` | Claude API or JSON parsing failure |
| `INTERNAL_ERROR` | Unexpected server error |
