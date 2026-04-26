# AI Course Generator API

A production-ready FastAPI backend that converts text, documents, or YouTube videos into structured educational courses using Google Gemini 2.0 Flash (free tier) and OpenAI Whisper (local, free).

---

## Features

- **3 input types:** plain text, file upload (PDF/DOCX/PPTX/TXT), YouTube URL
- **Free LLM:** Google Gemini 2.0 Flash with 1M token context
- **Free transcription:** OpenAI Whisper (runs locally, no API key needed)
- **Arabic + English** support with auto language detection
- **Two-stage prompting** for structural analysis then content generation
- **Strict content-faithful pipeline** — LLM only reorganizes, never adds external knowledge

---

## Prerequisites

### 1. Get a Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key — you get **free** access with generous limits (no credit card needed)

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
cd course_generator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
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
  -F "num_lectures=2" \
  -F "num_quiz_questions=5"
```

---

### Input Type 2 — File Upload (PDF/DOCX/PPTX/TXT)
```bash
curl -X POST http://localhost:8000/api/generate \
  -F "file=@/path/to/your/document.pdf" \
  -F "num_lectures=3" \
  -F "num_quiz_questions=10"
```

---

### Input Type 3 — YouTube URL
```bash
curl -X POST http://localhost:8000/api/generate \
  -F "video_url=https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  -F "num_lectures=4" \
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
  "course": {
    "title": "...",
    "description": "...",
    "summary": "...",
    "subject": "...",
    "difficulty": "Intermediate",
    "key_topics": ["...", "..."],
    "lectures": [
      {
        "lecture_number": 1,
        "title": "...",
        "content": "...",
        "objectives": ["...", "...", "..."]
      }
    ],
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
| `GEMINI_API_KEY` | *(required)* | Your Google AI Studio API key |
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `MAX_TEXT_LENGTH` | `50000` | Max characters of text to send to Gemini |
| `TEMP_AUDIO_DIR` | `./temp_audio` | Temporary directory for downloaded audio files |

> **Whisper model sizes:** `tiny` (fastest, least accurate) → `large` (slowest, most accurate). `base` is a good balance for most use cases.

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
| `LLM_QUOTA_EXCEEDED` | Gemini free tier quota hit |
| `LLM_ERROR` | Gemini API or JSON parsing failure |
| `INTERNAL_ERROR` | Unexpected server error |
