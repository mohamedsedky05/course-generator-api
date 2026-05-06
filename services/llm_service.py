import asyncio
import json
import logging
import re
import time

import google.genai as genai
from google.genai.errors import ClientError

from config import settings

logger = logging.getLogger("llm_service")

MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT = 120.0        # seconds per call
_RETRY_DELAYS = [1, 2, 4]    # exponential backoff for 503 / timeout

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are an expert educational content organizer.

Convert the following text into a structured educational output.

STRICT CONTENT RULES — THIS IS THE MOST IMPORTANT PART:
- Use ONLY the information explicitly present in the provided text
- Do NOT add any external knowledge, examples, or analogies
- Every quiz question must be answerable ONLY from the provided text
- Respond ONLY in valid JSON with no markdown formatting

Generate exactly {q} quiz questions (70% MCQ with 4 options, 30% True/False).
Output language must match the input content language.

Required JSON schema (respond with NOTHING else):
{{
  "title": "A concise title derived from the text",
  "description": "2-3 sentence description using only what is in the text",
  "quiz": [
    {{
      "question_number": 1,
      "type": "mcq",
      "question": "Question text from the content",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": 0,
      "explanation": "Citation from the text explaining the answer"
    }},
    {{
      "question_number": 2,
      "type": "true_false",
      "question": "Statement from the content",
      "correct_answer": true,
      "explanation": "Citation from the text explaining why"
    }}
  ]
}}

TEXT:
{text}"""

RETRY_SUFFIX = "\n\nCRITICAL: Your previous response was not valid JSON. Respond ONLY with valid JSON. No markdown. No explanation. No code blocks. Start your response with {{ and end with }}."

TERM_FIX_PROMPT = """You are a technical text corrector for Arabic educational content.

Your task: fix Arabic phonetic transliterations of English technical terms back to their correct English spelling.

STRICT RULES:
1. Keep all genuine Arabic words and sentences exactly as-is.
2. Replace Arabic phonetic spellings of English terms with the correct English term inline.
3. NEVER modify: numbers, IP addresses (e.g. 192.168.1.1), subnet masks, CIDR notation (e.g. /24), port numbers, MAC addresses, code snippets, URLs, shell commands, or any sequence that is already in English.
4. Preserve the original sentence structure and paragraph breaks.
5. Return ONLY the corrected text — no explanations, no markdown.

Common patterns to recognise and fix:
راوتر / روتر → router
سويتش / سويش → switch
سيرفر / سيرفير → server
كلاود → cloud
نيتورك / نتورك → network
فايروول / فايروال → firewall
بروتوكول → protocol
آي بي / اي بي → IP
بريفيكس / بريفكس → prefix
سبنت ماسك / سابنت ماسك → subnet mask
فيكسد لينث → fixed length
ديفولت جيتواي / ديفولت قيتواي → default gateway
داتا بيس / داتابيس → database
سيكيوريتي → security
إنكريبشن / انكريبشن → encryption
أوثنتيكيشن / اوثنتيكيشن → authentication
يوزر نيم → username
باسورد → password
كونفيجيوريشن / كونفيجريشن → configuration
إنترفيس / انترفيس → interface
لوبباك / لوب باك → loopback
بنج → ping
تريسروت → traceroute
دي إن إس / DNS → DNS
دي إتش سي بي → DHCP
إن إيه تي / ناك → NAT
في بي إن → VPN
أو إس بي إف → OSPF
بي جي بي → BGP
في إل إيه إن → VLAN
تي سي بي → TCP
يو دي بي → UDP
إتش تي تي بي / اتش تي تي بي → HTTP
إس إس إل → SSL
تي إل إس → TLS
ويب سيرفر → web server
لود بالانسر → load balancer
كاش / كاشينج → cache / caching
ميكروسيرفيسز → microservices
كونتينر / كونتينرز → container / containers
دوكر → Docker
كيوبيرنتيز → Kubernetes
سي بي يو → CPU
رام / RAM → RAM
ستوريج → storage
باند ويدث → bandwidth
لاتنسي → latency
فايل سيستم → file system
أوبن سورس → open source
ريبوزيتوري → repository
بول ريكويست → pull request
كوميت → commit
برانش → branch

TEXT:
{text}"""


# ---------------------------------------------------------------------------
# Core Gemini call with retry + timeout
# ---------------------------------------------------------------------------

def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


async def _call_gemini(prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, len(_RETRY_DELAYS) + 2):  # up to 4 attempts
        try:
            client = _get_client()
            response = await asyncio.wait_for(
                client.aio.models.generate_content(model=MODEL, contents=prompt),
                timeout=GEMINI_TIMEOUT,
            )
            return response.text
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning(f"[Gemini] Attempt {attempt} timed out after {GEMINI_TIMEOUT}s")
        except ClientError as e:
            last_exc = e
            # ClientError stores the HTTP status in .code (google-genai SDK)
            status = getattr(e, 'code', None) or getattr(e, 'status_code', 0)
            logger.warning(f"[Gemini error] {type(e).__name__} status={status}: {e}")
            if status != 503:
                raise  # non-retryable (quota, auth, etc.)
            logger.warning(f"[Gemini] 503 on attempt {attempt}, will retry")
        except Exception as e:
            logger.error(f"[Gemini error] {type(e).__name__}: {e}")
            raise

        if attempt <= len(_RETRY_DELAYS):
            delay = _RETRY_DELAYS[attempt - 1]
            logger.info(f"[Gemini] Retrying in {delay}s (attempt {attempt + 1}/{len(_RETRY_DELAYS) + 1})...")
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


def _parse_json_safe(raw: str) -> dict:
    cleaned = _strip_markdown_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start:end + 1])
        raise


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

async def clean_transcription(text: str) -> str:
    """Fix Arabic transliterations of English technical terms."""
    t0 = time.time()
    prompt = TERM_FIX_PROMPT.format(text=text)
    try:
        result = (await _call_gemini(prompt)).strip()
        logger.info(f"[llm] clean_transcription done in {time.time()-t0:.2f}s")
        return result
    except Exception:
        logger.warning("[llm] clean_transcription failed, using original text")
        return text


async def generate_content(text: str, num_quiz_questions: int) -> dict:
    """
    Single-stage pipeline: clean the text then generate title + description + quiz.
    Returns a dict with keys: title, description, quiz.
    """
    t_total = time.time()

    # Clean up transliterations (no-op for pure English; safe fallback on error)
    cleaned_text = await clean_transcription(text)

    # Generate title + description + quiz in one call
    logger.info(f"[llm] Generating content ({num_quiz_questions} quiz questions)")
    prompt = GENERATION_PROMPT.format(q=num_quiz_questions, text=cleaned_text)
    raw = await _call_gemini(prompt)
    try:
        result = _parse_json_safe(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[llm] Invalid JSON on first attempt, retrying with explicit instruction")
        raw2 = await _call_gemini(prompt + RETRY_SUFFIX)
        result = _parse_json_safe(raw2)

    logger.info(
        f"[llm] generate_content done in {time.time()-t_total:.2f}s | "
        f"quiz_items={len(result.get('quiz', []))}"
    )
    return result
