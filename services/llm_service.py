import asyncio
import json
import logging
import re
import time

from anthropic import AsyncAnthropic

from config import settings

logger = logging.getLogger("llm_service")

GENERATION_MODEL = settings.claude_generation_model
CLEANUP_MODEL = settings.claude_cleanup_model
CLAUDE_TIMEOUT = 120.0
_RETRY_DELAYS = [1, 2, 4]

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    api_key = settings.effective_anthropic_api_key
    if _client is None:
        _client = AsyncAnthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATION_PROMPT_PREFIX = """You are an expert educational content organizer.

Convert the following text into a structured educational output.

STRICT CONTENT RULES — THIS IS THE MOST IMPORTANT PART:
- Use ONLY the information explicitly present in the provided text
- Do NOT add any external knowledge, examples, or analogies
- Every quiz question must be answerable ONLY from the provided text
- Respond ONLY in valid JSON with no markdown formatting

Output language must match the input content language.

Required JSON schema (respond with NOTHING else):
{{
  "title": "A concise title derived from the text",
  "description": "2-3 sentence description using only what is in the text",
  "quiz_title": "A short engaging title for the quiz section, derived from the content",
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

"""

GENERATION_PROMPT = GENERATION_PROMPT_PREFIX + "Generate exactly {q} quiz questions (70% MCQ with 4 options, 30% True/False).\n\nTEXT:\n{text}"

RETRY_SUFFIX = "\n\nCRITICAL: Your previous response was not valid JSON. Respond ONLY with valid JSON. No markdown. No explanation. No code blocks. Start your response with {{ and end with }}."

TERM_FIX_PROMPT_PREFIX = """You are a technical text corrector for Arabic educational content.

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
"""

TERM_FIX_PROMPT = TERM_FIX_PROMPT_PREFIX + "{text}"


# ---------------------------------------------------------------------------
# Core Claude call with retry + timeout
# ---------------------------------------------------------------------------

def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


async def _call_claude(prompt: str, model: str, prompt_prefix: str | None = None) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, len(_RETRY_DELAYS) + 2):
        try:
            client = _get_client()
            if prompt_prefix is None:
                user_content = prompt
            else:
                user_content = [
                    {
                        "type": "text",
                        "text": prompt_prefix,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": prompt},
                ]
            response = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system="You are a precise, structured assistant.",
                    messages=[{"role": "user", "content": user_content}],
                ),
                timeout=CLAUDE_TIMEOUT,
            )
            content = getattr(response, "content", response)
            if isinstance(content, str):
                return content
            if not isinstance(content, (list, tuple)) and hasattr(response, "text"):
                return response.text
            return "".join(block.text for block in content if getattr(block, "type", None) == "text")
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning(f"[Claude] Attempt {attempt} timed out after {CLAUDE_TIMEOUT}s")
        except Exception as e:
            last_exc = e
            logger.warning(f"[Claude error] {type(e).__name__}: {e}")
            status = getattr(e, "status_code", None)
            if status not in (500, 503):
                raise

        if attempt <= len(_RETRY_DELAYS):
            delay = _RETRY_DELAYS[attempt - 1]
            logger.info(f"[Claude] Retrying in {delay}s (attempt {attempt + 1}/{len(_RETRY_DELAYS) + 1})...")
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

# Backward-compatible alias for older tests and call sites.
_call_gemini = _call_claude


async def clean_transcription(text: str) -> str:
    """Fix Arabic transliterations of English technical terms."""
    t0 = time.time()
    prompt = text
    try:
        result = (await _call_claude(prompt, CLEANUP_MODEL, TERM_FIX_PROMPT_PREFIX)).strip()
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
    prompt = (
        f"Generate exactly {num_quiz_questions} quiz questions "
        "(70% MCQ with 4 options, 30% True/False).\n\n"
        f"TEXT:\n{cleaned_text}"
    )
    raw = await _call_claude(prompt, GENERATION_MODEL, GENERATION_PROMPT_PREFIX)
    try:
        result = _parse_json_safe(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[llm] Invalid JSON on first attempt, retrying with explicit instruction")
        raw2 = await _call_claude(
            prompt + RETRY_SUFFIX,
            GENERATION_MODEL,
            GENERATION_PROMPT_PREFIX,
        )
        result = _parse_json_safe(raw2)

    logger.info(
        f"[llm] generate_content done in {time.time()-t_total:.2f}s | "
        f"quiz_items={len(result.get('quiz', []))}"
    )
    return result
