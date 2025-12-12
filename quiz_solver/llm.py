# quiz_solver/llm.py
"""
Safe LLM planner wrapper with deterministic fallback.

Exports:
  - plan_python_solution(page_text: str) -> dict with keys:
      - "python_code": str  (python code that sets variable `answer`)
      - "notes": str
"""

import os
import re
import json
import logging
from typing import Dict, Optional

LOG = logging.getLogger("quiz_solver.llm")
if not LOG.handlers:
    # basic configuration if the outer app didn't set logging
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    LOG.addHandler(h)
    LOG.setLevel(logging.INFO)

# Regex patterns
NUMBERS_RE = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", re.IGNORECASE)
CSV_URL_RE = re.compile(r"(https?://[^\s'\"<>]+\.csv(?:\?[^\s'\"<>]*)?)", re.IGNORECASE)
PDF_URL_RE = re.compile(r"(https?://[^\s'\"<>]+\.pdf(?:\?[^\s'\"<>]*)?)", re.IGNORECASE)
POST_JSON_TEMPLATE_RE = re.compile(
    r"POST this JSON to\s*(https?://[^\s'\"<>]+)\s*({[\s\S]*?})",
    re.IGNORECASE
)

# LLM config (used only when OPENAI_API_KEY is set)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Try to import modern OpenAI types (safe fallback if not present)
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

try:
    # new SDK exposes exception types under openai.error or openai.types depending on version
    from openai.types import RateLimitError, APIError  # type: ignore
except Exception:
    try:
        from openai.error import RateLimitError, OpenAIError as APIError  # type: ignore
    except Exception:
        RateLimitError = Exception  # fallback
        APIError = Exception

# ----------------- Helper code emitters ----------------- #
def _emit_csv_code(csv_url: str, preferred_column: str = "value") -> str:
    """
    Return python code (string) that downloads a CSV and computes a sensible sum.
    The execution environment is expected to provide: pd, requests.
    """
    return f"""
# Download CSV and compute sum (execution env must provide pd and requests)
import io
text = requests.get({json.dumps(csv_url)}).text
df = pd.read_csv(io.StringIO(text))
col = {json.dumps(preferred_column)} if {json.dumps(preferred_column)} in df.columns else None
if col is None:
    # pick first numeric column
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        answer = float(df[numeric_cols].sum().sum())
    else:
        # attempt coercion fallback
        numeric_cols = df.columns.tolist()
        total = 0.0
        found = False
        for c in numeric_cols:
            try:
                s = pd.to_numeric(df[c], errors='coerce').dropna().sum()
                total += float(s)
                found = True
            except Exception:
                continue
        answer = float(total) if found else None
else:
    try:
        answer = float(pd.to_numeric(df[col], errors='coerce').sum())
    except Exception:
        answer = None
"""

def _emit_text_sum_code(sample_text: str) -> str:
    safe = json.dumps(sample_text[:2000])
    return f"""
# Sum numbers extracted from page text
import re
_text = {safe}
_nums = re.findall(r"{NUMBERS_RE.pattern}", _text)
_vals = []
for n in _nums:
    try:
        _vals.append(float(n))
    except:
        pass
answer = float(sum(_vals)) if _vals else None
"""

# ----------------- Deterministic planner ----------------- #
def _deterministic_plan(page_text: Optional[str]) -> Dict:
    """
    Heuristic-only planner: looks for explicit templates, CSV links, lots of numbers, or PDF hints.
    Returns a dict: {"python_code": str, "notes": str}
    """
    page_text = page_text or ""
    if not page_text.strip():
        return {"python_code": "answer = None", "notes": "Empty page text"}

    # 1) Detect explicit "POST this JSON to <url>\n{ ... }" templates
    m_template = POST_JSON_TEMPLATE_RE.search(page_text)
    if m_template:
        submit_hint = m_template.group(1)
        json_block = m_template.group(2)
        tpl = None
        try:
            tpl = json.loads(json_block)
        except Exception:
            # best-effort: replace single quotes with double quotes then try
            try:
                tpl = json.loads(json_block.replace("'", '"'))
            except Exception:
                tpl = None

        # Decide a default answer (demo says "anything you want")
        default_answer = "anything you want"
        if isinstance(tpl, dict) and "answer" in tpl:
            sample = tpl.get("answer")
            if sample is not None and not (isinstance(sample, str) and "your" in sample.lower()):
                # If the template already has a non-placeholder sample, reuse it
                default_answer = sample

        # Build python code that sets answer to a sensible default
        python_code = "answer = " + json.dumps(default_answer) + "\n"
        notes = f"Detected JSON submit template pointing to {submit_hint}; setting answer to {default_answer!r}."
        return {"python_code": python_code, "notes": notes}

    # 2) CSV detection
    m_csv = CSV_URL_RE.search(page_text)
    if m_csv:
        csv_url = m_csv.group(1)
        return {
            "python_code": _emit_csv_code(csv_url),
            "notes": f"Found CSV link; will download and sum a numeric column from {csv_url}"
        }

    # 3) If many numbers appear in page text, sum them
    nums = NUMBERS_RE.findall(page_text)
    if len(nums) >= 3:
        return {
            "python_code": _emit_text_sum_code(page_text),
            "notes": "No CSV found — summing numeric values detected in page text."
        }

    # 4) PDF hint: instruct outer solver to download the PDF (we don't do PDF extraction here)
    m_pdf = PDF_URL_RE.search(page_text)
    if m_pdf:
        pdf_url = m_pdf.group(1)
        return {
            "python_code": "answer = None",
            "notes": f"Found PDF link ({pdf_url}) — outer solver should download and parse the PDF."
        }

    # Fallback
    return {"python_code": "answer = None", "notes": "No actionable content found."}

# ----------------- Optional OpenAI integration ----------------- #
def _call_openai_for_plan(page_text: str) -> Dict:
    """
    Try to call OpenAI to get a JSON plan. Raises on errors; caller should handle exceptions.
    Expected returned dict should include "python_code".
    """
    if OpenAIClient is None:
        LOG.info("openai package not available; skipping LLM planning")
        raise RuntimeError("openai-not-installed")

    if not OPENAI_API_KEY:
        LOG.info("OPENAI_API_KEY not set; skipping LLM planning")
        raise RuntimeError("no-openai-key")

    # build client
    try:
        client = OpenAIClient(api_key=OPENAI_API_KEY)
    except Exception as e:
        LOG.exception("Failed to construct OpenAI client: %s", e)
        raise

    system_prompt = (
        "You are a JSON-only assistant. Return EXACTLY one JSON object (no extra text) with keys:\n"
        "  - 'python_code' (string) : python code that sets a variable named 'answer'\n"
        "  - optionally 'notes' (string)\n"
        "The python code must be self-contained for computing the 'answer' using only standard libs "
        "and may assume the outer execution environment provides 'requests' and 'pd' if needed."
    )
    user_prompt = f"Page text (truncated 6000 chars):\n{(page_text or '')[:6000]}\n\nReturn a JSON object."

    try:
        # Modern SDK usage
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.0,
        )
    except Exception as e:
        # bubble up common API exceptions (RateLimit, APIError) to be handled by caller
        LOG.exception("OpenAI API call failed: %s", e)
        raise

    # Extract content robustly across SDK versions/shapes
    content = ""
    try:
        # Newer shape: resp.choices[0].message.content
        choice0 = resp.choices[0]
        # some SDKs: choice0.message is a dict-like; some SDKs: choice0.message.content
        try:
            content = choice0.message.get("content", "")  # dict-like
        except Exception:
            content = getattr(choice0.message, "content", "") if hasattr(choice0, "message") else ""
        if not content:
            # older shape: choice0.text
            content = getattr(choice0, "text", "") or ""
    except Exception:
        # fallback: string-ify response
        try:
            content = str(resp)
        except Exception:
            content = ""

    # find JSON object substring
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        LOG.warning("OpenAI returned unexpected content; falling back.")
        raise RuntimeError("invalid-llm-output")

    try:
        plan = json.loads(m.group(0))
    except Exception:
        LOG.exception("Failed to parse JSON from LLM response")
        raise RuntimeError("invalid-llm-json")

    if "python_code" in plan and isinstance(plan["python_code"], str):
        return {"python_code": plan["python_code"], "notes": plan.get("notes", "From LLM")}
    raise RuntimeError("llm-did-not-provide-python_code")

# ----------------- Public entrypoint ----------------- #
def plan_python_solution(page_text: Optional[str]) -> Dict:
    """
    Safe planner entrypoint.
    - If OPENAI_API_KEY is set, attempt LLM planning (best-effort).
    - On any LLM failure, fall back to deterministic heuristics.
    Returns a dict with keys: "python_code" and "notes".
    """
    page_text = page_text or ""
    # If no key, deterministic only
    if not OPENAI_API_KEY:
        LOG.debug("OPENAI_API_KEY empty — using deterministic planner")
        return _deterministic_plan(page_text)

    # Try LLM but handle failures gracefully
    try:
        plan = _call_openai_for_plan(page_text)
        if plan and isinstance(plan, dict) and plan.get("python_code"):
            LOG.info("LLM plan obtained")
            return plan
        LOG.info("LLM produced no python_code — falling back")
    except RateLimitError as e:
        LOG.warning("OpenAI RateLimitError: %s", e)
    except APIError as e:
        LOG.warning("OpenAI APIError: %s", e)
    except Exception as e:
        LOG.warning("LLM planning failed; falling back to deterministic planner. error=%s", str(e))

    return _deterministic_plan(page_text)


# When run as script, provide a tiny self-test (no OpenAI call)
if __name__ == "__main__":
    sample = (
        "POST this JSON to https://tds-llm-analysis.s-anand.net/submit\n"
        '{\n  "email": "your email",\n  "secret": "your secret",\n  "url": "https://tds-llm-analysis.s-anand.net/demo",\n  "answer": "anything you want"\n}\n'
    )
    print(plan_python_solution(sample))
