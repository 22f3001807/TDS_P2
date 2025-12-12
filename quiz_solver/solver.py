# quiz_solver/solver.py
"""
Quiz session runner.

Provides:
  - run_quiz_session_sync(start_url: str, limit: int=170) -> dict

Depends on:
  - quiz_solver.config: STUDENT_EMAIL, QUIZ_SECRET
  - quiz_solver.browser: async get_page_text(url) -> str (must render JS)
  - quiz_solver.llm: plan_python_solution(page_text) -> {"python_code":..., "notes":...}
"""

import asyncio
import time
import json
import logging
import traceback
import re
from typing import Optional

import httpx

from .config import STUDENT_EMAIL, QUIZ_SECRET, TOTAL_QUIZ_WINDOW_SECONDS
from .browser import get_page_text
from .llm import plan_python_solution

LOG = logging.getLogger("quiz_solver.solver")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    LOG.addHandler(h)
    LOG.setLevel(logging.INFO)


def extract_submit_url_from_text(page_text: str) -> str:
    """Find a submit URL like https://.../submit in the page text. Raise if not found."""
    match = re.search(r"https?://[^\s\"'<>]*submit[^\s\"'<>]*", page_text)
    if not match:
        raise ValueError("Submit URL not found")
    return match.group(0)


def execute_generated_code(python_code: str) -> dict:
    """
    Execute python code produced by planner in a restricted environment.
    The executed code should set a variable named `answer`.
    Returns dict with ok, answer, error, traceback.
    """
    safe_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "sorted": sorted,
        "print": print,
        "__import__": __import__,
    }

    safe_globals = {"__builtins__": safe_builtins}
    # Provide common libs that LLM might assume
    try:
        import pandas as pd  # may be heavy but typical in this project
    except Exception:
        pd = None
    import requests
    from bs4 import BeautifulSoup

    safe_locals = {
        "pd": pd,
        "requests": requests,
        "BeautifulSoup": BeautifulSoup,
    }

    try:
        exec(python_code, safe_globals, safe_locals)
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "answer": None,
        }

    if "answer" not in safe_locals:
        return {
            "ok": False,
            "error": "LLM code did not set answer",
            "traceback": "",
            "answer": None,
        }

    answer = safe_locals["answer"]
    # ensure JSON-serializable or stringify
    try:
        json.dumps(answer)
    except Exception:
        answer = str(answer)

    return {"ok": True, "error": None, "traceback": "", "answer": answer}


async def solve_single_quiz(url: str) -> dict:
    """
    Visit the given url (using browser.get_page_text), ask the planner for a python snippet,
    run it and return execution result and diagnostics.
    """
    LOG.info("Handling quiz page: %s", url)
    page_text = await get_page_text(url)
    plan = plan_python_solution(page_text)
    python_code = plan.get("python_code", "") or ""
    notes = plan.get("notes", "") or ""

    exec_result = execute_generated_code(python_code)

    return {
        "url": url,
        "page_text_preview": (page_text or "")[:1000],
        "python_code": python_code,
        "llm_notes": notes,
        "exec_result": exec_result,
    }


async def submit_answer(submit_url: str, quiz_url: str, answer):
    """
    POST the answer to submit_url using STUDENT_EMAIL and QUIZ_SECRET.
    Returns the parsed JSON response or raises.
    """
    payload = {
        "email": STUDENT_EMAIL,
        "secret": QUIZ_SECRET,
        "url": quiz_url,
        "answer": answer,
    }
    LOG.info("Submitting to %s payload keys=%s", submit_url, list(payload.keys()))
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(submit_url, json=payload)
        resp.raise_for_status()
        # try parse JSON
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}


async def handle_quiz_session(start_url: str, limit: int = 170) -> dict:
    """
    Drive a quiz session asynchronously:
      - Solve page
      - Find submit URL
      - Submit, follow next URL if provided
      - Stop when no next URL or time limit exceeded
    """
    history = []
    start_time = time.time()
    current_url = start_url

    while current_url and (time.time() - start_time < limit):
        LOG.info("Rendering page: %s", current_url)
        one = await solve_single_quiz(current_url)
        page_text = await get_page_text(current_url)

        # try to extract submit URL
        try:
            submit_url = extract_submit_url_from_text(page_text)
        except Exception as e:
            one["extract_error"] = str(e)
            history.append(one)
            LOG.info("No submit url and no scrape instruction on %s: %s", current_url, str(e))
            break

        answer = one["exec_result"]["answer"]
        submission = None

        if one["exec_result"]["ok"]:
            try:
                submission = await submit_answer(submit_url, current_url, answer)
                LOG.info("Platform returned next url: %s", submission.get("url") if isinstance(submission, dict) else submission)
            except Exception as e:
                LOG.exception("Submission failed")
                submission = {"correct": False, "reason": str(e)}

        one["submit_url"] = submit_url
        one["submission"] = submission
        history.append(one)

        if not submission:
            break

        next_url = submission.get("url")
        if not next_url:
            break
        current_url = next_url

    return {"history": history, "duration_seconds": time.time() - start_time}


def run_quiz_session_sync(start_url: str, limit: int = 170):
    """
    Synchronous wrapper to run the async session.
    Handles environments where an event loop is already running (e.g. debug servers).
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    # If loop already running (common inside debug servers), apply nest_asyncio if available.
    if loop and loop.is_running():
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            LOG.info("Applied nest_asyncio to allow nested asyncio.run in running loop")
        except Exception:
            LOG.warning("nest_asyncio not available; attempting run via new event loop")

        # run the coroutine directly using loop.create_task and wait until done
        coro = handle_quiz_session(start_url, limit)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    # Otherwise safe to use asyncio.run
    return asyncio.run(handle_quiz_session(start_url, limit))
