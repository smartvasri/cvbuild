"""
ai_pipeline.py
Calls the Gemini 2.0 Flash REST API and returns structured ATS analysis.

Requires environment variable: GEMINI_API_KEY
Get a key at: https://aistudio.google.com/app/apikey
"""

import os
import json
import re
import time
import urllib.request
import urllib.error

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior HR specialist and ATS (Applicant Tracking System) expert. "
    "Always respond with valid JSON only — no markdown fences, no extra commentary."
)

USER_PROMPT = """Analyze the resume below and produce a structured ATS assessment.

## Scoring Rubric (Total: 100 points)
- Keyword optimization & industry relevance : 30 pts
- Resume structure, formatting, and clarity : 20 pts
- Skills breadth and depth                  : 20 pts
- Work experience impact and quantification : 20 pts
- Education, certifications, achievements   : 10 pts

## Resume Text:
{resume_text}

## Instructions:
1. ATS Score  — integer 0–100 based on the rubric above.
2. Skills     — every technical tool, programming language, framework, methodology,
                and notable soft skill found in the resume.
3. Suggestions — 5 to 7 specific, actionable improvements to raise the ATS score.
4. Job Roles  — top 5 job titles this candidate is best suited for right now.

Respond ONLY with this exact JSON (no markdown, no extra text):

{{
  "ats_score": <integer 0-100>,
  "skills": ["skill1", "skill2", "..."],
  "suggestions": [
    "Suggestion 1: ...",
    "Suggestion 2: ...",
    "Suggestion 3: ...",
    "Suggestion 4: ...",
    "Suggestion 5: ..."
  ],
  "job_roles": ["Role 1", "Role 2", "Role 3", "Role 4", "Role 5"]
}}"""

# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def analyze_resume(resume_text: str) -> dict:
    """
    Send resume text to Gemini 2.0 Flash and return structured ATS data.

    Returns:
        {
            "ats_score":   int,
            "skills":      list[str],
            "suggestions": list[str],
            "job_roles":   list[str]
        }

    Raises:
        EnvironmentError  — API key not set
        RuntimeError      — quota / rate-limit / HTTP error (user-friendly message)
        ValueError        — could not parse JSON from response
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    payload = json.dumps({
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": USER_PROMPT.format(resume_text=resume_text[:8000])}]
            }
        ],
        "generationConfig": {
            "temperature":     0.3,
            "topP":            0.9,
            "maxOutputTokens": 2048,
        }
    }).encode("utf-8")

    url     = f"{GEMINI_API_URL}?key={api_key}"
    headers = {"Content-Type": "application/json"}

    max_retries = 3

    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body     = json.loads(resp.read().decode("utf-8"))
                raw_text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
            break  # success

        except urllib.error.HTTPError as e:
            status   = e.code
            err_body = e.read().decode("utf-8", errors="replace")

            if status == 400:
                raise RuntimeError(
                    f"Bad request to Gemini API. Check your prompt or API key. Detail: {err_body[:200]}"
                ) from e

            if status == 401 or status == 403:
                raise RuntimeError(
                    "Gemini API key is invalid or does not have permission. "
                    "Check GEMINI_API_KEY in Railway environment variables."
                ) from e

            if status == 404:
                raise RuntimeError(
                    f"Model '{GEMINI_MODEL}' not found. "
                    "It may not be available for your API key. Try enabling billing on Google AI Studio."
                ) from e

            if status == 429:
                # Check if it is daily quota (no point retrying) or per-minute (retry)
                if "PerDay" in err_body or attempt >= max_retries - 1:
                    raise RuntimeError(
                        "Gemini API quota exceeded. "
                        "Please wait until your quota resets or upgrade your plan."
                    ) from e
                wait = 2 ** attempt * 5  # 5s, 10s
                time.sleep(wait)
                continue

            if status >= 500 and attempt < max_retries - 1:
                time.sleep(2 ** attempt * 4)  # 4s, 8s  — transient server error
                continue

            raise RuntimeError(
                f"Gemini API returned HTTP {status}: {err_body[:300]}"
            ) from e

        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise RuntimeError(f"Could not reach Gemini API: {e.reason}") from e

    # ── Parse the JSON out of the response ───────────────────────
    raw_text = re.sub(r"```json\s*", "", raw_text)
    raw_text = re.sub(r"```\s*",     "", raw_text)

    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match:
        raise ValueError(
            f"Gemini returned a non-JSON response:\n{raw_text[:500]}"
        )

    data = json.loads(json_match.group())

    return {
        "ats_score":   max(0, min(100, int(data.get("ats_score", 0)))),
        "skills":      [str(s) for s in data.get("skills",      [])][:30],
        "suggestions": [str(s) for s in data.get("suggestions", [])][:10],
        "job_roles":   [str(r) for r in data.get("job_roles",   [])][:5],
    }
