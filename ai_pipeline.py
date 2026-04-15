"""
ai_pipeline.py
Calls the Gemini 2.0 Flash REST API and returns structured ATS analysis.
Validates that the uploaded document is actually a resume/CV before analyzing.

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

USER_PROMPT = """You will be given text extracted from a document. Your first job is to determine
whether this document is a resume or CV (a professional document listing a person's education,
work experience, and skills for job applications).

STEP 1 — Identify the document type.
Look for clear resume/CV signals:
- Candidate name, contact info (email, phone, LinkedIn)
- Work experience / employment history section
- Education section
- Skills section
- Professional summary or objective

If the document is clearly NOT a resume/CV (e.g. it is an invoice, contract, research paper,
article, letter, certificate, report, product manual, etc.), respond with ONLY this JSON:

{{
  "is_resume": false,
  "document_type": "<what type of document it actually is>",
  "reason": "<one sentence explaining why it is not a resume>"
}}

STEP 2 — If it IS a resume/CV, perform the full ATS analysis below.

## Scoring Rubric (Total: 100 points)
- Keyword optimization & industry relevance : 30 pts
- Resume structure, formatting, and clarity : 20 pts
- Skills breadth and depth                  : 20 pts
- Work experience impact and quantification : 20 pts
- Education, certifications, achievements   : 10 pts

## Instructions:
1. ATS Score  — integer 0–100 based on the rubric above.
2. Skills     — every technical tool, programming language, framework, methodology,
                and notable soft skill found in the resume.
3. Suggestions — 5 to 7 specific, actionable improvements to raise the ATS score.
4. Job Roles  — top 5 job titles this candidate is best suited for right now.

Respond ONLY with this exact JSON (no markdown, no extra text):

{{
  "is_resume": true,
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
}}

## Document text:
{resume_text}"""


# ─────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────

def _call_gemini(api_key: str, user_text: str) -> str:
    """
    Send a single request to the Gemini REST API.
    Returns the raw text response.
    Retries up to 3 times on transient errors.
    """
    payload = json.dumps({
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}]
            }
        ],
        "generationConfig": {
            "temperature":     0.2,
            "topP":            0.9,
            "maxOutputTokens": 2048,
        }
    }).encode("utf-8")

    url     = f"{GEMINI_API_URL}?key={api_key}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(3):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["candidates"][0]["content"]["parts"][0]["text"].strip()

        except urllib.error.HTTPError as e:
            status   = e.code
            err_body = e.read().decode("utf-8", errors="replace")

            if status in (401, 403):
                raise RuntimeError(
                    "Gemini API key is invalid or unauthorised. "
                    "Check GEMINI_API_KEY in Railway environment variables."
                ) from e
            if status == 404:
                raise RuntimeError(
                    f"Model '{GEMINI_MODEL}' not available for your API key. "
                    "Make sure billing is enabled on Google Cloud."
                ) from e
            if status == 429:
                if "PerDay" in err_body or attempt >= 2:
                    raise RuntimeError(
                        "Gemini API quota exceeded. "
                        "Please wait until your quota resets or upgrade your plan."
                    ) from e
                time.sleep(2 ** attempt * 5)
                continue
            if status >= 500 and attempt < 2:
                time.sleep(2 ** attempt * 4)
                continue

            raise RuntimeError(f"Gemini API error (HTTP {status}): {err_body[:300]}") from e

        except urllib.error.URLError as e:
            if attempt < 2:
                time.sleep(5)
                continue
            raise RuntimeError(f"Could not reach Gemini API: {e.reason}") from e

    raise RuntimeError("No response from Gemini API after retries.")


# ─────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────

class NotAResumeError(Exception):
    """Raised when the uploaded document is not a resume/CV."""
    def __init__(self, document_type: str, reason: str):
        self.document_type = document_type
        self.reason        = reason
        super().__init__(reason)


def analyze_resume(resume_text: str) -> dict:
    """
    Validate the document is a resume, then return structured ATS data.

    Returns:
        {
            "ats_score":   int,
            "skills":      list[str],
            "suggestions": list[str],
            "job_roles":   list[str]
        }

    Raises:
        EnvironmentError   — API key not set
        NotAResumeError    — document is not a resume/CV
        RuntimeError       — quota / rate-limit / HTTP error
        ValueError         — cannot parse JSON from response
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    prompt   = USER_PROMPT.format(resume_text=resume_text[:8000])
    raw_text = _call_gemini(api_key, prompt)

    # Strip accidental markdown fences
    raw_text = re.sub(r"```json\s*", "", raw_text)
    raw_text = re.sub(r"```\s*",     "", raw_text)

    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match:
        raise ValueError(f"Gemini returned a non-JSON response:\n{raw_text[:500]}")

    data = json.loads(json_match.group())

    # ── Document type check ───────────────────────────────────────
    if not data.get("is_resume", True):
        raise NotAResumeError(
            document_type=data.get("document_type", "Unknown document"),
            reason=data.get("reason", "This does not appear to be a resume or CV."),
        )

    # ── Sanity-check required ATS fields ─────────────────────────
    if "ats_score" not in data:
        raise ValueError("Gemini response is missing required ATS fields.")

    return {
        "ats_score":   max(0, min(100, int(data.get("ats_score", 0)))),
        "skills":      [str(s) for s in data.get("skills",      [])][:30],
        "suggestions": [str(s) for s in data.get("suggestions", [])][:10],
        "job_roles":   [str(r) for r in data.get("job_roles",   [])][:5],
    }
