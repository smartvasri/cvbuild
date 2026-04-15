"""
ai_pipeline.py
Sends extracted resume text to Google Gemini and returns structured ATS analysis.

Requires environment variable: GEMINI_API_KEY
Get a free key at: https://aistudio.google.com/app/apikey
"""

import os
import json
import re
import google.generativeai as genai


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

ATS_ANALYSIS_PROMPT = """You are a senior HR specialist and ATS (Applicant Tracking System) expert.

Analyze the resume below and produce a structured assessment.

## Scoring Rubric (Total: 100 points)
- Keyword optimization & industry relevance : 30 pts
- Resume structure, formatting, and clarity : 20 pts
- Skills breadth and depth                  : 20 pts
- Work experience impact and quantification : 20 pts
- Education, certifications, achievements   : 10 pts

## Resume:
{resume_text}

## Your Task:
1. **ATS Score** — Give an integer 0–100 based on the rubric above.
2. **Skills** — List every technical tool, programming language, framework, methodology, and notable soft skill found in the resume.
3. **Suggestions** — Write 5 to 7 specific, actionable improvements the candidate can make to increase their ATS score and appeal to recruiters.
4. **Job Roles** — Recommend the top 5 job titles this candidate is best qualified for right now.

Respond ONLY with valid JSON. No markdown fences, no extra commentary.
Use this exact structure:

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
# Main Analysis Function
# ─────────────────────────────────────────────

def analyze_resume(resume_text: str) -> dict:
    """
    Send resume text to Gemini 1.5 Flash and parse the structured JSON response.

    Returns:
        {
            "ats_score": int,
            "skills": list[str],
            "suggestions": list[str],
            "job_roles": list[str]
        }
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    # Configure the Gemini SDK
    genai.configure(api_key=api_key)

    # Use gemini-2.0-flash — fast, free-tier friendly, and currently supported
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.3,      # Low temp for consistent, factual output
            "top_p": 0.9,
            "max_output_tokens": 2048,
        },
    )

    # Build and send the prompt
    prompt = ATS_ANALYSIS_PROMPT.format(resume_text=resume_text[:8000])  # Trim to ~8k chars
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Strip markdown code fences if Gemini wraps the JSON anyway
    raw_text = re.sub(r"```json\s*", "", raw_text)
    raw_text = re.sub(r"```\s*", "", raw_text)

    # Extract the JSON object from the response
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match:
        raise ValueError(f"Could not parse JSON from Gemini response:\n{raw_text[:500]}")

    data = json.loads(json_match.group())

    # Sanitize and return
    return {
        "ats_score": max(0, min(100, int(data.get("ats_score", 0)))),
        "skills":      [str(s) for s in data.get("skills", [])][:30],
        "suggestions": [str(s) for s in data.get("suggestions", [])][:10],
        "job_roles":   [str(r) for r in data.get("job_roles", [])][:5],
    }
