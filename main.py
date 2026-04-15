"""
main.py
FastAPI — Resume ATS Analyzer API
Deploy on Railway. Set GEMINI_API_KEY in Railway environment variables.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from resume_parser import extract_text
from ai_pipeline import analyze_resume, GEMINI_MODEL

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Resume ATS Analyzer API",
    description=f"AI-powered ATS scoring via {GEMINI_MODEL}. Accepts PDF/DOCX, returns score, skills, suggestions, and job roles.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE      = 5 * 1024 * 1024  # 5 MB

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status":  "ok",
        "message": "Resume ATS Analyzer API is running.",
        "model":   GEMINI_MODEL,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts a PDF or DOCX resume (multipart/form-data, field name = 'file').
    Returns JSON: { ats_score, skills, suggestions, job_roles }
    """
    # 1 — Validate extension
    filename = (file.filename or "").lower()
    ext      = ("." + filename.rsplit(".", 1)[-1]) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or DOCX file.",
        )

    # 2 — Read & size-check
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 5 MB.",
        )

    # 3 — Extract text
    try:
        resume_text = extract_text(content, filename)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from file: {e}",
        )

    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text found. "
                "Make sure the resume is not a scanned image (use a text-based PDF or DOCX)."
            ),
        )

    # 4 — AI analysis
    try:
        result = analyze_resume(resume_text)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {e}")

    # 5 — Return result
    return JSONResponse(content=result)


# ─────────────────────────────────────────────
# Local dev
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
