"""
main.py
FastAPI application — Resume ATS Analyzer API
Deploy on Railway. Set GEMINI_API_KEY in Railway environment variables.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from resume_parser import extract_text
from ai_pipeline import analyze_resume

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Resume ATS Analyzer API",
    description="Extracts resume text and returns ATS score, skills, suggestions, and job roles via Gemini AI.",
    version="1.0.0",
)

# Allow Hostinger PHP (and any other origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    """Simple health check — confirms the API is live."""
    return {"status": "ok", "message": "Resume ATS Analyzer API is running."}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Main endpoint.
    Accepts a multipart file upload (PDF or DOCX).
    Returns JSON: { ats_score, skills, suggestions, job_roles }
    """
    # 1. Validate file extension
    filename = (file.filename or "").lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or DOCX file.",
        )

    # 2. Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum allowed size is 5 MB.",
        )

    # 3. Extract text from the resume
    try:
        resume_text = extract_text(content, filename)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to extract text from the file: {str(e)}",
        )

    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No readable text found in the file. Make sure the resume is not a scanned image.",
        )

    # 4. Run AI analysis
    try:
        result = analyze_resume(resume_text)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}",
        )

    # 5. Return structured result
    return JSONResponse(content=result)


# ─────────────────────────────────────────────
# Local dev entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
