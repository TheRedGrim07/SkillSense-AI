from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import shutil

from app.pipeline import run_skillsense_pipeline
from app.ats_scorer import extract_text_from_pdf

app = FastAPI(title="SkillSense AI", version="1.0")

# Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    jd_text: Optional[str] = Form(None),
    jd_pdf: Optional[UploadFile] = File(None)
):
    """Processes a resume against a JD (provided as text or PDF)."""
    
    if not jd_text and not jd_pdf:
        raise HTTPException(status_code=400, detail="You must provide either JD Text or a JD PDF.")

    temp_resume_path = f"temp_resume_{resume.filename}"
    temp_jd_path = f"temp_jd_{jd_pdf.filename}" if jd_pdf else None
    
    try:
        # 1. Save Resume
        with open(temp_resume_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
            
        # 2. Extract JD Content
        final_jd_content = jd_text or ""
        if jd_pdf:
            with open(temp_jd_path, "wb") as buffer:
                shutil.copyfileobj(jd_pdf.file, buffer)
            final_jd_content = extract_text_from_pdf(temp_jd_path)
            
        # 3. Run Pipeline
        results = run_skillsense_pipeline(temp_resume_path, final_jd_content)
        return JSONResponse(content=results)
        
    finally:
        # 4. Bulletproof Cleanup (Executes even if the pipeline crashes)
        if os.path.exists(temp_resume_path):
            os.remove(temp_resume_path)
        if temp_jd_path and os.path.exists(temp_jd_path):
            os.remove(temp_jd_path)

# Serve the frontend UI from the /static folder
app.mount("/", StaticFiles(directory="static", html=True), name="static")