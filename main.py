# main.py
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ats-api")

# --- Import AI modules safely ---
try:
    from AI.embedding import get_embedding
    from AI.skills import extract_skills
    from AI.matching import recommend_roles, match_jobs
    from AI.roadmap import build_roadmap
except ModuleNotFoundError as e:
    logger.error(f"AI module import failed: {e}")
    get_embedding = extract_skills = recommend_roles = match_jobs = build_roadmap = None

# --- Initialize FastAPI ---
app = FastAPI(title="ATS & Career API")

# --- Enable CORS for frontend ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # Update to your deployed frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper ---
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Request models ---
class JobItem(BaseModel):
    id: str
    description: str

class ATSRequest(BaseModel):
    resume: str
    jobDescription: Optional[str] = ""
    jobs: Optional[List[JobItem]] = []

# --- Health check endpoint ---
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "ok",
        "message": "ATS & Career API is live!"
    }

# --- ATS analyze endpoint ---
@app.post("/ats/analyze")
def analyze(data: ATSRequest):
    logger.info("Received ATS analyze request")
    try:
        if get_embedding is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "AI modules not loaded"}
            )

        resume = (data.resume or "").strip()[:5000]
        job_desc = (data.jobDescription or "").strip()[:5000]
        jobs = data.jobs or []

        if not resume:
            return JSONResponse(status_code=400, content={"success": False, "message": "Resume is required"})

        resume_embedding = get_embedding(resume)

        # ATS score
        if job_desc:
            job_embedding = get_embedding(job_desc)
            similarity = cosine_similarity(resume_embedding, job_embedding)
            ats_score = float(((similarity + 1) / 2) * 100)  # normalized 0–100
        else:
            ats_score = 0.0

        resume_skills = extract_skills(resume)
        job_skills = extract_skills(job_desc)
        missing_skills = [s for s in job_skills if s not in resume_skills]

        recommended_roles = recommend_roles(resume_embedding)
        roadmap = build_roadmap(missing_skills)
        recommended_jobs = match_jobs(resume_embedding, jobs)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "atsScore": round(ats_score, 2),
                "skills": resume_skills,
                "missingSkills": missing_skills,
                "recommendedRoles": recommended_roles,
                "learningRoadmap": roadmap,
                "recommendedJobs": recommended_jobs,
            },
        )

    except Exception as e:
        logger.exception("Error in /ats/analyze")
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})

# --- Run with Render-assigned port ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn_kwargs = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": port,
    }
    # Reload only for local dev
    if os.environ.get("ENV") != "production":
        uvicorn_kwargs["reload"] = True

    import uvicorn
    uvicorn.run(**uvicorn_kwargs)