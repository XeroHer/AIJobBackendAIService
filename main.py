from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from AI.embedding import get_embedding
from AI.skills import extract_skills
from AI.matching import recommend_roles, match_jobs
from AI.roadmap import build_roadmap

app = FastAPI(title="ATS & Career API")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class JobItem(BaseModel):
    id: str
    description: str


class ATSRequest(BaseModel):
    resume: str
    jobDescription: Optional[str] = ""
    jobs: Optional[List[JobItem]] = []

from fastapi.responses import JSONResponse

@app.post("/ats/analyze")
def analyze(data: ATSRequest):
    try:
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
        # Always return JSON even on exception
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})