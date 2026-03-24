from AI.embedding import get_embedding
from sentence_transformers import util


ROLE_DATABASE = {
    "Backend Developer":
        "python api backend fastapi flask microservices docker",

    "Machine Learning Engineer":
        "python tensorflow pytorch machine learning mlops ai models",

    "Data Scientist":
        "python pandas numpy machine learning statistics data analysis",

    "DevOps Engineer":
        "docker kubernetes ci cd aws infrastructure automation",

    "AI Engineer":
        "llm transformers deep learning ai machine learning python"
}


def recommend_roles(resume_embedding):
    results = []

    for role, desc in ROLE_DATABASE.items():
        role_embedding = get_embedding(desc)
        score = util.cos_sim(resume_embedding, role_embedding).item() * 100

        results.append({
            "role": role,
            "score": round(score, 2)
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:3]


def match_jobs(resume_embedding, jobs):
    if not jobs:
        return []

    job_texts = [j.description for j in jobs if j.description]

    if not job_texts:
        return []

    job_embeddings = [get_embedding(text) for text in job_texts]

    results = []

    for idx, job_emb in enumerate(job_embeddings):
        score = util.cos_sim(resume_embedding, job_emb).item() * 100

        if score > 40:
            results.append({
                "id": jobs[idx].id,
                "score": round(score, 2)
            })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:5]