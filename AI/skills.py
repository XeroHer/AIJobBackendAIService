import re

SKILL_KEYWORDS = {
    "python", "django", "fastapi", "flask", "aws", "docker",
    "kubernetes", "tensorflow", "pytorch", "sql", "postgres",
    "redis", "git", "ci", "cd", "react", "node", "typescript",
    "ml", "ai", "machine learning", "data science"
}


def extract_skills(text: str):
    if not text:
        return []

    text = text.lower()
    skills = set()

    # clean text for better matching (optional)
    clean_text = re.sub(r"[^a-z ]", " ", text)

    for skill in SKILL_KEYWORDS:
        # word boundary prevents substring noise
        if re.search(rf"\b{re.escape(skill)}\b", clean_text):
            skills.add(skill)

    return sorted(skills)