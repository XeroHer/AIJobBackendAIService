def build_roadmap(missing_skills):
    roadmap = []

    for skill in missing_skills[:7]:
        roadmap.append({
            "skill": skill,
            "plan": f"Learn {skill} and build a small project using it.",
            "resources": [
                "Official documentation",
                "Hands-on project",
                "Open-source contribution"
            ]
        })

    return roadmap