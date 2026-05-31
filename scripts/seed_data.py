# scripts/seed_data.py

"""
Seeds the database with sample job descriptions.
Run after init_db.py:
    python scripts/seed_data.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import db_manager
from backend.services.nlp_pipeline import nlp_pipeline
from backend.services.skill_extractor import skill_extractor
from backend.services.parser import resume_parser
import uuid
from datetime import datetime


SAMPLE_JOBS = [
    {
        "title":       "Senior Python Engineer",
        "company":     "TechCorp Inc",
        "description": """
            We are looking for a Senior Python Engineer with 5+ years of experience.
            You will build and maintain scalable REST APIs using FastAPI or Django.
            Required: Python, FastAPI, PostgreSQL, Docker, Kubernetes, AWS.
            Experience with Redis for caching and Celery for async tasks.
            Strong knowledge of microservices architecture and CI/CD pipelines.
            Agile environment using GitHub Actions and Jenkins.
            Bachelor's degree in Computer Science or equivalent.
        """,
        "min_experience_years": 5,
        "education_requirement": "bachelor",
        "employment_type": "full_time",
        "remote_policy": "hybrid",
    },
    {
        "title":       "Machine Learning Engineer",
        "company":     "AI Ventures",
        "description": """
            Seeking a Machine Learning Engineer to build production ML systems.
            Must have strong Python skills and experience with TensorFlow or PyTorch.
            Required: machine learning, deep learning, scikit-learn, pandas, numpy.
            Experience deploying models on AWS SageMaker or Google Cloud AI Platform.
            Knowledge of MLflow for experiment tracking and DVC for data versioning.
            Docker and Kubernetes for containerized model serving.
            Master's degree in Computer Science, Mathematics, or Statistics preferred.
        """,
        "min_experience_years": 3,
        "education_requirement": "master",
        "employment_type": "full_time",
        "remote_policy": "remote",
    },
    {
        "title":       "DevOps Engineer",
        "company":     "CloudFirst Solutions",
        "description": """
            DevOps Engineer needed to manage cloud infrastructure at scale.
            Required: AWS, Terraform, Ansible, Docker, Kubernetes.
            Strong experience with CI/CD: Jenkins, GitHub Actions, ArgoCD.
            Monitoring: Prometheus, Grafana, Datadog, ELK stack.
            Scripting in Python and Bash for automation.
            Linux system administration and networking fundamentals.
            3+ years of experience in a DevOps or SRE role.
        """,
        "min_experience_years": 3,
        "education_requirement": "bachelor",
        "employment_type": "full_time",
        "remote_policy": "onsite",
    },
]


async def seed():
    """Insert sample job descriptions into the database."""
    print("Seeding sample job descriptions...")

    db_manager.connect()
    db = db_manager.get_db()

    seeded = 0
    for job_data in SAMPLE_JOBS:
        job_id      = str(uuid.uuid4())
        description = job_data["description"]

        # Run NLP pipeline on job description
        cleaned     = resume_parser._clean_text(description)
        nlp_features = nlp_pipeline.process(cleaned)
        skill_result = skill_extractor.extract(nlp_features, cleaned)

        doc = {
            "job_id":              job_id,
            "title":               job_data["title"],
            "company":             job_data["company"],
            "description":         description,
            "cleaned_description": cleaned,
            "required_skills":     skill_result.all_skills,
            "min_experience_years": job_data["min_experience_years"],
            "education_requirement": job_data["education_requirement"],
            "employment_type":     job_data["employment_type"],
            "remote_policy":       job_data["remote_policy"],
            "nlp_features":        nlp_features.model_dump(mode="json"),
            "skill_extraction_result": skill_result.model_dump(mode="json"),
            "status":              "active",
            "is_active":           True,
            "created_at":          datetime.utcnow().isoformat(),
        }

        # Upsert by title + company to avoid duplicates
        await db["jobs"].update_one(
            {"title": job_data["title"], "company": job_data["company"]},
            {"$set": doc},
            upsert=True
        )

        print(f"  ✓ Seeded: '{job_data['title']}' at {job_data['company']}")
        print(f"    Skills found: {skill_result.all_skills[:5]}...")
        seeded += 1

    print(f"\nSeeded {seeded} job descriptions.")
    db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(seed())