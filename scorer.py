from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read resume text
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Read job description
with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

# Generate embeddings
resume_embedding = model.encode([resume_text])
jd_embedding = model.encode([jd_text])

# Compute similarity
score = cosine_similarity(resume_embedding, jd_embedding)[0][0]

print(f"Candidate Match Score: {score:.4f}")