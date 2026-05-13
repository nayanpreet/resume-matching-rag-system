import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read resume text
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Read job description
with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

# Chunk the resume
resume_chunks = chunk_text(resume_text)

# Create embeddings for resume chunks
resume_embeddings = model.encode(resume_chunks)
resume_embeddings = np.array(resume_embeddings).astype("float32")

# Create FAISS index
dimension = resume_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resume_embeddings)

# Create embedding for JD
jd_embedding = model.encode([jd_text])
jd_embedding = np.array(jd_embedding).astype("float32")

# Search top 3 most relevant chunks
k = 3
distances, indices = index.search(jd_embedding, k)

print("Top matching resume chunks:\n")

for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}")
    print(f"Distance: {distances[0][i]}")
    print(resume_chunks[idx])
    print("\n" + "=" * 50 + "\n")