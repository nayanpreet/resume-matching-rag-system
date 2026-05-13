import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("faiss_store/resume_index.faiss")

# Load chunks
with open("faiss_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Read job description
with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

# Convert JD to embedding
jd_embedding = model.encode([jd_text])
jd_embedding = np.array(jd_embedding).astype("float32")

# Search more chunks first
k = 5
distances, indices = index.search(jd_embedding, k)

# Section priority: smaller number = better
section_priority = {
    "experience": 1,
    "projects": 2,
    "skills": 3,
    "certifications": 4,
    "education": 5,
    "other": 6
}

results = []

for i, idx in enumerate(indices[0]):
    chunk = chunks[idx]
    results.append({
        "distance": float(distances[0][i]),
        "section": chunk["section"],
        "text": chunk["text"],
        "priority": section_priority.get(chunk["section"], 6)
    })

# Re-rank: first by section priority, then by distance
reranked_results = sorted(results, key=lambda x: (x["priority"], x["distance"]))

print("Re-ranked chunks:\n")

for i, result in enumerate(reranked_results, 1):
    print(f"Rank {i}")
    print(f"Section: {result['section']}")
    print(f"Distance: {result['distance']}")
    print(result["text"])
    print("\n" + "=" * 50 + "\n")