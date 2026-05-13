import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("faiss_store/resume_index.faiss")

# Load resume chunks
with open("faiss_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Read JD
with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

# Embed JD
jd_embedding = model.encode([jd_text])
jd_embedding = np.array(jd_embedding).astype("float32")

# Retrieve top chunks from FAISS
k = 5
distances, indices = index.search(jd_embedding, k)

# Section priority
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
    chunk_text = chunk["text"]
    chunk_section = chunk["section"]

    # Semantic similarity between JD and this chunk
    chunk_embedding = model.encode([chunk_text])
    sim = cosine_similarity(jd_embedding, np.array(chunk_embedding).astype("float32"))[0][0]

    results.append({
        "section": chunk_section,
        "text": chunk_text,
        "faiss_distance": float(distances[0][i]),
        "semantic_similarity": float(sim),
        "priority": section_priority.get(chunk_section, 6)
    })

# Re-rank by:
# 1. section priority
# 2. semantic similarity descending
for item in results:
    # Convert priority to score (lower priority = higher score)
    priority_score = 1 / item["priority"]

    # Final hybrid score
    item["final_score"] = (0.7 * item["semantic_similarity"]) + (0.3 * priority_score)

# Sort by final score
reranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
print("\nHYBRID MATCH RESULTS:\n")

for i, item in enumerate(reranked, 1):
    print(f"Rank {i}")
    print(f"Section: {item['section']}")
    print(f"FAISS Distance: {item['faiss_distance']:.4f}")
    print(f"Semantic Similarity: {item['semantic_similarity']:.4f}")
    print(f"Final Score: {item['final_score']:.4f}")
    print(item["text"])
    print("\n" + "=" * 60 + "\n")