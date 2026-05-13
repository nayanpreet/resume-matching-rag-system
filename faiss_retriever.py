import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("faiss_store/resume_index.faiss")

# Load section-tagged chunks
with open("faiss_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Read job description
with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

# Convert JD to embedding
jd_embedding = model.encode([jd_text])
jd_embedding = np.array(jd_embedding).astype("float32")

# Search top K chunks
k = 5
distances, indices = index.search(jd_embedding, k)

print("Top matching chunks from FAISS:\n")

for i, idx in enumerate(indices[0]):
    chunk = chunks[idx]
    print(f"Rank {i+1}")
    print(f"Distance: {distances[0][i]}")
    print(f"Section: {chunk['section']}")
    print(chunk["text"])
    print("\n" + "=" * 50 + "\n")