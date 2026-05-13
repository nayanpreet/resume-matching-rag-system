import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read resume text
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Chunk resume
chunks = chunk_text(resume_text)

# Extract only chunk text for embeddings
chunk_texts = [chunk["text"] for chunk in chunks]

# Create embeddings
embeddings = model.encode(chunk_texts)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_store/resume_index.faiss")

# Save full chunk metadata
with open("faiss_store/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and section-tagged chunks saved successfully.")