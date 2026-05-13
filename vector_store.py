import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read resume text
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create chunks
chunks = chunk_text(text)

# Create embeddings
embeddings = model.encode(chunks)

# Convert embeddings to NumPy float32
embeddings = np.array(embeddings).astype("float32")

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(embeddings)

print("FAISS index created successfully.")
print(f"Number of chunks stored: {index.ntotal}")
print(f"Embedding dimension: {dimension}")