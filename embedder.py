from sentence_transformers import SentenceTransformer
from chunker import chunk_text

# Load model (this downloads once, then reuses)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read resume text
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create chunks
chunks = chunk_text(text)

# Generate embeddings
embeddings = model.encode(chunks)

print(f"Total chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")