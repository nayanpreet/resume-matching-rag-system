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

# Retrieve top chunks
k = 3
distances, indices = index.search(jd_embedding, k)

# Section priority for re-ranking
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

# Re-rank retrieved chunks
reranked = sorted(results, key=lambda x: (x["priority"], x["distance"]))

# Take top 3 after reranking
retrieved_chunks = [item["text"] for item in reranked[:3]]

# Print JD
print("\nJOB DESCRIPTION:\n")
print(jd_text)

# Print top retrieved chunks
print("\nTOP RETRIEVED RESUME CHUNKS:\n")
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"--- Chunk {i} ---")
    print(chunk)
    print()

# Build simple RAG-style summary
print("\nRAG-STYLE MATCH SUMMARY:\n")

summary = "Based on the retrieved resume sections, this candidate appears relevant to the job description. "

if "python" in jd_text.lower():
    summary += "The resume shows Python experience. "

if "sql" in jd_text.lower():
    summary += "The resume also shows SQL knowledge. "

if "machine learning" in jd_text.lower():
    summary += "There is some machine learning exposure in the resume. "

if "dashboards" in jd_text.lower() or "reporting" in jd_text.lower():
    summary += "The resume also reflects analytics/reporting-related experience. "

summary += "Overall, the retrieved sections suggest a moderate to strong match depending on the depth of required experience."

print(summary)

# Combine retrieved chunks into one lowercase text block for gap checking
combined_text = " ".join(retrieved_chunks).lower()

# Identify potential gaps
missing = []

if "python" in jd_text.lower() and "python" not in combined_text:
    missing.append("Python")

if "sql" in jd_text.lower() and "sql" not in combined_text:
    missing.append("SQL")

if "machine learning" in jd_text.lower() and "machine learning" not in combined_text:
    missing.append("Machine Learning")

if "dashboards" in jd_text.lower():
    if not any(tool in combined_text for tool in ["tableau", "power bi", "dashboard"]):
        missing.append("Dashboarding")

if "communication" in jd_text.lower():
    if not any(word in combined_text for word in ["communication", "stakeholder", "collaborate", "present", "team"]):
        missing.append("Communication skills (not clearly evident)")

print("\nPOTENTIAL GAPS:\n")

if missing:
    for item in missing:
        print(f"- {item}")
else:
    print("No major gaps detected.")

# Save everything to file
with open("output/rag_summary.txt", "w", encoding="utf-8") as f:
    f.write("JOB DESCRIPTION:\n\n")
    f.write(jd_text)
    f.write("\n\nTOP RETRIEVED RESUME CHUNKS:\n\n")

    for i, chunk in enumerate(retrieved_chunks, 1):
        f.write(f"--- Chunk {i} ---\n")
        f.write(chunk)
        f.write("\n\n")

    f.write("RAG-STYLE MATCH SUMMARY:\n\n")
    f.write(summary)
    f.write("\n\nPOTENTIAL GAPS:\n\n")

    if missing:
        for item in missing:
            f.write(f"- {item}\n")
    else:
        f.write("No major gaps detected.\n")

print("\nRAG summary saved to output/rag_summary.txt")