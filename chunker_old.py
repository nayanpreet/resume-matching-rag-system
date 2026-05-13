def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Simple section tagging
        lower_chunk = chunk.lower()

        if "experience" in lower_chunk or "worked" in lower_chunk:
            section = "experience"
        elif "project" in lower_chunk:
            section = "project"
        elif "skill" in lower_chunk:
            section = "skills"
        elif "education" in lower_chunk:
            section = "education"
        else:
            section = "other"

        chunks.append({
            "text": chunk,
            "section": section
        })

        start += (chunk_size - overlap)

    return chunks


# Test the function
if __name__ == "__main__":
    with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print("\n")