import re

def chunk_text(text):
    section_patterns = [
        ("education", r"(?im)^EDUCATION\s*$"),
        ("skills", r"(?im)^TECHNICAL SKILLS\s*$|^SKILLS\s*$"),
        ("experience", r"(?im)^PROFESSIONAL EXPERIENCE\s*$|^EXPERIENCE\s*$"),
        ("projects", r"(?im)^PROJECTS\s*$"),
        ("certifications", r"(?im)^CERTIFICATIONS\s*$"),
    ]

    # Find headings and their positions
    matches = []
    for section_name, pattern in section_patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), section_name, match.group()))

    # Sort by position in resume
    matches.sort(key=lambda x: x[0])

    chunks = []

    # If no headings found, return whole text as one chunk
    if not matches:
        return [{"text": text, "section": "other"}]

    for i in range(len(matches)):
        start_pos = matches[i][0]
        section_name = matches[i][1]

        if i < len(matches) - 1:
            end_pos = matches[i + 1][0]
        else:
            end_pos = len(text)

        chunk_text = text[start_pos:end_pos].strip()

        chunks.append({
            "text": chunk_text,
            "section": section_name
        })

    return chunks


# Test block
if __name__ == "__main__":
    with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    print(f"Total section chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} | Section: {chunk['section']} ---")
        print(chunk["text"][:800])
        print("\n" + "=" * 50 + "\n")