import re

# Read resume text
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    resume_text = f.read().lower()

# Read job description text
with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read().lower()

# Stopwords to ignore
stopwords = {
    "and", "or", "the", "with", "for", "a", "to", "of", "in", "we", "are",
    "be", "is", "should", "have", "has", "will", "can", "our", "their",
    "candidate", "looking", "skills", "experience", "role", "job", "work",
    "business", "team", "responsibilities", "basics", "familiar", "working"
}

# Extract single words from JD
words = re.findall(r'\b[a-zA-Z]+\b', jd_text)
single_keywords = [w for w in words if w not in stopwords and len(w) > 2]

# Keep only unique single words
single_keywords = list(set(single_keywords))

# Define useful multi-word phrases to check
important_phrases = [
    "machine learning",
    "data analysis",
    "data cleaning",
    "business stakeholders",
    "power bi"
]

# Only keep phrases that actually appear in JD
jd_phrases = [phrase for phrase in important_phrases if phrase in jd_text]

# Combine single keywords + useful phrases
jd_keywords = list(set(single_keywords + jd_phrases))

# Match keywords in resume
matched_keywords = [kw for kw in jd_keywords if kw in resume_text]

# Score
score = len(matched_keywords) / len(jd_keywords) if jd_keywords else 0

print("JD Keywords:", jd_keywords)
print("Matched Keywords:", matched_keywords)
print(f"Improved Skill Match Score: {score:.4f}")