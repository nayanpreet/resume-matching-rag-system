import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# STEP 1: READ FILES
# -----------------------------
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    resume_text = f.read().lower()

with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read().lower()

# -----------------------------
# STEP 2: SEMANTIC SIMILARITY SCORE
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

resume_embedding = model.encode([resume_text])
jd_embedding = model.encode([jd_text])

semantic_score = cosine_similarity(resume_embedding, jd_embedding)[0][0]

# -----------------------------
# STEP 3: SKILL MATCH SCORE
# -----------------------------
stopwords = {
    "and", "or", "the", "with", "for", "a", "to", "of", "in", "we", "are",
    "be", "is", "should", "have", "has", "will", "can", "our", "their",
    "candidate", "looking", "skills", "experience", "role", "job", "work",
    "business", "team", "responsibilities", "basics", "familiar", "working"
}

words = re.findall(r'\b[a-zA-Z]+\b', jd_text)
single_keywords = [w for w in words if w not in stopwords and len(w) > 2]
single_keywords = list(set(single_keywords))

important_phrases = [
    "machine learning",
    "data analysis",
    "data cleaning",
    "business stakeholders",
    "power bi"
]

jd_phrases = [phrase for phrase in important_phrases if phrase in jd_text]

jd_keywords = list(set(single_keywords + jd_phrases))
matched_keywords = [kw for kw in jd_keywords if kw in resume_text]

skill_score = len(matched_keywords) / len(jd_keywords) if jd_keywords else 0

# -----------------------------
# STEP 4: FINAL SCORE
# -----------------------------
final_score = (0.7 * semantic_score) + (0.3 * skill_score)

# -----------------------------
# STEP 5: PRINT RESULTS
# -----------------------------
print(f"Semantic Score: {semantic_score:.4f}")
print(f"Skill Match Score: {skill_score:.4f}")
print(f"Final Candidate Score: {final_score:.4f}")
print("Matched Keywords:", matched_keywords)