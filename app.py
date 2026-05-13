import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Resume Matching & Candidate Scoring System")

# Read files
with open("output/sample_resume_text.txt", "r", encoding="utf-8") as f:
    resume_text = f.read().lower()

jd_input = st.text_area("Paste Job Description here")

if jd_input:
    jd_text = jd_input.lower()
else:
    with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
        jd_text = f.read().lower()

# Semantic score
model = SentenceTransformer("all-MiniLM-L6-v2")
resume_embedding = model.encode([resume_text])
jd_embedding = model.encode([jd_text])
semantic_score = cosine_similarity(resume_embedding, jd_embedding)[0][0]

# Skill score
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

# Final score
final_score = (0.7 * semantic_score) + (0.3 * skill_score)

# Interpret score
if final_score >= 0.75:
    match_level = "Strong Match"
elif final_score >= 0.50:
    match_level = "Moderate Match"
else:
    match_level = "Weak Match"

# Show results
st.subheader("Scores")
st.write(f"Semantic Score: {semantic_score:.4f}")
st.write(f"Quick Skill Match Score: {skill_score:.4f}")
st.write(f"Final Candidate Score: {final_score:.4f}")
st.progress(float(final_score))
st.info(
    "Final score is based on both semantic similarity and skill match. "
    "A short input like 'python sql' can give a high skill score but still a lower overall score "
    "because it does not contain enough job context."
)

st.subheader("Matched Keywords")

if matched_keywords:
    for keyword in matched_keywords:
        st.write(f"- {keyword}")
else:
    st.write("No important keyword matches found.")

st.subheader("Job Description")
st.write(jd_text)

st.subheader("Resume Text")
st.write(resume_text)