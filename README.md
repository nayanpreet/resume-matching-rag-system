# AI-Powered Resume Matching & Candidate Scoring System

An end-to-end AI-powered Resume Matching System that uses:

- Semantic Search
- Sentence Transformers
- FAISS Vector Database
- Retrieval-Augmented Generation (RAG)
- Hybrid Ranking Logic

to intelligently match resumes against job descriptions.

---

## Features

- Resume parsing and chunking
- Semantic embeddings using Sentence Transformers
- FAISS-based vector similarity search
- Section-aware retrieval and reranking
- Hybrid scoring system using:
  - semantic similarity
  - section priority
- RAG-style candidate analysis
- Missing skill / gap detection
- Streamlit-based interactive UI

---

## Tech Stack

- Python
- Sentence Transformers
- FAISS
- Scikit-learn
- Streamlit
- NumPy
- KeyBERT

---

## Project Workflow

1. Parse resume
2. Split resume into semantic sections
3. Generate embeddings
4. Store embeddings in FAISS
5. Embed job description
6. Retrieve relevant resume chunks
7. Re-rank using hybrid scoring
8. Generate RAG-style analysis
9. Detect missing skills/gaps

---

## Current Status

Project currently supports:

- Single resume vs single job description
- Semantic retrieval
- Hybrid reranking
- RAG-style explanation generation

Future improvements planned:

- Multi-candidate ranking
- LLM-generated summaries
- Dynamic weighting systems
- Advanced skill extraction
- Production deployment

---

## Repository Structure

```text
resume_matching_project/
│
├── app.py
├── parser.py
├── chunker.py
├── embedder.py
├── faiss_builder.py
├── faiss_retriever.py
├── hybrid_matcher.py
├── rag_answer.py
├── keyword_extractor.py
├── scorer.py
├── final_scorer.py
│
├── data/
│   ├── resumes/
│   └── job_descriptions/
│
├── faiss_store/
├── output/
└── README.md


Author
Nayanpreet Chhabra