from keybert import KeyBERT

def extract_keywords_from_jd(jd_text, top_n=12):
    kw_model = KeyBERT()
    
    keywords = kw_model.extract_keywords(
        jd_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )
    
    return keywords


if __name__ == "__main__":
    with open("data/job_descriptions/sample_jd.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()

    keywords = extract_keywords_from_jd(jd_text)

    print("Extracted Keywords:\n")
    for kw, score in keywords:
        print(f"- {kw} ({score:.4f})")