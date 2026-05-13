import fitz  # PyMuPDF

pdf_path = "data/resumes/sample_resume.pdf"
output_path = "output/sample_resume_text.txt"

doc = fitz.open(pdf_path)
text = ""

for page in doc:
    text += page.get_text()

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Resume text extracted and saved successfully.")