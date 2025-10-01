from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import PyPDF2
import docx

# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Preprocess text (tokenization, lemmatization, stopword removal)
def preprocess_text(text):
    stop_words = nlp.Defaults.stop_words
    doc = nlp(text.lower())
    processed_text = " ".join(
        [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    )
    return processed_text



# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


# Calculate similarity using TF-IDF and Cosine Similarity
def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        resume_file = request.files["resume"]
        job_desc_file = request.files["job_desc"]

        # Extract text based on file type
        resume_text = (
            extract_text_from_pdf(resume_file)
            if resume_file.filename.endswith(".pdf")
            else extract_text_from_docx(resume_file)
        )
        job_desc_text = (
            extract_text_from_pdf(job_desc_file)
            if job_desc_file.filename.endswith(".pdf")
            else extract_text_from_docx(job_desc_file)
        )

        # Preprocess text
        resume_text = preprocess_text(resume_text)
        job_desc_text = preprocess_text(job_desc_text)

        # Calculate similarity score
        score = calculate_similarity(resume_text, job_desc_text)

        # Render the result template with the score
        return render_template("result.html", score=round(score * 100, 2))

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
