import pdfplumber
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    """Reads a PDF and extracts all text into a single string."""
    full_text = ""
    # We use a context manager ('with') so the file automatically closes in memory even if it crashes
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + " "
    return full_text

# Load the small English pipeline. This downloads a small neural network into memory.
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """Uses NLP to extract only Nouns and Proper Nouns (skills, tools, titles)."""
    doc = nlp(text.lower()) # Lowercase for normalization
    
    # List comprehension: Keep token if its Part-of-Speech is NOUN or PROPN
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    # Return as a single string of space-separated keywords for the TF-IDF vectorizer
    return " ".join(keywords)

def calculate_ats_score(resume_text, jd_text):
    """Calculates TF-IDF cosine similarity and returns a Pandas DataFrame analysis."""
    
    # 1. Extract only the meaningful nouns/skills
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)
    
    # 2. Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    # Fit the model on both texts so it learns the vocabulary, then transform them into vectors
    tfidf_matrix = vectorizer.fit_transform([jd_keywords, resume_keywords])
    
    # 3. Calculate Cosine Similarity (Row 0 is JD, Row 1 is Resume)
    match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # 4. Find exact missing keywords for our gap analysis
    resume_set = set(resume_keywords.split())
    jd_set = set(jd_keywords.split())
    
    missing_skills = list(jd_set - resume_set)
    matched_skills = list(jd_set.intersection(resume_set))
    
    # 5. Structure the output using Pandas for clean data handling later
    analysis_df = pd.DataFrame({
        "Metric": ["ATS Match Score", "Total JD Skills", "Matched Skills", "Missing Skills"],
        "Value": [
            f"{round(match_score * 100, 2)}%", 
            len(jd_set), 
            len(matched_skills), 
            len(missing_skills)
        ]
    })
    
    return analysis_df, missing_skills

if __name__ == "__main__":
    # Dummy data to test our logic
    sample_jd = """
    We are looking for a Software Engineer. 
    Requirements: Python, Django, React, SQL, PostgreSQL, Docker, Git. 
    Strong communication skills and experience with REST APIs.
    """
    
    sample_resume = """
    Software Developer. 
    Experience building web applications using Python, Flask, and React. 
    Managed databases using MySQL and SQL. Used Git for version control.
    """
    
    print("Running Deterministic ATS Scorer...\n")
    df, missing = calculate_ats_score(sample_resume, sample_jd)
    
    print(df.to_string(index=False))
    print(f"\nCritical Missing Keywords: {missing[:5]}...") # Show top 5 missing