import os
import time
from app.ats_scorer import extract_text_from_pdf, calculate_ats_score
from app.vector_store import process_resume_to_vector_db, search_resume_skills, CHROMA_PATH
from app.llm_analyzer import analyze_resume_gaps, rewrite_resume_bullets
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def run_skillsense_pipeline(pdf_path: str, job_description: str):
    """
    Executes the full end-to-end SkillSense ATS evaluation pipeline.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF not found at {pdf_path}")

    print("=== Step 1: Lexical Parsing & Deterministic ATS Scoring ===")
    raw_resume_text = extract_text_from_pdf(pdf_path)
    score_df, missing_lexical_skills = calculate_ats_score(raw_resume_text, job_description)
    print(score_df.to_string(index=False))

    print("\n=== Step 2: Semantic Chunking & Vector Store Ingestion ===")
    process_resume_to_vector_db(pdf_path)

    # Retrieve context relevant to the job requirements
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    relevant_docs = vector_db.similarity_search(job_description, k=4)
    retrieved_context = "\n".join([doc.page_content for doc in relevant_docs])

    print("\n=== Step 3: LLM Gap Analysis & Strategic Recommendations ===")
    gap_analysis = analyze_resume_gaps(job_description, retrieved_context)
    print(f"Matched Skills: {gap_analysis.get('matched_skills')}")
    print(f"Missing Skills: {gap_analysis.get('missing_skills')}")
    print(f"Advice: {gap_analysis.get('feedback')}")

    print("\n[Loading...]")
    time.sleep(5)

    print("\n=== Step 4: Guardrailed Bullet Point Rewriting ===")
    # Extract the top relevant snippet to test rewriting
    sample_snippet = relevant_docs[0].page_content if relevant_docs else raw_resume_text[:400]
    rewritten_output = rewrite_resume_bullets(sample_snippet, job_description)

    return {
        "deterministic_summary": score_df.to_dict(orient="records"),
        "missing_keywords": missing_lexical_skills,
        "semantic_gap_analysis": gap_analysis,
        "rewritten_bullets": rewritten_output.get("rewritten_bullets", [])
    }

if __name__ == "__main__":
    test_pdf = "sample_resume.pdf"
    test_jd = """
    Software Engineer - Backend & Cloud Services
    Responsibilities:
    - Design and maintain scalable microservices using Python, FastAPI, and Docker.
    - Build and optimize relational database models using PostgreSQL or SQLite.
    - Implement secure user authentication using OAuth2 / RBAC standards.
    - Write clean, maintainable code following CI/CD practices with Git.
    """

    if os.path.exists(test_pdf):
        results = run_skillsense_pipeline(test_pdf, test_jd)
        print("\n\n=== FINAL REWRITTEN BULLETS ===")
        for item in results["rewritten_bullets"]:
            print(f"\n[ORIGINAL]: {item.get('original')}")
            print(f"[IMPROVED]: {item.get('improved')}")
            print(f"[REASONING]: {item.get('reasoning')}")
    else:
        print(f"Please ensure '{test_pdf}' exists in your root folder.")