# SkillSense AI 🚀

**SkillSense AI** is an enterprise-grade Applicant Tracking System (ATS) simulator and strategic gap-analysis pipeline. It utilizes a Retrieval-Augmented Generation (RAG) architecture to evaluate technical resumes against job descriptions, providing candidates with deterministic scoring, semantic skill matching, and guardrailed resume rewriting.

Built as a microservice using **FastAPI** and **LangChain**, this project demonstrates modern AI backend orchestration, context-aware vector retrieval, and strict JSON enforcement for frontend integration.

## 🏗 System Architecture

SkillSense AI processes candidate data through a multi-stage evaluation pipeline:

1. **Deterministic Lexical Engine (TF-IDF):** Parses the resume PDF and extracts exact keyword matches against the Job Description to simulate legacy ATS platforms.
2. **Semantic Vector Store (ChromaDB):** Chunks the resume text using recursive semantic splitting and converts it into dense vector embeddings. It performs a K-Nearest Neighbors (KNN) search to retrieve the candidate's most relevant projects and experience.
3. **LLM Gap Analysis (Groq API):** Analyzes the retrieved vector chunks against the JD, reasoning through conceptual matches (e.g., recognizing *Flask* as a valid subset of a *Python Backend* requirement) and outputting strict JSON payload feedback.
4. **Guardrailed Rewriting:** Rewrites the candidate's most relevant experience into the quantifiable Harvard XYZ format using strict prompt guardrails to prevent the AI from fabricating unmentioned tools or metrics.

## 🛠 Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **AI Orchestration:** LangChain
* **Vector Database:** ChromaDB, Hugging Face Embeddings
* **LLM Inference:** Groq API (GPT-OSS / Llama 3)
* **Frontend:** Vanilla HTML/CSS/JavaScript, AJAX (Single Page Application)
* **Parsing & Data Handling:** PyPDF, Pandas

## ✨ Core Features

* **Multi-Format Ingestion:** Accepts Job Descriptions as either raw text or uploaded PDFs.
* **Strict Structured Output:** Enforces JSON parsing on the LLM to ensure the FastAPI endpoint delivers a predictable, strictly typed payload to the web frontend.
* **Anti-Hallucination Constraints:** Employs zero-shot system prompts to prevent the LLM from inventing candidate experience during the resume rewriting phase.
* **Non-Blocking Architecture:** Decouples the heavy vector ingestion and API calls from the main routing thread.

## 🚀 Local Development Setup

Prerequisites: Python 3.10+ and a free Groq API Key.

1. Clone the repository:
`git clone https://github.com/yourusername/skillsense-ai.git`
`cd skillsense-ai`

2. Set up the virtual environment:
`python -m venv .venv`
Windows: `.venv\Scripts\activate`
Mac/Linux: `source .venv/bin/activate`

3. Install dependencies:
`pip install -r requirements.txt`

4. Configure Environment Variables:
Create a `.env` file in the root directory and add your API key:
`GROQ_API_KEY=your_groq_api_key_here`

5. Start the FastAPI Server:
`uvicorn app.main:app --reload`

6. Access the Application:
Frontend Web UI: `http://127.0.0.1:8000/`
Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

## 🔮 Roadmap & Security Enhancements

As part of the continuous evolution of this project, the following enterprise security and architecture features are slated for Phase 2:

* **Hybrid Search Implementation:** Upgrading the pure semantic search to a hybrid model (Vector Search + Keyword Search) to prevent foundational languages located in dense list formats from being bypassed by the vector similarity threshold.
* **API Security & Rate Limiting:** Implementing strict payload size limits and string truncation to secure the endpoints against Denial of Service (DoS) memory overload attacks.
* **Authentication & RBAC:** Integrating Google OAuth2 and Role-Based Access Control to support multi-tenant user profiles, scaling on the architecture principles established in previous backend systems like UniVoice.

---
*Designed and engineered for highly accurate, hallucination-free AI recruiting workflows.*