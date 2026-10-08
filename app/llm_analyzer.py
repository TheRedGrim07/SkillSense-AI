import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1. Load the hidden API key from the .env file
load_dotenv()

def analyze_resume_gaps(job_description, retrieved_resume_chunks):
    """Uses Gemini to compare the JD against the semantic chunks of the resume."""
    
    # 2. Initialize the Gemini LLM
    # We use gemini-3.8-flash because it is extremely fast and perfect for text analysis
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0
    )
    
    # 3. Define the Structured Output Format
    # We want exact JSON keys so our frontend UI doesn't crash trying to read paragraphs
    parser = JsonOutputParser()
    
    # 4. Construct the Prompt Template
    # The System message sets the persona and rules. The Human message provides the data.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical recruiter and ATS system. 
        Your job is to analyze a candidate's resume snippets against a Job Description.
        You MUST return ONLY valid JSON with the following schema:
        {{
            "matched_skills": ["skill1", "skill2"],
            "missing_skills": ["skill3", "skill4"],
            "feedback": "A short, actionable sentence on what the candidate should improve."
        }}
        Do not include markdown code blocks or any other text outside the JSON."""),
        ("human", """
        Job Description: {jd}
        
        Candidate's Resume Snippets (Retrieved via Vector Search):
        {resume_context}
        """)
    ])
    
    # 5. Build the LangChain (Prompt -> LLM -> JSON Parser)
    chain = prompt | llm | parser
    
    # 6. Execute the Chain
    print("Sending data to Gemini for analysis...")
    result = chain.invoke({
        "jd": job_description,
        "resume_context": retrieved_resume_chunks
    })
    
    return result

def rewrite_resume_bullets(original_bullets, target_jd):
    """
    Rewrites resume bullet points using the Harvard format to better align with the JD,
    with strict guardrails against fabricating tools or experience.
    """
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2  # Low temperature for controlled creativity without hallucination
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an elite technical career coach. 
Your task is to rewrite the candidate's existing resume bullet points to better align with the target Job Description.

STRICT CONSTRAINTS:
1. NEVER invent or assume any tool, framework, library, or metric not mentioned in the original text.
2. If the original text mentions Flask, do NOT say they used Django, even if the JD asks for Django.
3. Use the XYZ Formula: "Accomplished [X] by doing [Z], resulting in [Y]".
4. Start every bullet point with a strong past-tense action verb (e.g., Engineered, Architected, Spearheaded).
5. Output ONLY valid JSON with this format:
{{
    "rewritten_bullets": [
        {{
            "original": "original bullet point",
            "improved": "improved bullet point",
            "reasoning": "why this version is stronger for the target JD"
        }}
    ]
}}
Do not wrap your output in markdown backticks or commentary."""),
        ("human", """
Target Job Description:
{jd}

Original Resume Experience:
{bullets}
""")
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    print("Generating optimized bullet points...")
    return chain.invoke({"jd": target_jd, "bullets": original_bullets})

if __name__ == "__main__":
    # Dummy data to test if the API key and LLM are working
    sample_jd = "Looking for a backend engineer with Python, Django, and Docker experience."
    sample_chunks = "I am a developer. I build APIs using Python and Flask. I deploy using Docker."
    
    try:
        analysis = analyze_resume_gaps(sample_jd, sample_chunks)
        print("\n--- LLM JSON Output ---")
        print(analysis)
    except Exception as e:
        print(f"Error: {e}")
        print("Did you forget to add GOOGLE_API_KEY to your .env file?")