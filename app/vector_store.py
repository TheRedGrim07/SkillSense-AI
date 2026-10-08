import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define the local folder where the database files will be saved
CHROMA_PATH = "chroma_db"

# Initialize the embedding model globally
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_resume_to_vector_db(pdf_path):
    """Loads a PDF, chunks the text, and stores semantic embeddings in ChromaDB."""
    
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # 2. Configure the Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # 3. Perform the chunking
    chunks = text_splitter.split_documents(pages)

# 4. Create and populate the database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    return vector_db

def search_resume_skills(query_text, k=3):
    """Searches the database for the chunks that best match the query."""
    
    # 1. Connect to the existing database folder using the same AI model
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 2. Perform KNN (K-Nearest Neighbors) Search
    results = vector_db.similarity_search_with_score(query_text, k=k)
    
    print(f"\n--- Top matches for: '{query_text}' ---")
    for doc, score in results:
        print(f"\n[Score: {round(score, 4)}] {doc.page_content}")

if __name__ == "__main__":
    import os
    sample_pdf = "sample_resume.pdf" # Put your KIIT resume in the folder and name it this
    
    if os.path.exists(sample_pdf):
        print("Processing resume...")
        process_resume_to_vector_db(sample_pdf)
        
        print("\nTesting Semantic Search...")
        # Notice we are asking for "relational databases", not the exact word "MySQL"
        search_resume_skills("Relational databases and SQL") 
    else:
        print("Please place a sample_resume.pdf in the project folder!")