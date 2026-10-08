import os
from dotenv import load_dotenv
from groq import Groq

# Load your API key from .env
load_dotenv()

# Initialize the native Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Fetch and print all models your account has access to
print("--- My Authorized Groq Models ---")
models = client.models.list()
for model in models.data:
    print(model.id)