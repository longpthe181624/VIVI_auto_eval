import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_DB_DIR = str(BASE_DIR / os.getenv("CHROMA_DB_DIR", "db"))
RULES_CHROMA_DB_DIR = str(BASE_DIR / os.getenv("RULES_CHROMA_DB_DIR", "db_rules"))
DATA_DIR = str(BASE_DIR / "data")
RULES_DIR = str(BASE_DIR / "data" / "rules")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Text chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_embedding_model():
    """Returns OpenAIEmbeddings if API key is set, otherwise uses local HuggingFace embeddings."""
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    else:
        print("ℹ️ OPENAI_API_KEY not set. Using local HuggingFace embeddings ('all-MiniLM-L6-v2').")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

