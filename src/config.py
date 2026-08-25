import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_DB_DIR = str(BASE_DIR / os.getenv("CHROMA_DB_DIR", "db"))
DATA_DIR = str(BASE_DIR / "data")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Text chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Evaluation Similarity Thresholds (Option A: Uncurated Web Fallback requires stricter bar)
EXPLICIT_SPEC_PASS_THRESHOLD = 0.45
RAG_PASS_THRESHOLD = 0.45
WEB_PASS_THRESHOLD = 0.50  # Fallback web source needs to clear a HIGHER bar (0.50) to override ambiguous score
BORDERLINE_LOW_THRESHOLD = 0.25

_CACHED_EMBEDDING_MODEL = None

def get_embedding_model():
    """Returns cached OpenAIEmbeddings if API key is set, otherwise cached local HuggingFace embeddings."""
    global _CACHED_EMBEDDING_MODEL
    if _CACHED_EMBEDDING_MODEL is not None:
        return _CACHED_EMBEDDING_MODEL

    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        from langchain_openai import OpenAIEmbeddings
        _CACHED_EMBEDDING_MODEL = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    else:
        print("ℹ️ OPENAI_API_KEY not set. Using local HuggingFace embeddings ('all-MiniLM-L6-v2').")
        from langchain_huggingface import HuggingFaceEmbeddings
        _CACHED_EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _CACHED_EMBEDDING_MODEL

