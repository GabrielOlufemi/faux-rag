from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# file upload stuff
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE_MB = 15
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# pinecone shit
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "faux-rag-documents")

# llm stuff
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

