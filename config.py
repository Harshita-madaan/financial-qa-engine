from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
INDEX_DIR      = BASE_DIR / "data" / "index"

# ── Filing download ────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "MSFT", "GOOGL"]
FILING_TYPES   = ["10-K", "10-Q"]
NUM_FILINGS    = 2

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 150

# ── Embedding (free — runs locally, no API needed) ─────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K_FAISS    = 20
TOP_K_FINAL    = 5

# ── Generation (free — Google Gemini) ──────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL      = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.0