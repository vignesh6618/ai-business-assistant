# config.py
#
# Why this file exists: every setting the app depends on (API key, model
# name, chunk size, etc.) lives here in ONE place, loaded from a .env file.
# This means changing the chunk size or swapping models never requires
# touching the actual logic in ingestion.py / retrieval.py / generation.py.

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- LLM settings ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Embedding model (runs locally, no API key needed) ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Chunking settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval settings ---
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))

# --- Hybrid search settings ---
# RRF_K softens how much a #1 rank matters vs a #10 rank when fusing BM25
# (keyword) and semantic search results. 60 is the standard default from
# the original Reciprocal Rank Fusion paper -- rarely needs tuning.
RRF_K = int(os.getenv("RRF_K", "60"))
HYBRID_CANDIDATE_POOL_SIZE = int(os.getenv("HYBRID_CANDIDATE_POOL_SIZE", "15"))

# --- Storage settings ---
VECTOR_STORE_DIRECTORY = os.getenv("VECTOR_STORE_DIRECTORY", "chroma_db")
RAW_DOCS_DIRECTORY = os.getenv("RAW_DOCS_DIRECTORY", "data/raw_docs")


def check_config():
    """
    Simple sanity check we can call at startup so the app fails with a clear
    message instead of a confusing error halfway through a run.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    print("Config loaded successfully.")
    print(f"  LLM model: {LLM_MODEL_NAME}")
    print(f"  Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Chunk size / overlap: {CHUNK_SIZE} / {CHUNK_OVERLAP}")
    print(f"  Top-K retrieval: {TOP_K_RESULTS}")