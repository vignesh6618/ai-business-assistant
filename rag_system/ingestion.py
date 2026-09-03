# ingestion.py
#
# Job of this file: take a raw file (PDF, for now) and turn it into small,
# embedded chunks stored in a vector database, so retrieval.py can later
# search over them.
#
# The three steps below map directly to the three functions:
#   1. load_document          -> read the file into plain text pages
#   2. split_into_chunks       -> break long text into small overlapping pieces
#   3. create_and_store_embeddings -> turn chunks into vectors and save them

import functools
import gc
import re
from pathlib import Path

import chromadb
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag_system import config

# Matches headings like "22CS502PC: COMPUTER NETWORKS" -- a course code
# (2 digits, 2 letters, 3 digits, 2 letters) followed by a course name.
COURSE_HEADER_PATTERN = re.compile(
    r"(\d{2}[A-Z]{2}\d{3}[A-Z]{2})\s*:\s*([A-Z][A-Za-z0-9 /&,.\-]+?)(?:\s*\(|\s*\n)"
)

# Matches "UNIT - II", "UNIT-II", "UNIT II", etc.
UNIT_HEADING_PATTERN = re.compile(r"UNIT\s*-?\s*([IVX]+)\b")


@functools.lru_cache(maxsize=1)
def get_embedding_model():
    """
    Caches the PyTorch embedding model in memory so it is only loaded once.
    Prevents running out of memory (OOM crashes) during repeated uploads.
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def load_document(file_path: str):
    """
    Loads a PDF file and returns a list of LangChain Document objects,
    one per page. We use PyPDFLoader because it's simple and works well
    for text-based PDFs (the most common case for business reports).
    """
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")
    return pages


def split_into_chunks(pages, chunk_size=None, chunk_overlap=None):
    """
    Splits pages of text into smaller overlapping chunks, and -- for
    syllabus-style documents -- tags each chunk with which course and
    unit it belongs to.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    print(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")

    looks_like_a_syllabus = any(COURSE_HEADER_PATTERN.search(p.page_content) for p in pages)

    if not looks_like_a_syllabus:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(pages)
        print(f"Created {len(chunks)} chunks.")
        return chunks

    print("Detected course-code headings -- tagging chunks with course/unit context...")
    chunks = _split_and_tag_by_course_and_unit(pages, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks (tagged by course/unit).")
    return chunks


def _split_and_tag_by_course_and_unit(pages, chunk_size, chunk_overlap):
    """
    Does the actual chunking + tagging for syllabus-style documents.
    """
    current_course_code, current_course_name = None, None
    tagged_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )

    for page in pages:
        text = page.page_content
        course_events = [
            (m.start(), m.group(1).strip(), m.group(2).strip())
            for m in COURSE_HEADER_PATTERN.finditer(text)
        ]
        unit_events = [
            (m.start(), f"UNIT-{m.group(1).upper()}") for m in UNIT_HEADING_PATTERN.finditer(text)
        ]

        page_chunks = splitter.split_documents([page])

        for chunk in page_chunks:
            start = chunk.metadata.get("start_index", 0)
            end = start + len(chunk.page_content)

            applicable_courses = [e for e in course_events if e[0] <= end]
            if applicable_courses:
                current_course_code = applicable_courses[-1][1]
                current_course_name = applicable_courses[-1][2]
            course_start_pos = applicable_courses[-1][0] if applicable_courses else -1

            applicable_units = [e for e in unit_events if course_start_pos <= e[0] <= start]
            current_unit = applicable_units[-1][1] if applicable_units else ""

            chunk.metadata["course_code"] = current_course_code or ""
            chunk.metadata["course_name"] = current_course_name or ""
            chunk.metadata["unit"] = current_unit

            if current_course_code:
                header = f"[Course: {current_course_code} - {current_course_name}"
                if current_unit:
                    header += f" | {current_unit}"
                header += "]\n"
                chunk.page_content = header + chunk.page_content

            tagged_chunks.append(chunk)

    return tagged_chunks


def create_and_store_embeddings(chunks, persist_directory=None):
    """
    Converts each text chunk into a numeric vector (an "embedding") and
    saves those vectors to a local Chroma vector database on disk.

    Uses Chroma's client.delete_collection() instead of deleting folders from disk.
    This prevents SQLite file-lock crashes and tenant errors when uploading
    consecutive files without leaving or restarting the app.
    """
    persist_directory = persist_directory or config.VECTOR_STORE_DIRECTORY

    print("Generating embeddings and saving to the vector store...")
    embedding_model = get_embedding_model()

    # 1. Initialize persistent client explicitly
    client = chromadb.PersistentClient(path=str(persist_directory))

    # 2. Reset collection cleanly without deleting disk files or breaking active connections
    try:
        client.delete_collection("business_docs")
        print("Cleared previous collection from database.")
    except Exception:
        # First-time upload: collection doesn't exist yet, safe to pass
        pass

    # 3. Store new embeddings under the same collection name
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name="business_docs",
    )
    print(f"Saved {len(chunks)} chunks to '{persist_directory}'.")

    # Clean up garbage and memory buffers immediately
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vector_store


def ingest_document(file_path: str):
    """
    Convenience function that runs all three ingestion steps in order.
    This is the single function the rest of the app calls when a new
    document is uploaded.
    """
    pages = load_document(file_path)
    chunks = split_into_chunks(pages)
    vector_store = create_and_store_embeddings(chunks)
    return vector_store
