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

import gc
import re
import shutil
import time
from pathlib import Path

import chromadb
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


def clear_vector_store(persist_directory=None):
    """
    Deletes the existing vector store folder from disk, if it exists.

    Why we need this: Chroma.from_documents() ADDS to whatever is already
    saved at persist_directory instead of replacing it. Without this step,
    uploading a second document doesn't start fresh -- it silently mixes
    the new document's chunks in with every document uploaded before it,
    and a question can end up answered using the wrong document's data.

    Why the retry loop: on Windows, a file can't be deleted while anything
    still has it open -- including a Chroma client object left over from
    an earlier question in this same app session. gc.collect() encourages
    Python to release any such lingering references, and the short retries
    give Windows a moment to actually free the file lock. This isn't
    needed on Mac/Linux (they allow deleting open files), but it's a
    no-op there, so it's safe to always include.
    """
    persist_directory = persist_directory or config.VECTOR_STORE_DIRECTORY
    path = Path(persist_directory)

    if not path.exists():
        return

    print(f"Clearing previous vector store at '{persist_directory}'...")

    # Release any Chroma/SQLite objects still sitting in memory before we
    # try to delete the files they might be holding open.
    gc.collect()

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Could not clear '{persist_directory}' -- a previous process "
                    "may still have it open. Close any other running instance of "
                    "this app and try again."
                )
            print(f"  File still in use, retrying ({attempt}/{max_attempts})...")
            time.sleep(0.5)


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

    Why the tagging step exists: a course header like "22CS502PC:
    COMPUTER NETWORKS" often appears only ONCE on a page, in the very
    first chunk of that page. Once the page is split into several
    chunks, later chunks lose that identifying text entirely. Two
    different courses covering near-identical material (e.g. a required
    course and an elective that teach the same topic) then become
    indistinguishable to semantic search, because the actual content is
    nearly the same -- the course code is the ONLY thing that tells them
    apart. Prepending "[Course: ... | UNIT-...]" to every chunk keeps
    that identity attached no matter where a chunk boundary falls, and
    storing it as metadata lets retrieval.py filter by course/unit
    directly instead of relying purely on similarity search.

    If the document doesn't look like a syllabus at all (no course-code
    pattern found anywhere), chunks are returned completely unchanged --
    this only activates for documents that actually have that structure.
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

    The core idea: find WHERE each course/unit heading appears (as a
    character position) within each page's raw text, then use those
    positions like a timeline to figure out which course/unit was "in
    effect" at each chunk's location -- rather than just scanning each
    chunk's own text in isolation, which breaks as soon as a chunk
    happens to fall between two headings.

    Course headers use the chunk's END position (a course transition
    should immediately relabel the chunk it appears in). Unit headings
    use the chunk's START position, since units are denser on a page and
    a heading near a chunk's tail usually belongs more to the NEXT
    chunk -- retrieval.py's text-match fallback catches any chunk this
    under-tags, by checking the chunk's own text for the unit heading
    directly, not just its metadata tag.
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

    We use a local embedding model (all-MiniLM-L6-v2) instead of a paid API
    for this step, since embeddings need to run once per chunk and doing
    that locally is free and fast enough for this project's scale.
    """
    persist_directory = persist_directory or config.VECTOR_STORE_DIRECTORY

    # Start fresh: without this, a second upload would mix its chunks in
    # with whatever was indexed before, instead of replacing it.
    clear_vector_store(persist_directory)

    print("Generating embeddings and saving to the vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # Initialize client explicitly with a string path to prevent tenant lookup issues
    client = chromadb.PersistentClient(path=str(persist_directory))

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name="business_docs",
    )
    print(f"Saved {len(chunks)} chunks to '{persist_directory}'.")
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
