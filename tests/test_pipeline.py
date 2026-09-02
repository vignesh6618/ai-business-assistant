# test_pipeline.py
#
# A few basic sanity checks — not exhaustive, just enough to confirm the
# core building blocks behave the way we expect before relying on them.
#
# Run with:  pytest tests/

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from rag_system import ingestion


def test_split_into_chunks_creates_multiple_chunks():
    """A long page of text should be broken into more than one chunk."""
    long_text = "This is a sentence about business growth. " * 200
    fake_page = [Document(page_content=long_text, metadata={"page": 0})]

    chunks = ingestion.split_into_chunks(fake_page, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1


def test_split_into_chunks_keeps_short_text_as_one_chunk():
    """Text shorter than the chunk size should not be split unnecessarily."""
    short_text = "Revenue grew by 20% this quarter."
    fake_page = [Document(page_content=short_text, metadata={"page": 0})]

    chunks = ingestion.split_into_chunks(fake_page, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0].page_content == short_text


def test_config_loads_default_values():
    """Config should fall back to sensible defaults when .env values are absent."""
    from rag_system import config

    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.TOP_K_RESULTS > 0
