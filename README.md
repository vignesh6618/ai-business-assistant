# AI Business Development Assistant (Hybrid RAG Engine)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-business-assistant-vignesh.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade financial and business document intelligence platform[cite: 1]. Combines dense semantic vector search with sparse lexical indexing (BM25) via Reciprocal Rank Fusion (RRF) to parse corporate filings, shareholder letters, and earnings reports without hallucination drift.

---

## Architecture Overview

* **Query Reformulation:** Contextual history heuristics resolve conversational references and pronouns.
* **Hybrid Retrieval Core:** ChromaDB dense semantic vectors (`all-MiniLM-L6-v2`) paired with BM25Okapi sparse lexical scoring.
* **Reciprocal Rank Fusion (RRF, k=60):** Combines rank lists to retain exact numbers, dates, and entity mentions.
* **Groq Inference Engine:** Low-latency token generation via Llama-3 / GPT-OSS models.
* **Dual-Pane UI:** Live Streamlit interface with token streaming and PyMuPDF PDF page rendering.

---

## Key Features

* **Hybrid Retrieval (RRF k=60):** Eliminates numeric drops on financial terms and dates by fusing lexical and dense search.
* **Dual-Pane Document Viewer:** PyMuPDF renders original PDF pages interactively alongside real-time token-streamed chat.
* **1-Click Citation Anchors:** Each retrieved chunk exposes source metadata with direct page-jump buttons.
* **Automated Offline Benchmarking:** Custom LLM-as-a-Judge test suite evaluating context recall, answer relevancy, and faithfulness.
* **Defensive System Guardrails:** Negative prompt containment preventing speculation on missing figures.

---

## Quantitative Evaluation Benchmark

Evaluated on corporate filings (Zomato Q3 FY23 Shareholder Letter) across numeric, operational, and adversarial queries:

| Metric | Score | Evaluation Objective |
| :--- | :---: | :--- |
| **Context Recall** | **1.00** | Ground-truth facts captured within top-8 retrieved contexts |
| **Answer Relevancy** | **0.90** | Conciseness, directness, and completeness of generated response |
| **Faithfulness** | **0.83** | Claims strictly supported by source chunks without hallucination |

---

## Tech Stack

* **Frameworks:** LangChain, Streamlit
* **Embeddings & Vector Store:** HuggingFace (`all-MiniLM-L6-v2`), ChromaDB
* **Keyword Index:** Rank-BM25 (BM25Okapi)
* **Inference:** Groq API (`openai/gpt-oss-20b` / `llama-3`)
* **Document Processing:** PyMuPDF (`fitz`)
* **Evaluation:** Automated LLM-as-a-Judge Harness

---

## Local Setup

Clone the repository:
git clone https://github.com/vignesh6618/ai-business-assistant.git
cd ai-business-assistant


Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Run the Streamlit application:
streamlit run ui/app.py


Run the automated evaluation suite:
python evals/run_eval.py
