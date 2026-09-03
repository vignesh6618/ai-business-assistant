# app.py
import os
import sys
import tempfile

import pymupdf  # PyMuPDF -- renders PDF pages as images for the viewer panel
import streamlit as st

# Allow this file to import from the rag_system package one folder up
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import config, generation, pipeline, retrieval

st.set_page_config(
    page_title="AI Business Development Assistant", page_icon="💬", layout="wide"
)

# --------------------------------------------------------------------------
# UI Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 6rem; }

    /* Chat bubbles -- distinct styling per speaker */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.20);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.20);
    }

    /* Citation Source Boxes */
    .source-box {
        background-color: #121826;
        color: #E2E8F0;
        border: 1px solid rgba(59,130,246,0.25);
        border-left: 3px solid #3B82F6;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .chunk-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3B82F6, #6366F1);
        color: white;
        font-weight: 600;
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 999px;
        margin-bottom: 6px;
        margin-right: 6px;
    }
    .page-pill {
        display: inline-block;
        background-color: rgba(255,255,255,0.08);
        color: #C9D1E8;
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 999px;
        margin-right: 6px;
    }
    .method-pill {
        display: inline-block;
        background-color: rgba(160,185,129,0.15);
        color: #34D399;
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 999px;
    }
    .status-badge {
        display: inline-block;
        background-color: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        color: #C9D1E8;
        font-size: 0.75rem;
        padding: 3px 12px;
        border-radius: 999px;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background-color: #34D399;
        margin-right: 6px;
    }

    .doc-viewer-header {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 2px;
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .doc-viewer-subheader {
        color: #8B93A7;
        font-size: 0.8rem;
        margin-bottom: 12px;
    }
    .doc-page-image img {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 AI Business Development Assistant")
st.caption("Upload a financial or business document, inspect retrieved chunks, and ask questions.")

# System Status Header
status_badges = (
    '<div style="margin-bottom: 1.2rem;">'
    '<span class="status-badge"><span class="status-dot"></span>Ready</span>'
    f'<span class="status-badge">Model: {config.LLM_MODEL_NAME}</span>'
    '<span class="status-badge">Engine: Hybrid BM25 + Dense RRF</span>'
    '</div>'
)
st.markdown(status_badges, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_ready" not in st.session_state:
    st.session_state.document_ready = os.path.exists(config.VECTOR_STORE_DIRECTORY)

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "viewer_page" not in st.session_state:
    st.session_state.viewer_page = 0

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Tracks uploader version to force-clear mobile browser input state
if "uploader_counter" not in st.session_state:
    st.session_state.uploader_counter = 0

# Startup configuration check
try:
    config.check_config()
except ValueError as error:
    st.error(str(error))
    st.stop()


def render_pdf_page(pdf_bytes: bytes, page_index: int):
    """Rasterizes a single page of the uploaded PDF to PNG bytes for preview."""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    # Clamp requested page between 0 and total pages
    total_pages = document.page_count
    safe_page = max(0, min(page_index, total_pages - 1))
    page = document.load_page(safe_page)
    pixmap = page.get_pixmap(dpi=140)
    image_bytes = pixmap.tobytes("png")
    document.close()
    return image_bytes, total_pages


# --------------------------------------------------------------------------
# Layout Architecture
# --------------------------------------------------------------------------
chat_col, viewer_col = st.columns([3, 2], gap="medium")

# ---- LEFT: Interactive Chat ----
with chat_col:
    chat_area = st.container(height=450, border=False)
    with chat_area:
        if st.session_state.chat_history:
            for msg_idx, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("sources"):
                        with st.expander(f"🔍 View {len(message['sources'])} source chunks"):
                            for i, source in enumerate(message["sources"], start=1):
                                page = source.get("page")
                                method = source.get("method", "")
                                page_label = f"Page {page + 1}" if isinstance(page, int) else "Page —"

                                badges = f'<span class="chunk-badge">Chunk {i}</span>'
                                badges += f'<span class="page-pill">{page_label}</span>'
                                if method:
                                    badges += f'<span class="method-pill">{method}</span>'

                                st.markdown(
                                    f'<div class="source-box">{badges}<br>{source["content"]}</div>',
                                    unsafe_allow_html=True,
                                )

                                # Clickable button to jump viewer to this chunk's page
                                if isinstance(page, int) and st.session_state.pdf_bytes is not None:
                                    if st.button(f"📍 View {page_label} in panel", key=f"jump_{msg_idx}_{i}"):
                                        st.session_state.viewer_page = page
                                        st.rerun()

            # Suggested questions chip row (displayed directly after file ingestion)
            only_first_greeting = (
                len(st.session_state.chat_history) == 1
                and st.session_state.chat_history[0]["role"] == "assistant"
            )
            if only_first_greeting:
                st.caption("Quick Starter Prompts:")
                suggestions = [
                    "What was total Adjusted Revenue and growth?",
                    "What was the reported cash balance?",
                    "Summarize key operational risks",
                ]
                cols = st.columns(len(suggestions))
                for col, prompt_text in zip(cols, suggestions):
                    with col:
                        if st.button(prompt_text, key=f"chip_{prompt_text}", use_container_width=True):
                            st.session_state.pending_question = prompt_text
                            st.rerun()
        else:
            st.info("No document uploaded yet. Click **➕** below to add a document.")

    # Bottom Input Row
    upload_col, input_col = st.columns([1, 11])

    with upload_col:
        with st.popover("➕", use_container_width=True):
            st.markdown("**Add document to index**")
            
            # Dynamic key prevents mobile file-lock bug
            uploader_key = f"pdf_file_{st.session_state.uploader_counter}"
            uploaded_file = st.file_uploader(
                "Upload PDF", 
                type=["pdf"], 
                label_visibility="collapsed", 
                key=uploader_key
            )
            
            if uploaded_file is not None:
                if st.button("Index File", use_container_width=True):
                    pdf_bytes = uploaded_file.getvalue()

                    with st.status("Indexing document...", expanded=True) as status:
                        st.write("📄 Extracting text...")
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(pdf_bytes)
                                tmp_path = tmp.name

                            st.write("✂️ Splitting into overlapping chunks...")
                            st.write("🧠 Generating embeddings & building BM25 index...")
                            pipeline.process_uploaded_file(tmp_path)
                            status.update(label="Document indexed successfully!", state="complete", expanded=False)
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)

                    # Update state and increment counter to reset file uploader cleanly on mobile/PC
                    st.session_state.document_ready = True
                    st.session_state.pdf_bytes = pdf_bytes
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.viewer_page = 0
                    st.session_state.uploader_counter += 1
                    st.session_state.chat_history = [
                        {
                            "role": "assistant",
                            "content": f"I've indexed **{uploaded_file.name}**. What insights would you like to explore?",
                        }
                    ]
                    st.rerun()

    with input_col:
        user_question = st.chat_input("Ask a question about the uploaded document...")

# Handle click from quick starter chips
if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None

# ---- RIGHT: Document Preview Pane ----
with viewer_col:
    with st.container(border=True):
        if st.session_state.pdf_bytes is None:
            st.markdown('<div class="doc-viewer-header">Document Viewer</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="doc-viewer-subheader">Upload a PDF to render its pages interactively.</div>',
                unsafe_allow_html=True,
            )
        else:
            image_bytes, total_pages = render_pdf_page(
                st.session_state.pdf_bytes, st.session_state.viewer_page
            )

            st.markdown(
                f'<div class="doc-viewer-header">{st.session_state.pdf_name}</div>'
                f'<div class="doc-viewer-subheader">Page {st.session_state.viewer_page + 1} of {total_pages}</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="doc-page-image">', unsafe_allow_html=True)
            st.image(image_bytes, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            prev_col, _, next_col = st.columns([1, 2, 1])
            with prev_col:
                if st.button("← Prev", disabled=st.session_state.viewer_page <= 0, use_container_width=True):
                    st.session_state.viewer_page -= 1
                    st.rerun()
            with next_col:
                if st.button(
                    "Next →",
                    disabled=st.session_state.viewer_page >= total_pages - 1,
                    use_container_width=True,
                ):
                    st.session_state.viewer_page += 1
                    st.rerun()

# --------------------------------------------------------------------------
# Question Processing with Live Streaming
# --------------------------------------------------------------------------
if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    if not st.session_state.document_ready:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "Please upload and index a PDF first using the **➕** button.",
            }
        )
        st.rerun()
    else:
        recent_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1][-6:]
        ]

        # 1. Retrieve relevant chunks
        retrieval_query = pipeline.build_retrieval_query(user_question, recent_history)
        source_chunks = retrieval.retrieve_relevant_chunks(retrieval_query)

        # 2. Build contextual prompt
        prompt = generation.build_prompt(user_question, source_chunks, recent_history)

        # 3. Stream output to UI in real-time
        with chat_area:
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                answer = st.write_stream(generation.stream_answer(prompt))

        # 4. Save response & citation metadata to history
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [
                    {
                        "content": chunk.page_content,
                        "page": chunk.metadata.get("page"),
                        "method": chunk.metadata.get("retrieval_method", ""),
                    }
                    for chunk in source_chunks
                ],
            }
        )
        st.rerun()
