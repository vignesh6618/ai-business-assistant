# pipeline.py
#
# Job of this file: be the single "front door" to the whole RAG system.
# The UI (ui/app.py) never talks to ingestion.py, retrieval.py, or
# generation.py directly — it just calls the two functions below.
# This keeps the UI code simple and means the backend logic can change
# without needing to touch the UI.

from rag_system import ingestion, retrieval, generation


def process_uploaded_file(file_path: str):
    """
    Called when the user uploads a new document from the chat bar.
    Runs the full ingestion pipeline: load -> chunk -> embed -> store.
    """
    print(f"\n--- Processing uploaded file: {file_path} ---")
    vector_store = ingestion.ingest_document(file_path)
    print("--- Document is now searchable. ---\n")
    return vector_store


FOLLOW_UP_WORDS = {"that", "it", "this", "those", "these", "previous", "last", "again", "also", "too", "instead"}


def looks_like_a_follow_up(question: str) -> bool:
    """
    Guesses whether a question is a vague follow-up ("what about that?")
    versus a fully self-contained question ("what is the purpose of a
    Kanban cumulative flow diagram?").

    Why this matters: build_retrieval_query used to glue the previous
    question onto EVERY question, not just follow-ups. That silently
    changed what got searched even for perfectly clear, standalone
    questions -- which made retrieval look "randomly inconsistent" when
    really it was quietly searching for something different than what was
    typed. This heuristic isn't perfect, but it only borrows the previous
    question's keywords when the current one actually seems to need them.
    """
    words = question.lower().replace("?", "").split()
    is_short = len(words) <= 4
    has_reference_word = any(w in FOLLOW_UP_WORDS for w in words)
    return is_short or has_reference_word


def build_retrieval_query(question: str, chat_history=None) -> str:
    """
    Builds the text we actually search the vector store with.

    Why we need this: similarity search only works well if the search text
    contains real keywords. A follow-up question like "what about last
    quarter?" has almost no useful keywords on its own -- so we tack on the
    user's previous question too, which usually carries the missing
    keywords (e.g. "revenue"), and lets retrieval find the right chunks.

    We only do this when the question actually looks like a follow-up
    (see looks_like_a_follow_up) -- otherwise a normal, fully-formed
    question gets searched exactly as typed, with nothing glued onto it.
    """
    if not chat_history or not looks_like_a_follow_up(question):
        return question

    previous_user_messages = [m["content"] for m in chat_history if m["role"] == "user"]
    if not previous_user_messages:
        return question

    last_question = previous_user_messages[-1]
    return f"{last_question} {question}"


def answer_question(question: str, chat_history=None):
    """
    Called when the user types a question in the chat bar.
    Runs the full question-answering pipeline: retrieve -> build prompt -> generate.

    chat_history is an optional list of {"role": ..., "content": ...}
    messages from earlier in the conversation. Passing it in lets the
    assistant handle follow-up questions ("what about last quarter?")
    instead of treating every question as if it's the first one asked.

    Returns both the answer AND the source chunks, so the UI can show the
    user which parts of the document the answer came from.
    """
    print(f"\n--- Answering question: {question} ---")

    retrieval_query = build_retrieval_query(question, chat_history)
    if retrieval_query != question:
        print(f"  (searching with expanded query: '{retrieval_query}')")
    relevant_chunks = retrieval.retrieve_relevant_chunks(retrieval_query)

    prompt = generation.build_prompt(question, relevant_chunks, chat_history)
    answer = generation.generate_answer(prompt)

    print("--- Done. ---\n")
    return answer, relevant_chunks
