# generation.py
#
# Job of this file: take the question + the relevant chunks we retrieved,
# build a prompt that tells the LLM to only use those chunks, and call the
# LLM to get back an answer.

from langchain_groq import ChatGroq

from rag_system import config


def format_chat_history(chat_history) -> str:
    """
    Turns a list of {"role": ..., "content": ...} messages into a plain
    text block the LLM can read, e.g.:

        User: What was Adjusted Revenue in Q3?
        Assistant: INR 23.63 billion.

    We keep this separate from the retrieved document context so the LLM
    can tell the difference between "what we already discussed" and
    "what's actually in the document."
    """
    if not chat_history:
        return "(no earlier messages)"

    lines = []
    for message in chat_history:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def build_prompt(question: str, relevant_chunks, chat_history=None) -> str:
    """
    Combines the retrieved chunks into a single "context" block and wraps
    it with instructions for the LLM.

    Why we instruct the model to only use the given context: without this,
    the LLM will happily answer from its own general knowledge, which
    defeats the purpose of RAG (grounding answers in the uploaded document).

    Why we also include chat_history: without it, a follow-up question like
    "what about last quarter?" makes no sense to the LLM on its own -- it
    needs to see the earlier question ("what was Q3 revenue?") to know what
    "that" or "it" or "last quarter" is even referring to.
    """
    context_text = "\n\n---\n\n".join(chunk.page_content for chunk in relevant_chunks)
    history_text = format_chat_history(chat_history)

    prompt = f"""You are a helpful business assistant. Answer the question
using ONLY the document context provided below.

Strict rules:
- Every fact in your answer must come from the document context below.
- Do not add facts, examples, or explanations from your own general
  knowledge, even if they are true or commonly known.
- Do not add analogies, illustrative comparisons, or metaphors (e.g. "it's
  like a delivery box") unless that exact analogy appears in the context.
- Do not add qualifiers about how common, rare, typical, or widely-used
  something is, unless the context states that directly.
- Do not introduce terms or concepts that are not present in the context,
  unless the question specifically asks you to define something that IS
  in the context.
- Stay close to the context's own wording rather than elaborating or
  generalizing beyond what it actually says.
- If the context only partially answers the question, answer only the
  part it supports, and say what's missing rather than filling the gap.
- If the context does not contain the answer at all, say exactly:
  "I could not find that in the uploaded document."

Use the conversation history only to understand what the user is referring
to (e.g. "that", "it", "the previous one") -- never use it as a source of
facts. All facts must come from the document context.

Conversation so far:
{history_text}

Document context:
{context_text}

Current question: {question}

Answer:"""
    return prompt


def generate_answer(prompt: str) -> str:
    """
    Sends the prompt to the LLM (hosted on Groq) and returns the generated
    answer as plain text.
    """
    print("Sending prompt to the LLM...")
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        groq_api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
    )
    response = llm.invoke(prompt)
    print("Received answer from the LLM.")
    return response.content


# (bottom of rag_system/generation.py)

def generate_answer(prompt: str) -> str:
    """
    Sends the prompt to the LLM (hosted on Groq) and returns the generated
    answer as plain text.
    """
    print("Sending prompt to the LLM...")
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        groq_api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
    )
    response = llm.invoke(prompt)
    print("Received answer from the LLM.")
    return response.content


def stream_answer(prompt: str):
    """
    Streams response tokens from Groq as a generator so Streamlit's
    st.write_stream can render typewriter-style output in real time.
    """
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        groq_api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
    )
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content


