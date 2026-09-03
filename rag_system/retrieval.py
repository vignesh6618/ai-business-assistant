# retrieval.py
#
# Job of this file: given a question, find the most relevant chunks of
# text we already stored in the vector database during ingestion.

import re

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_system import config

# Same patterns as ingestion.py, applied to the QUESTION instead of the
# document -- so we can tell when a user is asking about one specific
# course/unit, versus asking a normal open-ended question.
COURSE_CODE_PATTERN = re.compile(r"\d{2}[A-Z]{2}\d{3}[A-Z]{2}")
UNIT_ROMAN_PATTERN = re.compile(r"UNIT\s*-?\s*([IVX]+)\b", re.IGNORECASE)
UNIT_ARABIC_PATTERN = re.compile(r"UNIT\s*-?\s*(\d{1,2})\b", re.IGNORECASE)
ARABIC_TO_ROMAN_UNIT = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
    "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X"
}

# Common connector words that shouldn't count as "distinguishing" when
# matching a course name against a question.
COURSE_NAME_STOPWORDS = {"and", "for", "the", "to", "of", "in", "on", "with", "a", "an"}

# Cap how many chunks a "give me the whole unit" request can return, so a
# very long section still fits comfortably in the LLM's context window.
MAX_COMPLETE_SECTION_CHUNKS = 15


def load_vector_store(persist_directory=None):
    """
    Opens the existing vector database from disk so we can search it.
    This uses the SAME embedding model that was used during ingestion —
    if these ever mismatch, similarity search will not work correctly.
    """
    persist_directory = persist_directory or config.VECTOR_STORE_DIRECTORY
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # Initialize client explicitly with a string path to prevent tenant lookup issues
    client = chromadb.PersistentClient(path=str(persist_directory))

    vector_store = Chroma(
        client=client,
        embedding_function=embedding_model,
        collection_name="business_docs",
    )
    return vector_store


def _tokenize(text: str) -> list:
    """Simple whitespace/punctuation tokenizer for BM25 -- BM25 works on
    literal word overlap, so it doesn't need anything fancier than this."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _build_bm25_index(vector_store):
    """
    Builds a BM25 keyword-search index over every chunk currently stored.

    Why BM25 alongside embeddings: semantic search is good at "similar in
    meaning" but can be weaker on exact terms -- a specific dollar figure,
    a product code, a name. BM25 is the opposite: pure keyword/term-frequency
    matching, no sense of "meaning" at all. Combining them catches both
    kinds of questions instead of only one.

    This rebuilds the index fresh each call rather than caching it -- simple
    and correct, at the cost of a little speed. Fine for a single-document,
    interactive use case like this one.
    """
    collection = vector_store._collection
    raw = collection.get(include=["documents", "metadatas"])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, documents, metadatas


def _reciprocal_rank_fusion(ranked_lists: dict, top_k: int) -> list:
    """
    Combines several labeled ranked lists of chunk texts into one ranking,
    using Reciprocal Rank Fusion: each chunk's score is the sum of
    1/(RRF_K + rank) across every list it appears in.

    Why this works well: a chunk that shows up in BOTH the semantic
    search results AND the keyword search results gets points from both,
    so it naturally rises to the top -- without needing to tune how much
    to "trust" each method relative to the other.

    ranked_lists maps a method name ("dense", "bm25") to a list of chunk
    texts, already sorted best-first. Returns a list of (chunk_text,
    found_at_ranks) tuples, best first -- found_at_ranks records which
    method(s) surfaced this chunk and at what rank, e.g. {"dense": 2,
    "bm25": 1}, so the UI can show exactly how a chunk was found instead
    of a single opaque number.
    """
    scores = {}
    found_at_ranks = {}
    for method, ranked_list in ranked_lists.items():
        for rank, chunk_text in enumerate(ranked_list, start=1):
            scores[chunk_text] = scores.get(chunk_text, 0.0) + 1.0 / (config.RRF_K + rank)
            found_at_ranks.setdefault(chunk_text, {})[method] = rank

    ranked_texts = sorted(scores.keys(), key=lambda text: -scores[text])
    return [(text, found_at_ranks[text]) for text in ranked_texts[:top_k]]


def hybrid_search(question: str, vector_store, top_k: int) -> list:
    """
    Retrieves chunks using BOTH dense (semantic) and sparse (BM25 keyword)
    search, then fuses the two rankings with Reciprocal Rank Fusion.

    This is the general-purpose upgrade over plain similarity search: it
    doesn't need any document-specific pattern (unlike the course/unit exact
    matching below, which only helps syllabus-style PDFs) -- it helps any
    document where a question includes a specific term, number, or name
    that pure semantic similarity might under-rank.

    Each returned chunk carries a "retrieval_method" metadata field (e.g.
    "Dense #2 + BM25 #1") recording exactly how it was found, so the UI
    can show that instead of an opaque black box.
    """
    dense_docs = vector_store.similarity_search(question, k=config.HYBRID_CANDIDATE_POOL_SIZE)
    dense_texts = [d.page_content for d in dense_docs]

    bm25, documents, metadatas = _build_bm25_index(vector_store)
    bm25_scores = bm25.get_scores(_tokenize(question))
    bm25_ranked_indices = sorted(range(len(documents)), key=lambda i: -bm25_scores[i])
    bm25_texts = [documents[i] for i in bm25_ranked_indices[:config.HYBRID_CANDIDATE_POOL_SIZE]]

    fused = _reciprocal_rank_fusion({"dense": dense_texts, "bm25": bm25_texts}, top_k=top_k)

    # Base metadata (page number, source, etc.) for each chunk text
    text_to_base_meta = {d.page_content: d.metadata for d in dense_docs}
    for text, meta in zip(documents, metadatas):
        text_to_base_meta.setdefault(text, meta)

    results = []
    for text, ranks in fused:
        meta = dict(text_to_base_meta.get(text, {}))
        labels = []
        if "dense" in ranks:
            labels.append(f"Dense #{ranks['dense']}")
        if "bm25" in ranks:
            labels.append(f"BM25 #{ranks['bm25']}")
        meta["retrieval_method"] = " + ".join(labels)
        results.append(Document(page_content=text, metadata=meta))

    return results


def detect_course_code(text: str):
    """Finds a course code like '22CS502PC' in the given text, if present."""
    match = COURSE_CODE_PATTERN.search(text.upper())
    return match.group(0) if match else None


def detect_unit_mention(text: str):
    """
    Finds a unit mention in the given text, if present -- handles both
    "Unit II" (Roman numeral, matching how it's written in most syllabus
    PDFs) and "Unit 2" (arabic numeral, how people actually tend to type
    it when asking a question). Both normalize to the same "UNIT-II"
    format used in the stored chunk metadata.
    """
    match = UNIT_ROMAN_PATTERN.search(text)
    if match:
        return f"UNIT-{match.group(1).upper()}"

    match = UNIT_ARABIC_PATTERN.search(text)
    if match and match.group(1) in ARABIC_TO_ROMAN_UNIT:
        return f"UNIT-{ARABIC_TO_ROMAN_UNIT[match.group(1)]}"

    return None


def _normalize_word(word: str) -> str:
    """
    Strips a trailing 's' from longer words so singular/plural variants
    match each other (e.g. "network" and "networks"). Short words are left
    alone so this doesn't mangle things like "was" or "gas".
    """
    word = word.lower()
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _significant_words(name: str) -> set:
    """
    Breaks a course name into the words that actually distinguish it --
    dropping short connector words like "and"/"of"/"the" -- and normalizes
    each one so singular/plural phrasing doesn't cause a false mismatch.
    Used to check whether a question is specific enough to be talking
    about this course.
    """
    words = re.findall(r"[a-zA-Z]+", name.lower())
    return {_normalize_word(w) for w in words if len(w) >= 3 and w not in COURSE_NAME_STOPWORDS}


def detect_course_by_name(question: str, vector_store) -> list:
    """
    Finds which course(s) a question is referring to by NAME, for
    questions that don't include the exact course code -- e.g. "the
    computer networks subject" instead of "22CS502PC".

    Why this matters: two different courses can share words in their
    name (a required course and its lab, or a required course and a
    similarly-named elective). A question only "fully matches" a course
    if EVERY distinguishing word in that course's name appears in the
    question -- so "computer networks subject" matches "COMPUTER
    NETWORKS" but not "INTRODUCTION TO COMPUTER NETWORKS" (missing
    "introduction") or "COMPUTER NETWORKS LAB" (missing "lab").

    If a question satisfies more than one course this way, and one
    match's words are a strict subset of another's, the more specific
    (longer) name wins -- "computer networks lab" satisfies both
    "COMPUTER NETWORKS" and "COMPUTER NETWORKS LAB", and the lab is
    clearly the better answer.

    Returns a list of matching course codes: empty if none, one if
    unambiguous, more than one if genuinely ambiguous.
    """
    collection = vector_store._collection
    all_metadata = collection.get(include=["metadatas"]).get("metadatas", [])

    course_names = {}
    for meta in all_metadata:
        code = meta.get("course_code")
        name = meta.get("course_name")
        if code and name:
            course_names[code] = name

    if not course_names:
        return []

    question_words = {_normalize_word(w) for w in re.findall(r"[a-zA-Z]+", question.lower())}
    raw_matches = {}
    for code, name in course_names.items():
        words = _significant_words(name)
        if words and words.issubset(question_words):
            raw_matches[code] = words

    final_matches = [
        code for code, words in raw_matches.items()
        if not any(words < other_words for other_code, other_words in raw_matches.items() if other_code != code)
    ]
    return final_matches


def _chunk_matches_course(chunk, course_code: str) -> bool:
    """
    True if a chunk belongs to the given course -- checked two ways, not
    just one: via its stored metadata tag, OR by the course code appearing
    literally in its own text. The second check exists because a chunk
    sitting right at a course transition can end up mostly about a course
    its metadata tag doesn't reflect (see ingestion.py's tagging notes).
    """
    return chunk.metadata.get("course_code") == course_code or course_code in chunk.page_content


def _chunk_matches_unit(chunk, unit: str) -> bool:
    """
    True if a chunk belongs to the given unit -- same two-way check as
    above. The text-match fallback is what catches a chunk that starts a
    new unit partway through but whose metadata tag (based on the chunk's
    START position) hasn't caught up to it yet.
    """
    if chunk.metadata.get("unit") == unit:
        return True
    unit_number = unit.replace("UNIT-", "")
    return bool(re.search(rf"UNIT\s*-?\s*{unit_number}\b", chunk.page_content))


def get_chunks_by_course_and_unit(vector_store, course_code: str, unit: str = None):
    """
    Fetches ALL chunks belonging to a specific course (and optionally a
    specific unit within it) directly from the vector store -- bypassing
    similarity ranking entirely.

    Why bypass similarity search here: a question like "give me the
    complete Unit II syllabus" wants EVERY chunk that belongs to that
    section, not just the handful that happen to rank as "most similar"
    to the question's wording. Similarity search is the wrong tool for
    "fetch a complete, exact section" -- a direct metadata lookup is.
    """
    collection = vector_store._collection
    raw_results = collection.get(where={"course_code": course_code})

    all_course_chunks = []
    for text, meta in zip(raw_results.get("documents", []), raw_results.get("metadatas", [])):
        meta = dict(meta)
        meta["retrieval_method"] = "Exact match (course/unit)"
        all_course_chunks.append(Document(page_content=text, metadata=meta))

    if unit is None:
        return all_course_chunks[:MAX_COMPLETE_SECTION_CHUNKS]

    matched = [c for c in all_course_chunks if _chunk_matches_unit(c, unit)]
    return matched[:MAX_COMPLETE_SECTION_CHUNKS]


def retrieve_relevant_chunks(question: str, vector_store=None, top_k=None):
    """
    Searches the vector store for the chunks most relevant to the question.

    For most questions, this means semantic similarity search -- "similar
    in meaning" (not just matching keywords) is the whole point of using
    embeddings, since a question like "how much revenue did they make" can
    still match a chunk that says "total income was..." with no shared
    words.

    But if the question mentions a specific course code (and this document
    has course-coded sections -- see ingestion.py), similarity search is
    the wrong tool: two courses can cover near-identical material, and
    pure semantic ranking can't reliably tell them apart. In that case we
    fetch chunks by their exact course/unit tag instead of ranking by
    similarity.
    """
    top_k = top_k or config.TOP_K_RESULTS
    vector_store = vector_store or load_vector_store()

    course_code = detect_course_code(question)
    unit = detect_unit_mention(question)

    if course_code:
        print(f"Detected course code '{course_code}' in the question -- using exact matching.")
        results = get_chunks_by_course_and_unit(vector_store, course_code, unit)
        if not results and unit:
            print(f"No chunks tagged '{unit}' for this course -- it may not use a unit structure. "
                  f"Falling back to all chunks for this course.")
            results = get_chunks_by_course_and_unit(vector_store, course_code, None)
        if results:
            print(f"Found {len(results)} chunks for this course" + (f"/{unit}" if unit else "") + ".")
            return results
        print("No exact matches found for that course code -- falling back to similarity search.")

    else:
        # No exact code in the question -- see if the question names a
        # course closely enough (e.g. "the computer networks subject").
        matching_courses = detect_course_by_name(question, vector_store)
        if len(matching_courses) == 1:
            print(f"Matched course by name: '{matching_courses[0]}' -- using exact matching.")
            results = get_chunks_by_course_and_unit(vector_store, matching_courses[0], unit)
            if not results and unit:
                print(f"No chunks tagged '{unit}' for this course -- falling back to all chunks for this course.")
                results = get_chunks_by_course_and_unit(vector_store, matching_courses[0], None)
            if results:
                print(f"Found {len(results)} chunks for this course" + (f"/{unit}" if unit else "") + ".")
                return results
        elif len(matching_courses) > 1:
            # Genuinely ambiguous -- fetch ALL plausible courses rather than
            # silently guessing one. Each chunk already carries its own
            # "[Course: ...]" label, so the LLM can keep them straight and
            # either answer for all of them or ask which one was meant.
            print(f"Question matches multiple courses {matching_courses} -- fetching all of them.")
            results = []
            for code in matching_courses:
                results.extend(get_chunks_by_course_and_unit(vector_store, code, unit))
            if results:
                return results[:MAX_COMPLETE_SECTION_CHUNKS]

    print(f"No specific course/unit detected -- using hybrid search (semantic + keyword) for: '{question}'")
    results = hybrid_search(question, vector_store, top_k)
    print(f"Found {len(results)} relevant chunks.")
    return results


def retrieve_with_scores(question: str, vector_store=None, top_k=None):
    """
    Same search as retrieve_relevant_chunks, but also returns each chunk's
    similarity score alongside it.

    Why this exists separately: normal answering doesn't need the raw
    scores, only the chunk text -- but when two near-identical questions
    behave differently, seeing the actual scores is the only way to tell
    whether retrieval itself changed, or whether retrieval was fine and the
    LLM's answer changed instead. Chroma's default distance is smaller =
    more similar (it's a distance, not a 0-1 confidence score).
    """
    top_k = top_k or config.TOP_K_RESULTS
    vector_store = vector_store or load_vector_store()

    # similarity_search_with_score returns (Document, distance) pairs
    return vector_store.similarity_search_with_score(question, k=top_k)


def print_debug_retrieval(question: str, vector_store=None, top_k=None):
    """
    Prints a side-by-side-friendly diagnostic view of what got retrieved
    for a given question: rank, distance score, source page, and a short
    preview of each chunk's text.

    Meant to be called manually while debugging -- e.g. run the same
    question with and without a "?" and compare the printed output for
    each, to see whether retrieval changed or stayed the same.
    """
    results = retrieve_with_scores(question, vector_store, top_k)

    print(f"\n=== DEBUG: retrieval for question: {question!r} ===")
    for rank, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content.replace("\n", " ")[:160]
        print(f"  #{rank}  score={score:.4f}  page={page}")
        print(f"       preview: {preview}...")
    print("=== end debug ===\n")

    return results
