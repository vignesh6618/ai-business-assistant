import os
import re
import sys
import time
import pandas as pd
from langchain_groq import ChatGroq

# Allow importing directly from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_system import config, pipeline

# ---------------------------------------------------------------------------
# Benchmark Dataset (Positive Extractions + Strict Negative Guardrails)
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    # 1. Exact numeric revenue & growth
    {
        "question": "What was Zomato's total Adjusted Revenue in Q3FY23, and how much did it grow YoY?",
        "ground_truth": "INR 23.63 billion, growing 66% year-on-year."
    },
    # 2. Temporal product launch event
    {
        "question": "When was Zomato Gold launched, and in which quarter?",
        "ground_truth": "Late January 2023, in Q3FY23."
    },
    # 3. Subsidiary operational margin
    {
        "question": "Did Blinkit achieve positive contribution margin, and what was its value in Q3FY23?",
        "ground_truth": "No, it was negative 4.5% of GOV."
    },
    # 4. Balance sheet figure
    {
        "question": "What was the cash balance reported for December 31, 2022?",
        "ground_truth": "INR 5,803 million (or INR 113 billion total liquidity)."
    },
    # 5. Negative Control 1: Future speculation (Tests strict anti-hallucination)
    {
        "question": "What were Zomato's projected earnings for the year 2030 in the report?",
        "ground_truth": "I could not find that in the uploaded document.",
        "is_negative_test": True
    },
    # 6. Negative Control 2: Competitor data (Tests document boundary containment)
    {
        "question": "What is the stock market ticker symbol for Swiggy mentioned in the document?",
        "ground_truth": "I could not find that in the uploaded document.",
        "is_negative_test": True
    }
]

def invoke_with_retry(fn, *args, max_retries=5, delay=4.0):
    """Executes an LLM or pipeline call with exponential backoff on 429 rate limits."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate_limit" in err_msg:
                print(f"  [Rate Limit] Backing off for {delay:.1f}s (Attempt {attempt}/{max_retries})...")
                time.sleep(delay)
                delay *= 1.5
            else:
                raise e
    raise RuntimeError("Exceeded maximum retries due to Groq rate limits.")

def score_with_judge(judge_llm, prompt: str) -> float:
    """Invokes the evaluator LLM and extracts a strict 0.0 to 1.0 numeric float."""
    response = invoke_with_retry(judge_llm.invoke, prompt).content
    match = re.search(r"([01]?\.\d+|[01]\b)", response)
    return float(match.group(1)) if match else 0.0

def main():
    config.check_config()
    judge = ChatGroq(
        model=config.LLM_MODEL_NAME,
        groq_api_key=config.GROQ_API_KEY,
        temperature=0.0
    )

    records = []
    print("\n--- Running Evaluation Benchmark ---")

    for idx, item in enumerate(TEST_QUESTIONS, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        is_neg = item.get("is_negative_test", False)

        print(f"\n[{idx}/{len(TEST_QUESTIONS)}] Testing: {q}")

        # Run pipeline
        answer, retrieved_chunks = invoke_with_retry(pipeline.answer_question, q)
        context_str = "\n\n".join(c.page_content for c in retrieved_chunks)

        if is_neg:
            # Negative Control Test: Evaluates refusal guardrails
            if "could not find" in answer.lower():
                faith_score = 1.0
                rel_score = 1.0
                recall_score = 1.0
            else:
                faith_score = 0.0
                rel_score = 0.0
                recall_score = 0.0
        else:
            # 1. Faithfulness Score (0.0 to 1.0)
            faith_prompt = f"""You are an objective AI evaluator.
Rate the FAITHFULNESS of the answer against the provided context on a scale from 0.0 to 1.0.
- 1.0: Every claim in the answer is explicitly supported by the context without external speculation.
- 0.0: The answer contains hallucinations, external analogies, or ungrounded claims.

Context:
{context_str}

Answer:
{answer}

Respond ONLY with a single numeric score between 0.0 and 1.0:"""
            faith_score = score_with_judge(judge, faith_prompt)

            # 2. Answer Relevancy Score (0.0 to 1.0)
            relevancy_prompt = f"""Rate how directly and concisely the answer addresses the question on a scale from 0.0 to 1.0.
- 1.0: The answer directly and completely answers the prompt without evasion.
- 0.0: The answer is off-topic, evasive, or unhelpful.

Question: {q}
Answer: {answer}

Respond ONLY with a single numeric score between 0.0 and 1.0:"""
            rel_score = score_with_judge(judge, relevancy_prompt)

            # 3. Context Recall Score (0.0 to 1.0)
            recall_prompt = f"""Rate whether the retrieved context contains the necessary facts or figures to answer the question according to the ground truth.
- 1.0: The context contains the required data points from the ground truth.
- 0.0: The context lacks the essential information needed.

Ground Truth: {gt}
Retrieved Context:
{context_str}

Respond ONLY with a single numeric score between 0.0 and 1.0:"""
            recall_score = score_with_judge(judge, recall_prompt)

        records.append({
            "question": q,
            "answer": answer,
            "faithfulness": faith_score,
            "answer_relevancy": rel_score,
            "context_recall": recall_score
        })

        # Throttle call spacing to avoid Groq TPM limits
        time.sleep(3.0)

    # Output and export results
    df = pd.DataFrame(records)
    os.makedirs("evals", exist_ok=True)
    csv_path = "evals/eval_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n================ FINAL EVALUATION RESULTS ================")
    print(df[["question", "faithfulness", "answer_relevancy", "context_recall"]])
    print(f"\nAverage Faithfulness:     {df['faithfulness'].mean():.2f}")
    print(f"Average Answer Relevancy: {df['answer_relevancy'].mean():.2f}")
    print(f"Average Context Recall:   {df['context_recall'].mean():.2f}")
    print(f"\nDetailed CSV exported to: {csv_path}")
    print("==========================================================")

if __name__ == "__main__":
    main()