"""
eval/evaluate.py — RAGAS evaluation using Groq as judge (free).
Run: python eval/evaluate.py
"""
import sys
import os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from datasets import Dataset
from src.qa_chain import build_chain, ask
import json

# Use Groq as judge LLM (free)
judge_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=3,
))

# Use Cohere for embeddings in evaluation
judge_embeddings = LangchainEmbeddingsWrapper(CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
))

BENCHMARK = [
    {
        "question": "What are Microsoft's main risk factors?",
        "ground_truth": "Microsoft faces risks including competition, cybersecurity threats, regulatory changes, and economic uncertainty.",
    },
    {
        "question": "What does Microsoft say about cloud growth?",
        "ground_truth": "Microsoft reported cloud revenue growth driven by Azure and Microsoft 365.",
    },
    {
        "question": "What are Microsoft's biggest business segments?",
        "ground_truth": "Microsoft operates through Productivity and Business Processes, Intelligent Cloud, and More Personal Computing segments.",
    },
    {
        "question": "What risks does Microsoft mention about AI?",
        "ground_truth": "Microsoft mentions risks related to AI regulation, ethics, and competition in AI services.",
    },
    {
        "question": "What is Microsoft's approach to cybersecurity?",
        "ground_truth": "Microsoft invests heavily in cybersecurity and mentions it as both a risk and a business opportunity.",
    },
]


def run_evaluation():
    print("Building chain...")
    chain = build_chain()

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\nRunning {len(BENCHMARK)} benchmark questions...\n")
    for item in BENCHMARK:
        q = item["question"]
        print(f"Q: {q}")
        result = ask(chain, q)
        print(f"A: {result['answer'][:100]}...\n")

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["docs"]])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    print("Running RAGAS evaluation with Groq as judge...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # Get scores safely
    try:
        df = scores.to_pandas()
        faith = float(df["faithfulness"].mean())
        relevancy = float(df["answer_relevancy"].mean())
    except Exception:
        faith = 0.0
        relevancy = 0.0

    print("\n── RAGAS Results ──────────────────────")
    print(f"  Faithfulness:     {faith:.4f}")
    print(f"  Answer Relevancy: {relevancy:.4f}")
    print("────────────────────────────────────────")

    with open("eval/results.json", "w") as f:
        json.dump({
            "faithfulness":    round(faith, 4),
            "answer_relevancy": round(relevancy, 4),
        }, f, indent=2)
    print("\nSaved → eval/results.json")


if __name__ == "__main__":
    run_evaluation()