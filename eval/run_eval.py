import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(override=True)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from app.models import QueryRequest
from app.services.pipeline import run_ask_pipeline

async def evaluate_rag():
    print("Loading golden set...")
    golden_file = Path(__file__).parent / "golden_set.json"
    with open(golden_file, "r") as f:
        golden_data = json.load(f)
    
    questions = []
    ground_truths = []
    answers = []
    contexts = []
    
    print(f"Evaluating {len(golden_data)} questions...")
    for item in golden_data:
        question = item["question"]
        print(f"Running pipeline for: {question}")
        
        # Run pipeline
        req = QueryRequest(question=question, search_mode="hybrid")
        response = await run_ask_pipeline(req)
        
        questions.append(question)
        ground_truths.append(item["ground_truth"])
        answers.append(response.answer)
        
        # Extract context chunks
        context_list = [cite.text for cite in response.citations]
        contexts.append(context_list)
        
    print("Preparing Ragas dataset...")
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    print("Running Ragas evaluation...")
    # Using default OpenAI models for Ragas evaluation metrics.
    # Ensure OPENAI_API_KEY is set in your .env if using standard Ragas.
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )
    
    print("\n--- EVALUATION RESULTS ---")
    print(result)
    
    # Save results
    out_file = Path(__file__).parent / "eval_results.json"
    with open(out_file, "w") as f:
        # result is a dict-like object
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: Ragas typically requires OPENAI_API_KEY for evaluation models.")
        print("Set it in your .env or configure custom LLMs for Ragas.")
        
    asyncio.run(evaluate_rag())
