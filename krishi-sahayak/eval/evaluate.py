"""
Evaluates three configurations against the test profile set:
  1. Base model only (no RAG, no fine-tuning)
  2. RAG-only (retrieval + base model)
  3. RAG + fine-tuned LoRA adapter

Metrics:
  - scheme_match_accuracy: did the model recommend the correct scheme_id
    when one was expected?
  - false_recommendation_rate: did the model recommend a scheme when it
    should have asked a clarifying question or declined (ineligible case)?
  - clarification_recall: when clarification was needed, did the model ask
    for more info instead of guessing?

This is a lightweight keyword/heuristic-based scorer meant for quick
iteration. For a rigorous report, complement this with manual review of a
sample of outputs, or use an LLM-as-judge pass.

Usage:
    python evaluate.py --config base
    python evaluate.py --config rag
    python evaluate.py --config rag_finetuned
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "rag"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "finetune"))


CLARIFICATION_MARKERS = [
    "could you tell me", "which state", "how much land", "what type of farmer",
    "need a bit more information", "can you provide", "please share"
]


def looks_like_clarification(answer: str) -> bool:
    lower = answer.lower()
    return any(marker in lower for marker in CLARIFICATION_MARKERS) and "?" in answer


def mentions_scheme(answer: str, scheme_name: str) -> bool:
    return scheme_name.lower() in answer.lower()


def run_eval(config: str, test_profiles_path: str, schemes_path: str):
    with open(test_profiles_path, "r", encoding="utf-8") as f:
        test_profiles = json.load(f)
    with open(schemes_path, "r", encoding="utf-8") as f:
        schemes = {s["id"]: s for s in json.load(f)}

    from retriever import SchemeRetriever
    from inference import load_model, generate_answer, SYSTEM_PROMPT  # noqa: F401

    use_rag = config in ("rag", "rag_finetuned")
    adapter_dir = "../finetune/krishi-sahayak-lora" if config == "rag_finetuned" else None

    retriever = SchemeRetriever(persist_dir="../rag/chroma_store") if use_rag else None
    model, tokenizer = load_model(base_model="Qwen/Qwen2.5-3B-Instruct", adapter_dir=adapter_dir)

    correct = 0
    false_recs = 0
    clarification_hits = 0
    clarification_needed = 0
    results = []

    for profile in test_profiles:
        context = retriever.format_context(profile["query"], top_k=3) if use_rag else "(no retrieval context provided)"
        answer = generate_answer(model, tokenizer, profile["query"], context)

        expected_id = profile["expected_scheme_id"]
        needs_clarification = profile.get("should_ask_clarification", False)
        is_ineligible_case = expected_id is None and not needs_clarification

        asked_clarification = looks_like_clarification(answer)

        if needs_clarification:
            clarification_needed += 1
            if asked_clarification:
                clarification_hits += 1
        elif expected_id is not None:
            scheme_name = schemes[expected_id]["name"]
            if mentions_scheme(answer, scheme_name):
                correct += 1
        elif is_ineligible_case:
            # Should NOT recommend any scheme confidently
            any_scheme_mentioned = any(mentions_scheme(answer, s["name"]) and "eligible" in answer.lower() and "not" not in answer.lower() for s in schemes.values())
            if any_scheme_mentioned:
                false_recs += 1

        results.append({"id": profile["id"], "query": profile["query"], "answer": answer})

    n = len(test_profiles)
    n_expected = sum(1 for p in test_profiles if p["expected_scheme_id"])
    print(f"\n=== Config: {config} ===")
    print(f"Scheme match accuracy: {correct}/{n_expected}")
    print(f"Clarification recall: {clarification_hits}/{clarification_needed}")
    print(f"False recommendations on ineligible cases: {false_recs}")

    out_path = f"eval_results_{config}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full outputs saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["base", "rag", "rag_finetuned"], required=True)
    parser.add_argument("--test_profiles", type=str, default="test_profiles.json")
    parser.add_argument("--schemes", type=str, default="../data/schemes.json")
    args = parser.parse_args()

    run_eval(args.config, args.test_profiles, args.schemes)
