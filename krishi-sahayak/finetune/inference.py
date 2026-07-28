"""
End-to-end inference: retrieves relevant scheme context via RAG, then
generates an answer using the fine-tuned (LoRA) model.

Usage:
    python inference.py --query "I own 2 acres in Punjab and grow wheat, is there insurance for me?"
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "rag"))
from retriever import SchemeRetriever  # noqa: E402

SYSTEM_PROMPT = (
    "You are Krishi Sahayak, an assistant that helps farmers and rural field "
    "officers in India identify government schemes they are eligible for. "
    "You are given retrieved scheme documents as context. Only recommend a "
    "scheme if the farmer's profile clearly satisfies its eligibility "
    "criteria based on the provided context. If the profile is missing "
    "information needed to confirm eligibility, ask a specific clarifying "
    "question instead of guessing. Always cite the scheme name for any "
    "recommendation you make. Never invent scheme details not present in the "
    "provided context."
)


def load_model(base_model: str, adapter_dir: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"Loaded fine-tuned adapter from {adapter_dir}")
    else:
        print("No adapter found -- running base model only (RAG context still applied).")
    return model, tokenizer


def generate_answer(model, tokenizer, query: str, context: str, max_new_tokens: int = 300):
    user_content = f"Context:\n{context}\n\nFarmer query: {query}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="./krishi-sahayak-lora")
    parser.add_argument("--persist_dir", type=str, default="../rag/chroma_store")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    retriever = SchemeRetriever(persist_dir=args.persist_dir)
    context = retriever.format_context(args.query, top_k=args.top_k)

    model, tokenizer = load_model(args.base_model, args.adapter_dir)
    answer = generate_answer(model, tokenizer, args.query, context)

    print("\n--- Retrieved context ---")
    print(context)
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
