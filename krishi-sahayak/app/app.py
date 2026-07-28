"""
Gradio demo for Krishi Sahayak.

Run locally or in Colab after building the RAG index and (optionally)
fine-tuning the LoRA adapter:

    python app.py

Then open the printed local URL, or set share=True for a public link when
running in Colab.
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parent.parent / "rag"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "finetune"))

from retriever import SchemeRetriever  # noqa: E402
from inference import load_model, generate_answer  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "../finetune/krishi-sahayak-lora"
PERSIST_DIR = "../rag/chroma_store"

print("Loading retriever and model -- this can take a minute on first run...")
retriever = SchemeRetriever(persist_dir=PERSIST_DIR)
model, tokenizer = load_model(BASE_MODEL, ADAPTER_DIR)


def respond(message, history):
    context = retriever.format_context(message, top_k=3)
    answer = generate_answer(model, tokenizer, message, context)
    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="Krishi Sahayak — Farm Scheme Eligibility Assistant",
    description=(
        "Describe your land, crop, and situation, and I'll help you find the "
        "government schemes you may be eligible for. I'll ask follow-up "
        "questions if I need more details to give you an accurate answer."
    ),
    examples=[
        "I own 2 acres in Punjab and grow wheat, what can I get?",
        "I want crop insurance for my cotton farm in Maharashtra.",
        "I am a farmer, what schemes are there for me?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
