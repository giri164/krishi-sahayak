"""
QLoRA fine-tuning for Krishi Sahayak on a small open LLM.

Designed to run on a free-tier Colab GPU (T4, ~15GB VRAM). Uses 4-bit
quantization + LoRA adapters so only a small fraction of parameters are
trained and memory footprint stays low.

Default base model: Qwen2.5-3B-Instruct (swap for Llama-3.2-3B-Instruct or
Phi-3.5-mini-instruct if preferred -- all fit comfortably on a T4 in 4-bit).

Usage:
    python train_qlora.py --train_file ../data/train_data.jsonl --output_dir ./krishi-sahayak-lora
"""

import argparse
import json

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_chat(example, tokenizer):
    """Apply the model's chat template to the messages list."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_file", type=str, default="../data/train_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./krishi-sahayak-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading base model in 4-bit: {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training data from {args.train_file}")
    raw_examples = load_jsonl(args.train_file)
    dataset = Dataset.from_list(raw_examples)
    dataset = dataset.map(lambda ex: format_chat(ex, tokenizer))

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
