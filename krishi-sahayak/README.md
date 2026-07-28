# Krishi Sahayak — RAG + Fine-Tuned Open LLM for Farm Scheme Eligibility

Krishi Sahayak ("farmer's helper") is an assistant that helps Indian farmers
and rural field officers figure out which government agricultural schemes
(PM-KISAN, PMFBY, KCC, soil health card, irrigation subsidies, FPO support,
etc.) they are actually eligible for — instead of guessing or missing out
because the rules are scattered across dozens of PDFs.

## Why this project

General-purpose LLMs hallucinate scheme details and confidently answer even
when they don't have enough information about the farmer's situation. This
project combines two techniques deliberately, each for a different job:

- **RAG** supplies up-to-date, grounded scheme facts (eligibility rules,
  benefits, documents, application steps) so the model isn't relying on
  stale or memorized details.
- **QLoRA fine-tuning** on a small open LLM teaches the *reasoning
  behavior*: how to match a farmer's profile against eligibility criteria,
  when to cite a scheme confidently, and — critically — **when to ask a
  clarifying question instead of guessing**, or explain why someone is
  *not* eligible.

That last point is the core contribution: most scheme-advisory bots answer
confidently regardless of how much information they were given. This system
is explicitly trained to recognize incomplete profiles and ask before
recommending.

## Architecture

```
Farmer query
     │
     ▼
┌─────────────┐      top-k relevant scheme chunks
│  Retriever   │ ───────────────────────────────┐
│ (Chroma +    │                                 │
│  multilingual│                                 ▼
│  embeddings +│                    ┌───────────────────────┐
│  reranker)   │                    │  Fine-tuned open LLM    │
└─────────────┘                    │  (Qwen2.5-3B + QLoRA)   │
                                    │  - grounded answer, OR   │
                                    │  - clarifying question,  │
                                    │  - OR ineligibility note │
                                    └───────────────────────┘
                                                 │
                                                 ▼
                                        Answer to farmer
```

## Repository structure

```
krishi-sahayak/
├── data/
│   ├── schemes.json                # Scheme knowledge base (12 sample schemes)
│   └── generate_synthetic_data.py  # Builds instruction-tuning data from schemes.json
├── rag/
│   ├── build_index.py              # Embeds schemes.json into a Chroma vector store
│   └── retriever.py                # Query-time retrieval + cross-encoder reranking
├── finetune/
│   ├── train_qlora.py              # QLoRA fine-tuning (Colab T4-friendly)
│   └── inference.py                # RAG + fine-tuned model inference
├── eval/
│   ├── test_profiles.json          # 10 test cases with known correct answers
│   └── evaluate.py                 # Compares base vs RAG vs RAG+fine-tuned
├── app/
│   └── app.py                      # Gradio chat demo
├── Krishi_Sahayak_Colab.ipynb      # One-click notebook running the full pipeline
├── requirements.txt
└── README.md
```

## Quickstart (Google Colab, free tier)

1. Open `Krishi_Sahayak_Colab.ipynb` in Colab.
2. Set runtime to **T4 GPU** (`Runtime > Change runtime type`).
3. Run cells top to bottom — it clones the repo, builds the RAG index,
   generates training data, fine-tunes with QLoRA, runs inference, and
   evaluates.

## Quickstart (local)

```bash
git clone https://github.com/YOUR_USERNAME/krishi-sahayak.git
cd krishi-sahayak
pip install -r requirements.txt

# 1. Build the vector index
cd rag && python build_index.py && cd ..

# 2. Generate fine-tuning data
cd data && python generate_synthetic_data.py --n 400 && cd ..

# 3. Fine-tune (needs a CUDA GPU)
cd finetune && python train_qlora.py && cd ..

# 4. Try it
python finetune/inference.py --query "I own 2 acres in Punjab and grow wheat, is there insurance for me?"

# 5. Launch the demo
cd app && python app.py
```

## Evaluation methodology

`eval/evaluate.py` runs the same 10 test profiles (`eval/test_profiles.json`)
through three configurations and reports:

| Metric | What it measures |
|---|---|
| Scheme match accuracy | Did the model recommend the correct scheme when the profile was complete? |
| Clarification recall | Did the model ask for more info when the profile was vague, instead of guessing? |
| False recommendation rate | Did the model wrongly recommend a scheme to a profile that fails an exclusion rule? |

Run all three and compare:

```bash
cd eval
python evaluate.py --config base            # no RAG, no fine-tuning
python evaluate.py --config rag              # RAG + base model
python evaluate.py --config rag_finetuned    # RAG + fine-tuned model
```

The expected result (and the point of the project) is that the base model
scores poorly on clarification recall and false-recommendation rate even
with RAG context available, while the fine-tuned version improves
specifically on those two behavioral metrics — demonstrating that
fine-tuning added reasoning behavior that retrieval alone couldn't.

## Data notes

`data/schemes.json` contains 12 real central/state scheme templates as a
starting point. **Before using this for anything beyond a demo, verify and
expand this against official sources** (e.g. [myscheme.gov.in](https://www.myscheme.gov.in/),
individual scheme portals, and current state government notifications), as
eligibility rules, benefit amounts, and cutoffs change over time.

## Models used

- Base LLM: `Qwen/Qwen2.5-3B-Instruct` (swap for `meta-llama/Llama-3.2-3B-Instruct`
  or `microsoft/Phi-3.5-mini-instruct` in the `--base_model` flag — all fit
  a free-tier T4 GPU in 4-bit).
- Embedding model: `intfloat/multilingual-e5-small` (chosen for future
  Hindi/regional-language query support).
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Possible extensions

- Add more schemes and state-specific variants to `schemes.json`.
- Support Hindi/regional-language queries end-to-end (the embedding model
  already supports this; the LLM and training data would need extending).
- Add a confidence/routing layer that decides between trusting fine-tuned
  knowledge vs. retrieved context vs. abstaining, and evaluate it
  separately.
- Replace the synthetic training data with real anonymized query logs from
  a pilot deployment.

## License

MIT — see [LICENSE](LICENSE).
