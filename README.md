# 🌾 Krishi Sahayak
### AI-Powered Farm Scheme Eligibility Assistant using RAG & Fine-Tuned LLM

Krishi Sahayak is an AI-powered assistant designed to help Indian farmers identify the government agricultural schemes they are eligible for. Instead of relying on static rule-based systems or generic chatbots, the project combines **Retrieval-Augmented Generation (RAG)** with a **QLoRA fine-tuned Qwen2.5-3B language model** to provide accurate, explainable, and context-aware recommendations.

The assistant retrieves the latest scheme information from a vector database and reasons over the farmer's profile before generating a response. When essential information is missing, the model intelligently asks follow-up questions instead of making incorrect assumptions.

---

## 🚀 Features

- 🌾 Government scheme eligibility prediction
- 🤖 Fine-tuned Qwen2.5-3B using QLoRA
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 ChromaDB vector database
- 🌐 Multilingual sentence embeddings
- 📄 PDF/knowledge retrieval support
- 💬 Intelligent clarification questions
- 📊 Evaluation pipeline
- 🎯 Gradio-based interactive chatbot
- ⚡ Google Colab compatible

---

## 🛠 Tech Stack

### AI & Machine Learning
- Python
- Hugging Face Transformers
- PEFT (QLoRA)
- BitsAndBytes
- Sentence Transformers

### RAG
- ChromaDB
- Cross Encoder Reranker
- Multilingual Embeddings

### LLM
- Qwen2.5-3B
- LoRA Fine-Tuning

### Frontend
- Gradio

### Data
- JSON Knowledge Base
- Synthetic Instruction Dataset

---

## 📂 Project Structure

```
krishi-sahayak/
│
├── app/
│   └── app.py
│
├── rag/
│   ├── build_index.py
│   └── retriever.py
│
├── finetune/
│   ├── train_qlora.py
│   └── inference.py
│
├── data/
│   ├── schemes.json
│   └── generate_synthetic_data.py
│
├── eval/
│   ├── evaluate.py
│   └── test_profiles.json
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🏗 System Architecture

```
              Farmer Query
                    │
                    ▼
        ┌────────────────────┐
        │   User Profile      │
        └─────────┬───────────┘
                  │
                  ▼
        ┌────────────────────┐
        │   RAG Retriever     │
        │ (Chroma + Embedding)│
        └─────────┬───────────┘
                  │
          Relevant Documents
                  │
                  ▼
      ┌────────────────────────┐
      │ Fine-Tuned Qwen2.5-3B   │
      │      (QLoRA)            │
      └─────────┬───────────────┘
                │
                ▼
      Government Scheme Advice
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/giri164/krishi-sahayak.git
```

Go into the project

```bash
cd krishi-sahayak
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

### Build Vector Database

```bash
cd rag
python build_index.py
```

### Generate Training Dataset

```bash
cd data
python generate_synthetic_data.py
```

### Fine-Tune the Model

```bash
cd finetune
python train_qlora.py
```

### Run Inference

```bash
python inference.py
```

### Launch Chatbot

```bash
python app/app.py
```

---

# 📊 Evaluation

The project includes an evaluation pipeline for comparing:

- Base LLM
- RAG only
- RAG + Fine-Tuned LLM

Metrics include:

- Accuracy
- Correct Scheme Retrieval
- Eligibility Prediction
- Clarification Question Accuracy
- Response Quality

Run:

```bash
python eval/evaluate.py
```

---

# 💡 Example Query

**Input**

```
I own 2 acres of land in Andhra Pradesh and cultivate paddy.
Which government schemes am I eligible for?
```

**Output**

```
Eligible Schemes

• PM-KISAN
• PMFBY
• Kisan Credit Card

Required Documents

• Aadhaar
• Land Records
• Bank Account
• Mobile Number
```

---

# 🎯 Applications

- Smart Agriculture
- Government Welfare Advisory
- Farmer Assistance
- Rural Digital Services
- Agricultural Extension Support
- AI for Social Good

---

# Future Improvements

- Voice-based interaction
- Mobile application
- OCR for government documents
- Real-time scheme updates
- Regional language support
- WhatsApp integration
- District-wise recommendations
- Multi-agent architecture

---

# 📚 Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Large Language Model Fine-Tuning
- LoRA & QLoRA
- Vector Databases
- Prompt Engineering
- Semantic Search
- Hugging Face Ecosystem
- AI Application Development

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. It helps others discover the project and supports future development.
