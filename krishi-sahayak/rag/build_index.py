"""
Builds a local vector index over the scheme knowledge base.

Uses a small multilingual embedding model (good for Hindi/regional-language
queries later) and Chroma as the vector store, both lightweight enough for a
free-tier Colab session.

Usage:
    python build_index.py --schemes ../data/schemes.json --persist_dir ./chroma_store
"""

import json
import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def load_schemes(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scheme_to_document(scheme: dict) -> str:
    """Flatten a scheme record into a single retrievable text chunk.

    Chunking is done per-scheme (not by fixed token windows) because
    eligibility clauses lose meaning if split mid-rule.
    """
    elig = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in scheme["eligibility"].items())
    return (
        f"Scheme name: {scheme['name']}\n"
        f"Category: {scheme['category']}\n"
        f"Level: {scheme['level']}\n"
        f"Eligibility: {elig}\n"
        f"Benefit: {scheme['benefit']}\n"
        f"Documents required: {', '.join(scheme['documents_required'])}\n"
        f"Application process: {scheme['application_process']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", type=str, default="../data/schemes.json")
    parser.add_argument("--persist_dir", type=str, default="./chroma_store")
    parser.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-small")
    args = parser.parse_args()

    schemes = load_schemes(args.schemes)
    print(f"Loaded {len(schemes)} schemes.")

    print(f"Loading embedding model: {args.embed_model} (small enough for CPU/free Colab)")
    model = SentenceTransformer(args.embed_model)

    documents = [scheme_to_document(s) for s in schemes]
    # e5 models expect a "passage: " prefix for documents
    embed_inputs = [f"passage: {d}" for d in documents]
    embeddings = model.encode(embed_inputs, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_or_create_collection(
        name="agri_schemes", metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[s["id"] for s in schemes],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[{"name": s["name"], "category": s["category"], "level": s["level"]} for s in schemes],
    )

    print(f"Indexed {collection.count()} schemes into '{args.persist_dir}'")


if __name__ == "__main__":
    main()
