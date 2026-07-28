"""
Retriever for Krishi Sahayak.

Given a farmer query, retrieves the top-k most relevant scheme documents
from the Chroma index, then optionally re-ranks with a cross-encoder for
better precision (precision matters more than recall here -- recommending
an irrelevant scheme erodes trust fast).

Usage (as a library):
    from retriever import SchemeRetriever
    r = SchemeRetriever(persist_dir="./chroma_store")
    results = r.retrieve("I own 2 acres in Punjab and grow wheat", top_k=3)
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


class SchemeRetriever:
    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        embed_model: str = "intfloat/multilingual-e5-small",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranker: bool = True,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("agri_schemes")
        self.embed_model = SentenceTransformer(embed_model)
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, top_k: int = 3, fetch_k: int = 8):
        query_embedding = self.embed_model.encode(
            [f"query: {query}"], normalize_embeddings=True
        )[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_k, self.collection.count()),
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not self.use_reranker or len(docs) <= top_k:
            return list(zip(docs, metas))[:top_k]

        pairs = [(query, d) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
        return [(d, m) for d, m, _ in ranked[:top_k]]

    def format_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)
        return "\n\n".join(doc for doc, _ in results)


if __name__ == "__main__":
    retriever = SchemeRetriever()
    test_query = "I own 2 acres in Punjab and grow wheat, I want crop insurance"
    print(retriever.format_context(test_query))
