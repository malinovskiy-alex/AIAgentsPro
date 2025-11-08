#!/usr/bin/env python3
"""
Hybrid RAG Demo з Reciprocal Rank Fusion (RRF)
===============================================

Демонструє комбінацію sparse (BM25) та dense (embeddings) retrieval методів
з використанням RRF для fusion результатів.

ВИПРАВЛЕНО: RRF bug - тепер всі scores різні та правильні!
"""

import time
from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simulated documents corpus
SAMPLE_DOCUMENTS = [
    "Python is a high-level programming language used for web development and data science.",
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
    "Neural networks are computing systems inspired by biological neural networks in animal brains.",
    "Deep learning uses multiple layers of neural networks to progressively extract higher-level features.",
    "Natural language processing helps computers understand, interpret and generate human language.",
    "TensorFlow is an open-source machine learning framework developed by Google.",
    "PyTorch is a popular deep learning framework known for its dynamic computational graphs.",
    "Data science combines statistics, mathematics, and computer science to extract insights from data.",
    "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence.",
    "Computer vision enables machines to interpret and understand visual information from the world."
]


class HybridRAG:
    """
    Hybrid RAG з правильним RRF алгоритмом

    Комбінує:
    - Sparse retrieval (TF-IDF/BM25): keyword-based
    - Dense retrieval (embeddings): semantic-based
    - RRF fusion: balanced ranking
    """

    def __init__(self, documents: List[str], alpha: float = 0.5, k: int = 60):
        """
        Initialize Hybrid RAG

        Parameters:
        - documents: corpus of documents
        - alpha: fusion weight (0=only sparse, 1=only dense)
        - k: RRF rank constant (default 60)
        """
        print("🚀 Ініціалізація Hybrid RAG з RRF...")
        self.documents = documents
        self.alpha = alpha
        self.k = k

        # Sparse retrieval (TF-IDF)
        print("📊 Створення TF-IDF індексу...")
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

        # Dense retrieval (простий word embeddings через TF-IDF як proxy)
        # У production: використайте sentence-transformers
        print("🎯 Створення dense embeddings...")
        self.dense_vectorizer = TfidfVectorizer(max_features=512)  # Симулюємо embeddings
        self.dense_matrix = self.dense_vectorizer.fit_transform(documents)

        print(f"✅ Готово! {len(documents)} документів проіндексовано")
        print(f"   Alpha: {alpha} (0=sparse, 1=dense)")
        print(f"   RRF k: {k}")

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Sparse retrieval (TF-IDF)

        Returns: List[(doc_idx, score)]
        """
        query_vector = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Сортувати за score (descending)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [(idx, scores[idx]) for idx in ranked_indices]
        return results

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Dense retrieval (embeddings)

        Returns: List[(doc_idx, score)]
        """
        query_vector = self.dense_vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.dense_matrix)[0]

        # Сортувати за score (descending)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [(idx, scores[idx]) for idx in ranked_indices]
        return results

    def reciprocal_rank_fusion(
        self,
        sparse_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        ✅ ПРАВИЛЬНИЙ RRF Алгоритм

        Formula: RRF_score = (1-α) * 1/(k+rank_sparse) + α * 1/(k+rank_dense)

        Parameters:
        - sparse_results: [(doc_idx, score)] від sparse search
        - dense_results: [(doc_idx, score)] від dense search

        Returns:
        - [(doc_idx, rrf_score)] sorted by RRF score (descending)
        """
        # Крок 1: Створити rank dictionaries
        sparse_ranks = {}
        dense_ranks = {}

        # ВАЖЛИВО: rank починається з 1, не з 0!
        for rank, (doc_idx, _) in enumerate(sparse_results, start=1):
            sparse_ranks[doc_idx] = rank

        for rank, (doc_idx, _) in enumerate(dense_results, start=1):
            dense_ranks[doc_idx] = rank

        # Крок 2: Знайти всі унікальні документи
        all_docs = set(sparse_ranks.keys()) | set(dense_ranks.keys())

        # Крок 3: Обчислити RRF scores
        rrf_scores = {}

        for doc_idx in all_docs:
            # Якщо документа немає в результатах, використати великий rank
            sparse_rank = sparse_ranks.get(doc_idx, len(sparse_results) + 100)
            dense_rank = dense_ranks.get(doc_idx, len(dense_results) + 100)

            # RRF formula з k константою
            sparse_score = 1.0 / (self.k + sparse_rank)
            dense_score = 1.0 / (self.k + dense_rank)

            # Weighted fusion
            rrf_score = (1 - self.alpha) * sparse_score + self.alpha * dense_score

            rrf_scores[doc_idx] = rrf_score

        # Крок 4: Сортувати за RRF score (descending)
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        show_details: bool = True
    ) -> Dict[str, Any]:
        """
        Hybrid search з RRF fusion

        Returns dictionary з results та debug info
        """
        start_time = time.time()

        print(f"\n🔍 Query: '{query}'")
        print("=" * 70)

        # Крок 1: Sparse search
        print("\n📊 Sparse Search (TF-IDF)...")
        sparse_start = time.time()
        sparse_results = self.sparse_search(query, top_k=10)
        sparse_time = time.time() - sparse_start
        print(f"   Знайдено: {len(sparse_results)} документів за {sparse_time*1000:.1f}ms")

        if show_details:
            print("   Top 3 sparse:")
            for i, (doc_idx, score) in enumerate(sparse_results[:3], 1):
                print(f"      {i}. Doc {doc_idx}: {score:.4f}")

        # Крок 2: Dense search
        print("\n🎯 Dense Search (Embeddings)...")
        dense_start = time.time()
        dense_results = self.dense_search(query, top_k=10)
        dense_time = time.time() - dense_start
        print(f"   Знайдено: {len(dense_results)} документів за {dense_time*1000:.1f}ms")

        if show_details:
            print("   Top 3 dense:")
            for i, (doc_idx, score) in enumerate(dense_results[:3], 1):
                print(f"      {i}. Doc {doc_idx}: {score:.4f}")

        # Крок 3: RRF Fusion
        print(f"\n🔀 RRF Fusion (α={self.alpha}, k={self.k})...")
        rrf_start = time.time()
        rrf_results = self.reciprocal_rank_fusion(sparse_results, dense_results)
        rrf_time = time.time() - rrf_start
        print(f"   Fusion завершено за {rrf_time*1000:.1f}ms")

        # Взяти top-k після fusion
        final_results = rrf_results[:top_k]

        total_time = time.time() - start_time

        # Результати
        print(f"\n✅ FINAL RESULTS (Top {top_k}):")
        print("=" * 70)
        for i, (doc_idx, rrf_score) in enumerate(final_results, 1):
            print(f"\n{i}. [RRF: {rrf_score:.6f}] Doc {doc_idx}:")
            print(f"   {self.documents[doc_idx][:80]}...")

            # Debug: показати sparse та dense ranks
            if show_details:
                sparse_rank = next((rank for rank, (idx, _) in enumerate(sparse_results, 1) if idx == doc_idx), "N/A")
                dense_rank = next((rank for rank, (idx, _) in enumerate(dense_results, 1) if idx == doc_idx), "N/A")
                print(f"   Sparse rank: {sparse_rank}, Dense rank: {dense_rank}")

        print(f"\n⏱️ Total time: {total_time*1000:.1f}ms")

        return {
            "query": query,
            "results": final_results,
            "sparse_time": sparse_time,
            "dense_time": dense_time,
            "rrf_time": rrf_time,
            "total_time": total_time,
            "alpha": self.alpha,
            "k": self.k
        }


def demo_hybrid_rag():
    """
    Демонстрація Hybrid RAG з різними α параметрами
    """
    print("\n" + "=" * 70)
    print("🔀 HYBRID RAG з RECIPROCAL RANK FUSION (RRF) - DEMO")
    print("=" * 70)
    print("\n✅ RRF BUG ВИПРАВЛЕНО!")
    print("   - Правильний ranking (1-indexed)")
    print("   - k=60 константа додана")
    print("   - Weighted fusion працює коректно")
    print("=" * 70)

    # Demo queries
    queries = [
        "machine learning frameworks",
        "how computers understand language",
        "Python programming"
    ]

    # Тестуємо різні α values
    alphas = [0.3, 0.5, 0.7]

    for alpha in alphas:
        print(f"\n\n{'='*70}")
        print(f"ALPHA = {alpha} ({'Favor Sparse' if alpha < 0.5 else 'Favor Dense' if alpha > 0.5 else 'Balanced'})")
        print("=" * 70)

        rag = HybridRAG(SAMPLE_DOCUMENTS, alpha=alpha, k=60)

        # Query 1
        result = rag.hybrid_search(queries[0], top_k=3, show_details=True)

        input("\n⏸️ Натисніть Enter для наступного alpha...")


def compare_alphas():
    """
    Порівняння різних α параметрів
    """
    print("\n" + "=" * 70)
    print("📊 ПОРІВНЯННЯ РІЗНИХ α ПАРАМЕТРІВ")
    print("=" * 70)

    query = "machine learning frameworks"

    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    print(f"\nQuery: '{query}'")
    print("\nRRF scores для різних α:")
    print("-" * 70)

    for alpha in alphas:
        rag = HybridRAG(SAMPLE_DOCUMENTS, alpha=alpha, k=60)
        result = rag.hybrid_search(query, top_k=3, show_details=False)

        label = "ONLY SPARSE" if alpha == 0.0 else \
                "ONLY DENSE" if alpha == 1.0 else \
                f"{int((1-alpha)*100)}% sparse, {int(alpha*100)}% dense"

        print(f"\nα = {alpha:.1f} ({label}):")
        for i, (doc_idx, score) in enumerate(result['results'], 1):
            print(f"  {i}. Doc {doc_idx}: RRF={score:.6f}")


def main():
    """Main demo function"""
    try:
        # Demo 1: Різні alpha values
        demo_hybrid_rag()

        # Demo 2: Порівняння
        compare_alphas()

    except KeyboardInterrupt:
        print("\n\n👋 Demo перервано")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("🎓 КЛЮЧОВІ TAKEAWAYS")
    print("=" * 70)
    print("""
1. RRF комбінує rankings з sparse (BM25) та dense (embeddings) методів
2. Formula: RRF = (1-α) × 1/(k+rank_sparse) + α × 1/(k+rank_dense)
3. Alpha parameter:
   - α=0.3: Favor keywords (technical docs, code)
   - α=0.5: Balanced (general use)
   - α=0.7: Favor semantic (natural language)
4. k=60: Standard constant від оригінальної RRF paper
5. Bug fixed: Rankings тепер 1-indexed, k додано, scores різні!

💡 Для production:
   - Tune α на вашому датасеті
   - Використайте real embeddings (sentence-transformers)
   - Implement parallel sparse+dense search
   - Add caching для frequent queries
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
