"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

# Configure UTF-8 encoding support for Windows terminal prints
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

VECTORSTORE_FILE = Path(__file__).parent.parent / "data" / "vectorstore.json"

_bm25 = None
_corpus = []


def get_bm25_index():
    global _bm25, _corpus
    if _bm25 is not None:
        return _bm25, _corpus

    if not VECTORSTORE_FILE.exists():
        print(f"[WARN] Corpus does not exist at {VECTORSTORE_FILE}. Please run task4 first.")
        return None, []

    try:
        _corpus = json.loads(VECTORSTORE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to load corpus: {e}")
        return None, []

    if not _corpus:
        return None, []

    # Tokenize corpus (simple split + lower)
    tokenized_corpus = [doc["content"].lower().split() for doc in _corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25, _corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25, corpus = get_bm25_index()
    if bm25 is None or not corpus:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx].get("metadata", {})
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
