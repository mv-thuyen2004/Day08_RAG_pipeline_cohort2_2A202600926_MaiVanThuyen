"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import json
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

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
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not VECTORSTORE_FILE.exists():
        print(f"[WARN] Vector store does not exist at {VECTORSTORE_FILE}. Please run task4 first.")
        return []

    # Load vector store
    try:
        chunks = json.loads(VECTORSTORE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to load vector store: {e}")
        return []

    if not chunks:
        return []

    # Load model and embed query
    model = get_embedding_model()
    query_vector = model.encode(query)
    query_vector = np.array(query_vector)

    query_norm = np.linalg.norm(query_vector)

    results = []
    for chunk in chunks:
        chunk_vector = np.array(chunk["embedding"])
        chunk_norm = np.linalg.norm(chunk_vector)
        
        if query_norm == 0 or chunk_norm == 0:
            score = 0.0
        else:
            score = float(np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm))

        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": chunk.get("metadata", {})
        })

    # Sắp xếp giảm dần theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
