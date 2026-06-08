"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import json

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

CHUNK_SIZE = 500        # Chọn 500 ký tự (~100-150 từ) để lưu trữ các thông tin/điều luật trọn vẹn, không quá dài để tránh loãng thông tin và không quá ngắn để mất ngữ nghĩa.
CHUNK_OVERLAP = 50      # Chọn overlap 50 ký tự (~10-15 từ) để liên kết thông tin giữa các chunk liên kề, tránh mất ngữ cảnh khi bị cắt giữa chừng.
CHUNKING_METHOD = "recursive"  # Sử dụng RecursiveCharacterTextSplitter để phân đoạn văn bản dựa trên các ký tự ranh giới tự nhiên thụ động như xuống dòng, dấu chấm, khoảng trắng.

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lựa chọn model nhẹ (90MB), chạy cực nhanh trên CPU/local, rất ổn định và đáp ứng tốt nhu cầu tìm kiếm ngữ nghĩa cơ bản.
EMBEDDING_DIM = 384

VECTOR_STORE = "local_json"  # Sử dụng file JSON cục bộ làm Vector Store để đảm bảo hoạt động offline 100% không phụ thuộc Docker/Weaviate server.


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"[WARN] Standardized directory does not exist: {STANDARDIZED_DIR}")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file() and not md_file.name.startswith("."):
            content = md_file.read_text(encoding="utf-8")
            # Xác định type dựa trên đường dẫn tương đối
            rel_path = str(md_file.relative_to(STANDARDIZED_DIR))
            doc_type = "legal" if "legal" in rel_path else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    vectorstore_file = Path(__file__).parent.parent / "data" / "vectorstore.json"
    vectorstore_file.parent.mkdir(parents=True, exist_ok=True)
    vectorstore_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Chunks successfully indexed and saved to {vectorstore_file}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
