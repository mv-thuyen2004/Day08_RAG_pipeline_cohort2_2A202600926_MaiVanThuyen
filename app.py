import os
import time
from pathlib import Path
import streamlit as st
import numpy as np
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Setup paths and import modules
import sys
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    from src.task9_retrieval_pipeline import retrieve
except ImportError as e:
    st.error(f"Failed to import retrieval pipeline: {e}")
    sys.exit(1)

# Set page config
st.set_page_config(
    page_title="RAG Law Chatbot - Drug Prevention",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism Look (Dark Mode theme)
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header styling */
    .header-container {
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-radius: 12px;
        border: 1px solid #312e81;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .header-title {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Sidebar Glassmorphic design */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e293b;
    }
    
    /* Chat layout styling */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .user-bubble {
        background-color: #3b82f6;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    .assistant-bubble {
        background-color: #1e293b;
        color: #e2e8f0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom tags */
    .pill-tag {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 0.5rem;
        text-transform: uppercase;
    }
    .tag-legal {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #047857;
    }
    .tag-news {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #1d4ed8;
    }
    .score-tag {
        background-color: #312e81;
        color: #c084fc;
        border: 1px solid #4c1d95;
    }

    /* Source Inspector styling */
    .source-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .5; }
    }
    .typing-indicator {
        animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        color: #94a3b8;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def summarize_chunks_offline(chunks: list[dict], query: str) -> str:
    """Offline Fallback summary generator."""
    if not chunks:
        return "Tôi không thể tìm thấy thông tin phù hợp trong cơ sở dữ liệu hiện có để trả lời câu hỏi này."
    
    best_chunk = chunks[0]
    source_name = best_chunk.get("metadata", {}).get("source", "Tài liệu hệ thống")
    doc_type = best_chunk.get("metadata", {}).get("type", "legal")
    
    answer = (
        f"**[Kết quả tìm kiếm local]** Dựa trên tài liệu trích dẫn từ **{source_name}** ({'Văn bản pháp luật' if doc_type == 'legal' else 'Tin tức'}), "
        f"tôi xin tóm tắt câu trả lời cho câu hỏi '{query}':\n\n"
        f"{best_chunk['content'].strip()}\n\n"
        f"*Lưu ý: Câu trả lời được trích xuất trực tiếp từ tài liệu offline vì OpenAI API key chưa được cấu hình hoặc không khả dụng.*"
    )
    return answer


def generate_rag_answer(
    query: str,
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
    history: list = None
) -> dict:
    """Custom RAG Generation Pipeline with parameters from Sidebar."""
    # 1. Rewrite query if there is conversation history to support follow-up questions
    if history and len(history) > 0:
        history_text = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history[-2:]])
        contextual_query = f"Câu hỏi trước đó:\n{history_text}\n\nHỏi mới: {query}"
    else:
        contextual_query = query

    # 2. Retrieve relevant chunks
    start_time = time.time()
    chunks = retrieve(
        contextual_query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking
    )
    latency = time.time() - start_time

    # 3. Reorder context for LLM (mitigate lost in the middle)
    from src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    # 4. Generate answer (Gemini or OpenAI)
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key and not gemini_key.startswith("gemini_") and not gemini_key.startswith("xxx"):
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=gemini_key)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
            
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            response = client.models.generate_content(
                model=gemini_model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    top_p=0.9,
                ),
            )
            answer = response.text
        except Exception as e:
            answer = f"**[Lỗi gọi Gemini API: {e}]**\n\n" + summarize_chunks_offline(chunks, query)
    elif openai_key and not openai_key.startswith("sk-xxx") and not openai_key.startswith("xxx"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                top_p=0.9,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"**[Lỗi gọi OpenAI API: {e}]**\n\n" + summarize_chunks_offline(chunks, query)
    else:
        answer = summarize_chunks_offline(chunks, query)

    return {
        "answer": answer,
        "sources": chunks,
        "latency": latency,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


# =============================================================================
# STREAMLIT UI LAYOUT
# =============================================================================

# Title Block
st.markdown("""
<div class="header-container">
    <h1 class="header-title">⚖️ CHATBOT HỖ TRỢ PHÁP LUẬT MA TÚY</h1>
    <p class="header-subtitle">Hệ thống hỏi đáp Luật Phòng chống ma tuý & Tin tức vụ việc liên quan nghệ sĩ sử dụng ma tuý</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    # A list of dictionaries: {"query": str, "answer": str}
    st.session_state.chat_history = []
if "last_rag_result" not in st.session_state:
    st.session_state.last_rag_result = None

# ================= SIDEBAR CONFIGURATION =================
st.sidebar.markdown("### ⚙️ Cấu Hình RAG Pipeline")

top_k = st.sidebar.slider(
    "Số lượng tài liệu trích dẫn (Top K)",
    min_value=1,
    max_value=10,
    value=5,
    help="Số lượng chunks văn bản tối đa gửi vào prompt của LLM."
)

score_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy (Score Threshold)",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
    help="Điểm tối thiểu của Hybrid Search để không kích hoạt PageIndex fallback."
)

use_reranking = st.sidebar.toggle(
    "Sử dụng Reranking (Task 7)",
    value=True,
    help="Sắp xếp lại tài liệu sử dụng mô hình Cross-Encoder để ưu tiên thông tin chính xác nhất."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Thông tin chỉ mục hiện tại")
# Load corpus info from vectorstore
vectorstore_path = PROJECT_DIR / "data" / "vectorstore.json"
if vectorstore_path.exists():
    import json
    try:
        with open(vectorstore_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        st.sidebar.info(f"Tổng số chunks đã index: **{len(corpus_data)}**")
        
        # Count types
        types = [c.get("metadata", {}).get("type", "unknown") for c in corpus_data]
        st.sidebar.markdown(f"- ⚖️ Chunks Pháp luật: **{types.count('legal')}**")
        st.sidebar.markdown(f"- 📰 Chunks Tin tức: **{types.count('news')}**")
    except Exception:
        st.sidebar.warning("Không đọc được dữ liệu Vector Store.")
else:
    st.sidebar.error("Không tìm thấy file Vector Store. Hãy chạy lại Task 4.")

# Clear Chat button
if st.sidebar.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.last_rag_result = None
    st.rerun()


# ================= MAIN PAGE LAYOUT =================
# Split screen into Chat Column and Source Inspector Column
col_chat, col_inspector = st.columns([1.4, 1.0], gap="large")

with col_chat:
    st.markdown("### 💬 Trò Chuyện Trực Tuyến")
    
    # Render chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            
    # Chat Input
    if user_query := st.chat_input("Hãy hỏi về luật phòng chống ma túy hoặc các vụ án nghệ sĩ..."):
        # Display user bubble instantly
        st.markdown(f'<div class="chat-bubble user-bubble">{user_query}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display typing indicator
        with st.empty():
            st.markdown('<div class="typing-indicator">🤖 Đang truy vấn tài liệu và suy nghĩ...</div>', unsafe_allow_html=True)
            
            # Execute RAG pipeline
            result = generate_rag_answer(
                query=user_query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
                history=st.session_state.chat_history
            )
            
        # Clean typing indicator and show response
        st.markdown(f'<div class="chat-bubble assistant-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        
        # Save to session memory
        st.session_state.chat_history.append({"query": user_query, "answer": result["answer"]})
        st.session_state.last_rag_result = result
        st.rerun()

with col_inspector:
    st.markdown("### 🔎 Nguồn Trích Dẫn Chi Tiết (Source Inspector)")
    
    # Render source documents from last RAG query
    result = st.session_state.last_rag_result
    if result and result.get("sources"):
        st.markdown(f"⏱️ Thời gian truy xuất: `{result['latency']:.3f}s` | Nguồn: `{result['retrieval_source'].upper()}`")
        
        for i, src in enumerate(result["sources"], 1):
            metadata = src.get("metadata", {})
            source_file = metadata.get("source", "Tài liệu")
            doc_type = metadata.get("type", "legal")
            score = src.get("score", 0.0)
            
            # Select tag class based on doc type
            tag_class = "tag-legal" if doc_type == "legal" else "tag-news"
            type_label = "⚖️ PHÁP LUẬT" if doc_type == "legal" else "📰 TIN TỨC"
            
            with st.container():
                st.markdown(f"""
                <div class="source-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>#{i} {source_file}</strong>
                        <div>
                            <span class="pill-tag {tag_class}">{type_label}</span>
                            <span class="pill-tag score-tag">Score: {score:.3f}</span>
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; white-space: pre-wrap; background-color: #020617; padding: 0.5rem; border-radius: 6px; border: 1px solid #1e293b;">
{src['content'].strip()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Chưa có truy vấn nào được thực hiện hoặc không tìm thấy tài liệu phù hợp.")
        st.markdown("""
        **Hướng dẫn đặt câu hỏi mẫu:**
        1. *Luật:* "Các hình thức cai nghiện ma tuý tự nguyện?"
        2. *Hình phạt:* "Tội tàng trữ trái phép chất ma tuý bị phạt tù như thế nào?"
        3. *Tin tức:* "Vụ án ca sĩ Chi Dân tổ chức sử dụng ma tuý có diễn biến ra sao?"
        """)
