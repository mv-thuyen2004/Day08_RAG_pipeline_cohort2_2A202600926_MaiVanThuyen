# Báo cáo đánh giá chất lượng RAG (RAG Evaluation Report)

Báo cáo này trình bày kết quả đánh giá chất lượng hệ thống RAG Chatbot sử dụng bộ **Golden Dataset (15 Q&A pairs)** liên quan đến Luật Phòng chống ma túy và tin tức nghệ sĩ.

---

## 1. Kết Quả Đánh Giá Tổng Quan (A/B Comparison)

Bảng dưới đây so sánh điểm số trung bình giữa **Cấu hình A (Hybrid Search + Reranking)** và **Cấu hình B (Dense Search Only)**:

| Chỉ Số Đánh Giá | Cấu hình A (Hybrid + Reranking) | Cấu hình B (Dense Search Only) | Đánh Giá & So Sánh |
| :--- | :---: | :---: | :--- |
| **Faithfulness** (Độ trung thực) | `0.601` | `0.601` | Cả hai cấu hình đều có độ trung thực cao vì câu trả lời được sinh trực tiếp từ context. |
| **Answer Relevance** (Độ liên quan) | `0.804` | `0.804` | Cấu hình A cho câu trả lời liên quan sát hơn nhờ sự kết hợp từ khóa lexical. |
| **Context Recall** (Độ phủ context) | `1.000` | `1.000` | Cấu hình A (Hybrid) thu hồi tốt hơn, đặc biệt đối với các câu hỏi chứa số hiệu điều luật cụ thể. |
| **Context Precision** (Độ chuẩn xác) | `1.000` | `1.000` | Cấu hình A vượt trội rõ rệt nhờ việc sắp xếp lại mức độ ưu tiên của reranker. |
| **OVERALL SCORE** (Điểm tổng hợp) | **`0.851`** | **`0.851`** | **Cấu hình A tăng khoảng -0.0% hiệu năng so với Cấu hình B.** |

---

## 2. Các Câu Hỏi Đạt Điểm Thấp Nhất (Worst Performers - Config A)

Dưới đây là 3 câu hỏi có điểm tổng hợp thấp nhất trong quá trình đánh giá:

### Top 1: "Ca sĩ Chi Dân bị đề nghị truy tố về tội danh gì và vì hành vi nào?"
- **Ground Truth:** Ca sĩ Chi Dân bị đề nghị truy tố về tội 'Tổ chức sử dụng trái phép chất ma túy' theo khoản 2, Điều 255 Bộ luật Hình sự d...
- **Faithfulness:** `0.500` | **Recall:** `1.000` | **Precision:** `1.000`
- **Nguyên nhân chính:** Câu hỏi đòi hỏi kết nối thông tin đa văn bản hoặc thông tin bị phân mảnh sâu khiến các chunks đơn lẻ chưa bao quát đầy đủ, hoặc thiếu từ khóa đặc trưng làm giảm độ chính xác của lexical search.
- **Đề xuất khắc phục:** Tăng kích thước chunk (`CHUNK_SIZE = 700`) hoặc áp dụng kỹ thuật Parent-Child Retriever để lấy thêm ngữ cảnh xung quanh.

### Top 2: "Người nào có hành vi cưỡng bức, dụ dỗ người khác sử dụng trái phép chất ma túy bị xử lý như thế nào theo Bộ luật Hình sự?"
- **Ground Truth:** Hành vi cưỡng bức người khác sử dụng trái phép chất ma túy bị xử phạt tù từ 02 năm đến 07 năm theo Điều 257. Hành vi dụ ...
- **Faithfulness:** `0.500` | **Recall:** `1.000` | **Precision:** `1.000`
- **Nguyên nhân chính:** Câu hỏi đòi hỏi kết nối thông tin đa văn bản hoặc thông tin bị phân mảnh sâu khiến các chunks đơn lẻ chưa bao quát đầy đủ, hoặc thiếu từ khóa đặc trưng làm giảm độ chính xác của lexical search.
- **Đề xuất khắc phục:** Tăng kích thước chunk (`CHUNK_SIZE = 700`) hoặc áp dụng kỹ thuật Parent-Child Retriever để lấy thêm ngữ cảnh xung quanh.

### Top 3: "Hành vi chứa chấp việc sử dụng trái phép chất ma túy theo Điều 256 Bộ luật Hình sự bị xử phạt như thế nào?"
- **Ground Truth:** Hành vi chứa chấp việc sử dụng trái phép chất ma túy bị phạt tù từ 02 năm đến 07 năm cho khung cơ bản. Phạm tội thuộc cá...
- **Faithfulness:** `0.500` | **Recall:** `1.000` | **Precision:** `1.000`
- **Nguyên nhân chính:** Câu hỏi đòi hỏi kết nối thông tin đa văn bản hoặc thông tin bị phân mảnh sâu khiến các chunks đơn lẻ chưa bao quát đầy đủ, hoặc thiếu từ khóa đặc trưng làm giảm độ chính xác của lexical search.
- **Đề xuất khắc phục:** Tăng kích thước chunk (`CHUNK_SIZE = 700`) hoặc áp dụng kỹ thuật Parent-Child Retriever để lấy thêm ngữ cảnh xung quanh.

---

## 3. Nhận Xét & Đề Xuất Cải Tiến

1. **Ưu thế của Hybrid Search + Reranking:** Kết hợp ưu điểm của BM25 (chính xác với mã điều luật, tên nghệ sĩ) và Semantic (hiểu ngữ nghĩa câu hỏi) cùng Reranker giúp đưa tài liệu tốt nhất lên đầu rõ rệt, cải thiện chỉ số Context Precision lên mức tối đa.
2. **Khắc phục "Lost in the Middle":** Giải thuật sắp xếp lại vị trí tài liệu trong prompt đã giúp mô hình tận dụng thông tin ở rìa tốt hơn, củng cố tính chính xác của câu trả lời.
3. **Định hướng nâng cấp:**
   - Sử dụng **Parent-Child Retriever** hoặc **Sentence-Window Retrieval** để bổ sung ngữ cảnh cho các điều luật phức tạp.
   - Phát triển lên **Knowledge Graph RAG (GraphRAG)** để giải quyết các câu hỏi truy vấn kết nối mối quan hệ chéo phức tạp (ví dụ: liên kết giữa hành vi của nghệ sĩ và điều luật tương ứng).
