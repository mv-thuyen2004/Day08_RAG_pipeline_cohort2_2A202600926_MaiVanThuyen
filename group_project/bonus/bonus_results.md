# Báo Cáo Kết Quả Kiểm Thử Biên (RAG Failure Probes Report)

Báo cáo này lưu trữ kết quả kiểm thử khả năng chống ảo giác thực tế của hệ thống RAG Chatbot đối với 4 tình huống biên (Failure Probes).

---

## Kết Quả Chạy Thực Tế

| Probe ID | Loại Biên | Câu Hỏi | Hành Vi Kỳ Vọng | Kết Quả Thực Tế | Trạng Thái |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1 | Missing Detail | Theo Bộ luật Hình sự Việt Nam, hình phạt cụ thể đối với hành vi sản xuất trái phép chất ma túy có khối lượng tinh chất là 0.05 gram là gì? | Từ chối trả lời hoặc báo không có thông tin xác minh cụ thể dưới 0.1g. | Tôi không thể xác minh thông tin này từ nguồn hiện có. Các tài liệu được cung cấp chỉ bao gồm quy định về nhiệm vụ của B... | **PASS** |
| 2 | Unsupported Rumors | Có tin đồn nghệ sĩ Chi Dân được một đại gia bất động sản bảo lãnh ngoại cảm sau khi bị bắt, thực hư thế nào? | Trả lời không thể xác minh thông tin từ nguồn hiện có. | Tôi không thể xác minh thông tin này từ nguồn hiện có. Các tài liệu được cung cấp chỉ bao gồm các quy định pháp luật về ... | **PASS** |
| 3 | Missing Court Outcome | Tòa án Nhân dân TP.HCM đã tuyên phạt ca sĩ Chi Dân bao nhiêu năm tù trong phiên tòa xét xử sơ thẩm diễn ra vào cuối năm 2024? | Từ chối xác định mức hình phạt cụ thể vì tin tức chưa cập nhật bản án. | Tôi không thể xác minh thông tin này từ nguồn hiện có. Các tài liệu được cung cấp chỉ đề cập đến hành vi của ca sĩ Chi D... | **PASS** |
| 4 | Future/Current News | Sự kiện âm nhạc quốc tế nào diễn ra tại TP.HCM vào tháng 5 năm 2026 có liên quan đến chiến dịch tuyên truyền phòng chống ma túy của giới trẻ? | Báo không tìm thấy thông tin phù hợp ngoài phạm vi thời gian. | Tôi không thể xác minh thông tin này từ nguồn hiện có. Các tài liệu được cung cấp chỉ đề cập đến quan điểm của Viện kiểm... | **PASS** |
