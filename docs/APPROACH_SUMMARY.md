# Tóm tắt Cách Làm & Hướng Triển Khai

Bản tóm tắt ngắn gọn — chi tiết đầy đủ xem [EVALUATION_JUDGE_SPEC.md](EVALUATION_JUDGE_SPEC.md) và [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## 1. Ý tưởng cốt lõi

Chấm câu trả lời của bot theo thứ tự ưu tiên nguồn tham chiếu, từ chắc chắn nhất đến ít chắc chắn nhất:

**Rule Riêng (Expected_resp trong file test) → Rule Chung (RAG vector DB) → Web search → Đánh giá mở**

Chỉ rơi xuống nguồn thấp hơn khi nguồn cao hơn không có. Điểm số dựa trên **độ trùng từ vựng có trọng số** (không phải AI hiểu ngữ nghĩa) giữa câu trả lời và nguồn tham chiếu, cộng thêm 1 lớp AI (LLM Judge) xử lý riêng cho các case điểm mập mờ.

## 2. Cách chấm điểm (compute_similarity)

- Đo: bao nhiêu % từ trong câu trả lời của bot thực sự xuất hiện trong tài liệu tham chiếu (precision), có ngưỡng sàn số từ trùng tối thiểu để tránh câu trả lời chung chung "ăn may" điểm cao.
- Không đo full ngữ nghĩa — đây là giới hạn đã biết: câu trả lời diễn giải đúng ý nhưng khác từ vẫn có thể bị điểm thấp.

## 3. Agent Loop — vòng lặp tự sửa (khi điểm mập mờ 25-45%)

2 bước cố định, dừng ngay khi 1 bước thành công:
1. Tìm lại trong RAG với từ khóa mở rộng
2. Nếu vẫn không đạt, tìm trên web (timeout 2s)

Không tìm được → giữ nguyên RETEST, **không bao giờ tự ý FAIL**.

## 4. LLM Judge — lớp AI cho case RETEST còn lại

1 lệnh gọi LLM local (Ollama qwen2.5:3b, miễn phí, không qua OpenAI) đọc câu hỏi + câu trả lời + tài liệu, tự phán PASS/FAIL. Nếu lỗi/timeout/không parse được → giữ nguyên RETEST, không đoán bừa.

*(Đã thử thiết kế multi-agent debate (2 AI tranh luận), nhưng kiểm chứng cho thấy kém chính xác hơn và chậm hơn 5-6 lần so với 1 AI phán trực tiếp — nên dùng bản đơn giản.)*

## 5. Phân loại mức độ nghiêm trọng

| Mức | % sai | Ý nghĩa |
|---|---|---|
| ✅ PASS | 0-15% | Đúng, đầy đủ |
| ℹ️ LOW | 15-30% | Sai nhỏ về diễn đạt |
| ⚠️ MEDIUM | 30-60% | Thiếu chi tiết quan trọng |
| 🚨 HIGH | 60-100% | Sai nghiêm trọng / bịa thông tin / từ chối oan |

## 6. Trace ngược nguyên nhân

Mọi case đều có `trace_id` + log đầy đủ: chunk/nguồn nào đã dùng để quyết định, lỗi mạng nếu có, từ khóa nào bị thiếu. Xem qua dashboard (click vào dòng kết quả) hoặc API JSON — **chưa** xuất đầy đủ vào file Excel.

## 7. Tốc độ

Phần lớn case chấm gần như tức thời (<100ms). Case cần LLM Judge mất ~4.6s/case. Batch ~1.800 dòng test: **~5 phút** kể cả phần Judge.

## 8. Hướng triển khai tiếp theo (chưa làm / còn tồn đọng)

- Batch chạy qua dashboard web chưa chạy song song 16 luồng như CLI — cần đồng bộ.
- Chưa có lớp kiểm tra số liệu/thực thể cụ thể (số kWh, %, tên nút bấm) — hiện tại câu trả lời sai 1 con số cụ thể nhưng giữ khung câu chung vẫn có thể lọt PASS.
- File test log dạng lạ (không đúng tên quy ước) có thể lọt vào vector DB gây nhiễu kết quả RAG.
- Trace log chi tiết chưa xuất ra Excel, chỉ xem được qua dashboard/API.
- Agent Benchmark (108 case) là dữ liệu giả lập, cần bổ sung benchmark trên dữ liệu thật định kỳ để theo dõi độ chính xác thực tế.
