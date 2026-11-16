# Benchmark Day 7 — Bộ luật Lao động 2012 & 2019

File sinh ra: `data/day7/questions.json` (26 câu), `data/day7/cases.json`. Tạo lại: `python -m src.day7.build_data --day6 <repo Day 6>`.

| Loại câu | Dev | Test |
|---|---:|---:|
| Có đáp án (từ benchmark Day 6, q01–q20) | 15 | 5 |
| Thiếu bằng chứng (mới, `unanswerable.json`, u01–u06) | 3 | 3 |
| **Tổng** | **18** | **8** |

- **Câu có đáp án**: giữ nguyên 20 câu Day 6 (`gold` = Điều gốc + Điều tương đương của bản luật kia khi nội dung giống hệt). `build_data` kiểm tra mọi evidence nằm trọn trong một chunk đúng (văn bản, Điều) của index Day 6. Day 6 chỉ chạy test **một lần** sau khi chốt, không dùng để chỉnh; Day 7 giữ nguyên split và không xem test trước khi chốt cấu hình.
- **Câu thiếu bằng chứng**: soạn gần chủ đề (lương tối thiểu, BHXH, thai sản lao động nam, mức phạt, làm việc từ xa, thuế TNCN). Mỗi câu ghi `absence_reason` và `check` (tìm chuỗi trên toàn bộ `data/raw` Day 6 + đọc Điều liên quan), không kết luận chỉ từ top-k. Các câu này không vào mẫu số retrieval metrics.
- **Filter**: chỉ có filter `year`, suy từ câu hỏi bằng quy tắc cố định (`common.infer_year_filter`: câu hỏi ghi đúng một năm 2012/2019). `build_data` xác nhận kết quả trùng với filter người dùng của Day 6 trên toàn bộ câu có đáp án. Gold không bao giờ được đưa vào retrieval hay agent.
- **Hit**: chunk thuộc (văn bản, Điều) gold **và** chứa trọn evidence sau chuẩn hóa NFC/lowercase/khoảng trắng (giống Day 6).
- Người soạn: Phan Trần Mạnh Cường. Câu thiếu bằng chứng được kiểm tra bằng script tìm chuỗi trên toàn corpus và đọc đối chiếu các Điều liên quan.

## Tiêu chí chọn cấu hình — chốt 2026-11-16, trước khi chạy agent trên dev

**Retrieval (tool của agent)**, trên dev, theo thứ tự: (1) Context Evidence Rate, (2) Evidence Hit@3, (3) Hit@1, (4) median ms/query.

**Agent**, trên dev, theo thứ tự:
1. Grounded answer trung bình (/2) trên câu có đáp án;
2. Abstention đúng trên câu thiếu evidence (không được thấp hơn baseline);
3. Citation hợp lệ (có trong context **và** hỗ trợ câu trả lời);
4. Median thời gian/câu (chênh < 2× được coi là chấp nhận được). Chi phí API = $0 với mọi cấu hình (model local).

Sau khi chọn: chạy **một lần** baseline (A1) và cấu hình chọn trên test, báo nguyên trạng; không chỉnh theo test.
