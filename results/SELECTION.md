# Chốt cấu hình — 2026-11-16, trước khi chạy test

Áp dụng tiêu chí trong [benchmark/README.md](../benchmark/README.md) (chốt trước khi chạy agent dev) lên kết quả dev đã chấm tay:

| Thứ tự tiêu chí | A1 (baseline) | A2 | **A3** | A4 |
|---|---|---|---|---|
| 1. Grounded TB /2 | 1.67 | 1.47 | **1.93** | 1.80 |
| 2. Abstention đúng (≥ baseline) | 2/3 | 2/3 | 2/3 ✓ | 2/3 |
| 3. Citation hợp lệ | 15/15 | 12/14 | 19/20 | 23/25 |
| 4. Median s/câu | 12.0 | 13.2 | 8.0 | 10.9 |

**Chọn A3**: tool `search_law` = hybrid RRF (dense pgvector + BM25 bigram, k=60, depth 30) + filter năm từ câu hỏi; top-k 3; lượt tìm đầu dùng câu hỏi gốc; mở rộng cùng Điều; budget 8000 ký tự; tối đa 3 lượt tìm; qwen2.5:7b, temperature 0.
Retrieval tương ứng: **R7** (hybrid + mở rộng Điều). Baseline so sánh trên test: **A1 / R1** (dense + filter).

A4 (thêm rerank) không được chọn: grounded thấp hơn (3 câu điểm 1 do câu thêm sai), chậm hơn ~1.25 s/lượt tìm, thêm 2.2 GB model.

Lỗi còn mở khi chốt (không chỉnh thêm): u03 — A3 khẳng định sai về thai sản lao động nam; q10 — thiếu điều kiện "báo trước".

Lệnh test (chạy một lần):
```
python -m src.day7.eval_retrieval --split test --configs R1 R7
python -m src.day7.run_agent --split test --configs A1 A3
```
