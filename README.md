# Day 7 — Agentic RAG

> ## Bài làm — Phan Trần Mạnh Cường (2A202603810)
>
> **Corpus:** Bộ luật Lao động 2012 và 2019, dùng lại từ Day 6: 462 Điều, 548 chunk (strategy B). **Embedding:** `AITeamVN/Vietnamese_Embedding`, 1024 chiều, chạy local. **Store:** pgvector của Day 6. **Agent:** `qwen2.5:7b` qua Ollama, có tool calling. **Chi phí API:** $0.
> **Retrieval:** dense (pgvector), BM25 (âm tiết / bigram), hybrid RRF, hybrid + cross-encoder rerank; thêm tùy chọn mở rộng context theo Điều. **Cấu hình chọn:** A3 (hybrid, lượt tìm đầu dùng câu hỏi gốc, mở rộng Điều).
> Báo cáo: [report/individual/2A202603810_PhanTranManhCuong.md](report/individual/2A202603810_PhanTranManhCuong.md) · Chốt cấu hình: [results/SELECTION.md](results/SELECTION.md) · Benchmark: [benchmark/README.md](benchmark/README.md)
>
> ```bash
> # 0) Chuẩn bị: pgvector Day 6 đã ingest (cd <repo Day 6>/starter && docker compose -p day6_student_starter up -d --wait)
> #    và Ollama: ollama pull qwen2.5:7b
> python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
> cp .env.example .env
> python -m src.day7.build_data --day6 ../K5-L3B-DAY06-PhanTranManhCuong-2A202603810-DataIngestion   # export chunk + benchmark 26 câu
> python -m unittest discover -s tests -v                          # test offline (mock), 25 test
> python -m src.day7.eval_retrieval --split dev                    # R0–R7: Hit@1, Hit@3, Context Evidence
> python -m src.day7.run_agent --split dev --configs A1 A2 A3 A4   # agent thật, lưu trace
> python -m src.day7.review show --split dev --config A1           # in trace để chấm tay -> reviews.json
> python -m src.day7.review summarize --split dev --configs A1 A2 A3 A4
> # sau khi chốt (results/SELECTION.md):
> python -m src.day7.eval_retrieval --split test --configs R1 R7
> python -m src.day7.run_agent --split test --configs A1 A3
> python -m src.day7.review summarize --split test --configs A1 A3
> ```
>
> | Đường dẫn | Nội dung |
> |---|---|
> | `src/day7/` | `retrievers.py` (dense/BM25/hybrid/rerank), `context.py`, `agent.py`, `eval_retrieval.py`, `run_agent.py`, `review.py`, `build_data.py` |
> | `data/day7/` | `chunks.json` (export từ pgvector, không có vector), `questions.json`, `cases.json`, `export_info.json` |
> | `benchmark/` | Câu thiếu evidence + cách kiểm tra, tiêu chí chọn cấu hình chốt trước |
> | `results/retrieval/{dev,test}/` | Trace top-10 từng câu và bảng tổng hợp cho mỗi cấu hình R* |
> | `results/agent/{dev,test}/A*/` | `traces.json`, `reviews.json` (nhãn tay), `agent_metrics.json`, `auto_summary.json` |
> | `results/context_review_test.json` | Chấm context 0/1/2 trên test |

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
