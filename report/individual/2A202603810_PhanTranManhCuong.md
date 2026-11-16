# Báo cáo Day 7 — Phan Trần Mạnh Cường (2A202603810)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Bộ luật Lao động 2012 (10/2012/QH13) và 2019 (45/2019/QH14), text từ Wikisource, dùng lại corpus Day 6. Văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả (Điều 15 Luật SHTT); phần trình bày trên Wikisource theo CC BY-SA. Corpus gồm 462 Điều, 344k ký tự (≈ 229 trang × 1.500 ký tự). Index Day 6 **strategy B** (chunk theo Điều/khoản, tối đa 1.200 ký tự) có **548 chunk**. Trường `page` là số Điều; metadata gồm năm, chương, `articles`, `part/parts`.
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8.7 (Docker `pgvector/pgvector:0.8.7-pg16`, DB của Day 6), cosine exact search. Embedding `AITeamVN/Vietnamese_Embedding` (BGE-M3, 1024 chiều), chạy local qua sentence-transformers 5.7.0 / torch 2.15.0 trên MPS, chuẩn hóa L2, không prefix — giống lúc ingest. `DenseRetriever` kiểm tra tên model và số chiều của bảng trước khi truy vấn. Thông tin export: `data/day7/export_info.json`.
- **Agent model, cách chạy và phân chia dev/test:** `qwen2.5:7b` qua Ollama local (tool calling, temperature 0, seed 7, `num_ctx` 8192), chi phí API $0. Reranker thử nghiệm: `BAAI/bge-reranker-v2-m3`. Dev 18 câu (15 có đáp án + 3 thiếu evidence), test 8 câu (5 + 3). Chi tiết: [benchmark/README.md](../../benchmark/README.md).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* 20 câu từ benchmark Day 6; `gold` gồm Điều gốc và Điều tương đương ở bản luật còn lại khi nội dung giống hệt. `build_data` kiểm tra mọi evidence nằm trọn trong một chunk đúng Điều. Test Day 6 chỉ chạy một lần sau khi chốt, không dùng để chỉnh.
  - *Câu thiếu evidence:* 6 câu gần chủ đề (lương tối thiểu vùng, tỷ lệ đóng BHXH, thai sản lao động nam, mức phạt, làm việc từ xa, thuế TNCN). Mỗi câu được kiểm tra bằng tìm chuỗi trên toàn bộ `data/raw` và đọc các Điều liên quan; lý do và cách kiểm tra ghi trong `benchmark/unanswerable.json`.
  - *Hit:* chunk đúng (văn bản, Điều) **và** chứa trọn evidence sau chuẩn hóa NFC/lowercase/khoảng trắng.
  - *Filter năm:* suy từ câu hỏi bằng quy tắc cố định (`infer_year_filter`: câu hỏi ghi đúng một năm 2012/2019), trùng với filter người dùng của Day 6 trên cả 20 câu. Gold không đi vào retrieval hay agent.
  - *Tiêu chí chọn cấu hình:* công bố trong `benchmark/README.md` trước khi chạy agent trên dev.

## Retrieval và thử nghiệm

Query = câu hỏi gốc; cùng corpus, index và nhãn. Context = top-3 sau dedupe, giữ thứ tự rank; budget tính bằng ký tự. Thời gian là median ms/query trên M1 Pro (dense gồm cả embed query). Chi phí $0. Nguồn: `results/retrieval/{dev,test}/summary.md`; trace từng câu (top-10, score, thành phần RRF, cờ hit) trong `R*.json`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| R0 | Dense pgvector, k=3, không filter, 4000 ký tự | Điểm xuất phát (= Day 6 B) | dev | 10/15 (67%) | 15/15 (100%) | 15/15 (100%) | 37 ms / $0 |
| **Baseline R1** | Dense pgvector, k=3, filter năm, 4000 | Filter năm loại bản luật sai (= Day 6 C) | dev | 12/15 (80%) | 15/15 (100%) | 15/15 (100%) | 33 ms / $0 |
| Phương pháp 2: R2 | BM25 âm tiết (k1 1.5, b 0.75), filter, 4000 | Lexical khớp tốt số liệu, thuật ngữ | dev | 12/15 (80%) | 13/15 (87%) | 13/15 (87%) | 1,5 ms / $0 |
| Phương pháp 2': R3 | BM25 âm tiết + bigram, filter, 4000 | Bigram xấp xỉ từ ghép tiếng Việt | dev | 12/15 (80%) | 14/15 (93%) | 14/15 (93%) | 2,1 ms / $0 |
| Phương pháp 3 / Cải thiện 1: R4 | Hybrid RRF (dense + BM25 bigram, k=60, depth 30), filter, 4000 | Kéo evidence dense xếp hạng 2–3 lên hạng 1 | dev | 13/15 (87%) | 14/15 (93%) | 14/15 (93%) | 37 ms / $0 |
| Phương pháp 4: R5 | R4 + cross-encoder bge-reranker-v2-m3 trên 20 candidate | Rerank tăng Hit@1 | dev | 14/15 (93%) | 15/15 (100%) | 15/15 (100%) | 1.254 ms / $0 |
| R6 | R5 + mở rộng cùng Điều, 6000 | Giữ đủ điều kiện của một Điều | dev | 14/15 (93%) | 15/15 (100%) | 15/15 (100%) | 1.268 ms / $0 |
| Cải thiện 2: R7 | R4 + mở rộng cùng Điều, 6000 | Sửa lỗi q05/q12: hit nằm ở phần khác của cùng Điều | dev | 13/15 (87%) | 14/15 (93%) | 15/15 (100%) | 39 ms / $0 |
| **Cấu hình đã chọn (R7, tool của A3)** | như trên | — | test | 3/5 (60%) | 5/5 (100%) | 5/5 (100%); chấm tay mức 2 ở 5/5 câu | 35 ms / $0 |
| Baseline R1 | như trên | — | test | 3/5 (60%) | 5/5 (100%) | 5/5 (100%); chấm tay mức 2 ở 4 câu, mức 1 ở q17 | 32 ms / $0 |

Ghi chú:
- R0/R1 khớp chính xác kết quả Day 6 (B 10/15, C 12/15), xác nhận kết nối index đúng. R4 vừa là phương pháp mới (hybrid) vừa là cải thiện 1; R7 là cải thiện 2 (chọn context); R5/R6 là phương pháp thứ 4 (rerank).
- Context Evidence Rate tính trên chunk thực sự nằm trong context sau `select_context`, nên với R7 có thể lớn hơn Hit@3.
- **Chấm context 0/1/2 trên test** (`results/context_review_test.json`): R1 0/1/4 — ở q17 có khoản 2b nhưng thiếu phần B-00 định nghĩa "điểm d = hoàn cảnh khó khăn"; R7 0/0/5; agent A1 2/0/3 (q17 lấy sai đoạn, q20 không có lượt tìm); agent A3 0/0/5.

**Hai lỗi dev, nguyên nhân và cách thử**

1. **q05 và q12 — hit nằm ở phần khác của cùng một Điều.**
   - *Hiện tượng:* R4 xếp `blld-2019-d020-B-01` (khoản 2–3 Điều 20) hạng 1; phần chứa "không quá 36 tháng" (`d020-B-00`) rơi xuống hạng 4 vì BM25 xếp nó hạng 16. Ở q12, `d125-B-00` đứng hạng 1 trong khi evidence "05 ngày cộng dồn" nằm ở `d125-B-01`.
   - *Giả thuyết:* đưa mọi phần cùng Điều vào context ngay sau hit.
   - *Kết quả:* R4 → R7 tăng Context Evidence 14/15 → 15/15, Hit@1 giữ nguyên. Ở mức agent, A2 (hybrid, không mở rộng) trả lời sai q05 ("thời hạn không được quy định cụ thể"); A3 trả lời đúng.
2. **q12 trên agent — model viết query kém và dừng sau 1 lượt.**
   - *Hiện tượng:* A1/A2 gọi `search_law("nghỉ không phép", year=2019)` — năm do model tự thêm dù câu hỏi không ghi năm — và nhận Điều 111/113/109 (nghỉ hằng tuần, hằng năm). Model từ chối, không tìm lại dù còn 2 lượt.
   - *Giả thuyết:* lượt tìm đầu luôn dùng câu hỏi gốc; model vẫn được phép tìm thêm.
   - *Kết quả:* A3 lấy được `d125-B-01`, trả lời đúng (0 → 2).
   - *Bằng chứng bổ sung:* retrieval với câu hỏi gốc (R1) có Context 15/15 trên dev, còn agent A1 (query do model viết) chỉ 14/15 trên dev và 3/5 trên test — query do model viết lại làm giảm chất lượng retrieval.

Hai lỗi khác phát hiện trên dev và được xử lý trong code (áp dụng chung cho mọi cấu hình agent):
- **Tool call hỏng:** model in lời gọi tool dạng text (`CallCheck_law(...)`, `<tool_call>`) ở 10/18 câu dev của A1. Agent nhắc lại đúng 1 lần và ghi event `no_search_reprompt`.
- **Citation sai định dạng:** model ghi `[chunk-id: X]` hoặc `[X, Y]`; parser được sửa để nhận các dạng này.

**Lý do chọn** (chi tiết: [results/SELECTION.md](../../results/SELECTION.md)): retrieval xét theo Context Evidence → Hit@3 → Hit@1 → thời gian. R5/R6/R7 cùng đạt Context 15/15; R6 có Hit@1/Hit@3 cao hơn nhưng chậm hơn khoảng 33 lần. Quyết định cuối nằm ở mức agent: A3 (dùng R7) có grounded 1,93, cao hơn A4 (dùng R6) 1,80, và nhanh hơn.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** (`src/day7/agent.py`):
  - *Tool:* `search_law(query, year?)`.
  - *Filter:* nếu câu hỏi ghi đúng một năm luật, hệ thống áp filter đó (nguồn `user_question`); nếu không, dùng tham số `year` của model (nguồn `model`). Cả hai được ghi vào trace.
  - *Dừng:* model trả lời mà không gọi tool, hoặc đã đủ `max_searches=3` (khi đó bỏ tools, buộc trả lời từ context hiện có).
  - *Context:* dedupe giữa các lượt; budget cộng dồn theo ký tự (6.000 cho A1/A2, 8.000 cho A3/A4); chunk không vừa bị bỏ và ghi vào `dropped_for_budget`.
  - *Citation:* bắt buộc `[chunk-id]`. Nếu mọi ID được trích đều nằm ngoài context, câu trả lời bị chuyển thành `abstain` (event `invalid_or_missing_citations`); ID bịa được lưu trong `invalid_citations`.
  - *Lỗi:* lỗi Ollama/DB ghi `status: error`, không thay bằng câu trả lời giả. Trong toàn bộ 88 lượt chạy không có lỗi hệ thống.
  - *An toàn:* system prompt quy định nội dung tài liệu là dữ liệu, không phải chỉ dẫn.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A1 dense, model viết query / dev | 18/18 | 1 lượt: 18 | 2/1/12 — 1.67 | 15/15 | 14/14 | 2/3 | 2/15 |
| A2 hybrid / dev | 18/18 | 1 lượt: 18 | 3/2/10 — 1.47 | 12/14 | 13/13 | 2/3 | 2/15 |
| **A3** hybrid + query gốc lượt 1 + mở rộng Điều / dev | 18/18 | 1 lượt: 18 | 0/1/14 — **1.93** | 19/20 | 16/16 | 2/3 | 0/15 |
| A4 = A3 + rerank / dev | 18/18 | 1 lượt: 18 | 0/3/12 — 1.80 | 23/25 | 16/16 | 2/3 | 0/15 |
| A1 baseline / test | 7/8 | 0 lượt: 1, 1 lượt: 7 | 2/0/3 — 1.20 | 3/4 | 4/4 | 3/3 | 1/5 |
| **A3 đã chọn / test** | 8/8 | 1 lượt: 8 | 0/2/3 — **1.60** | 4/8 | 6/6 | **1/3** | 0/5 |

Thời gian median/câu và token trung bình (prompt + completion); chi phí API $0 cho mọi cấu hình:

| | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| dev | 12,0 s · 2.366+207 | 13,2 s · 2.392+232 | 8,0 s · 1.447+104 | 10,9 s · 1.523+130 |
| test | 9,5 s · 2.150+197 | — | 8,7 s · 1.506+107 | — |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q17 test, A3 | Có đáp án (filter 2012) | `search_law(<câu hỏi gốc>, year=2012 [user])` → d037-B-00, d038-B-01, d037-B-01; mở rộng Điều 37 đưa cả B-00 (điểm d "hoàn cảnh khó khăn") và B-01 (khoản 2b) vào context | "phải báo trước ít nhất 30 ngày [blld-2012-d037-B-01]" | **2**, citation hỗ trợ. Cùng câu này A1 trả lời "30 ngày" khi context **không có** khoản 2 → chấm 0 (đúng nhờ kiến thức sẵn của model, không có evidence) |
| u05 test, A3 | Thiếu evidence | `search_law(<câu hỏi gốc>, year=2019)` → Điều 136, 178, 40 | "Không đủ bằng chứng để trả lời. Điều 136… không nêu hỗ trợ chi phí internet…" | **Từ chối đúng**, không thêm khẳng định |
| u06 test, A3 | Thiếu evidence (lỗi) | `search_law(<câu hỏi gốc>)` → Điều 97 (2012), 98 (2019), 106 (2012) | "tiền lương làm thêm giờ **không được miễn thuế TNCN** … [d097][d098]" | **Sai**: khẳng định không có căn cứ; Điều 97/98 chỉ quy định mức 150/200/300%. Citation có trong context nhưng không hỗ trợ ý chính |
| q18 test, A3 | Có đáp án (lỗi) | → d036-B-02 (NSDLĐ báo trước), d035-B-00 (NLĐ báo trước), d038-B-01 | "45 ngày [blld-2019-d036-B-02]" | **1**: con số đúng nhưng trích Điều 36 (người sử dụng lao động) thay vì Điều 35, dù Điều 35 có trong context |
| q20 test, A1 | Có đáp án (lỗi) | Không có lượt tìm: model in `{"name": "search_law", …}</tool_call>` dạng text cả hai lần, kể cả sau khi được nhắc | Output không hợp lệ → abstain | **0**, vẫn tính vào mẫu số |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - *Trace:* `results/agent/{dev,test}/A*/traces.json` — mỗi câu có mọi tool call (query, tham số của model, filter và nguồn filter, top-k + score), context cuối, câu trả lời, citation, ID không hợp lệ, token, thời gian.
  - *Nhãn chấm tay:* `reviews.json` cạnh từng trace, mỗi nhãn có lý do; chấm context: `results/context_review_test.json`.
  - *Metrics:* `agent_metrics.json` (tổng hợp bằng `starter/agent_metrics.aggregate`) và `results/agent/{dev,test}/summary.md`.
  - *Thời gian chạy:* dev A1 220 s, A2 239 s, A3 151 s, A4 203 s; test A1 88 s, A3 70 s.
  - *Lệnh chạy:* xem README.
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - *Cải thiện từ A1 sang A3 giữ được trên test:* context có evidence 3/5 → 5/5, grounded 1,20 → 1,60, từ chối nhầm 1/5 → 0/5; câu trả lời ngắn hơn, ít token hơn khoảng 40% (dev: 2.573 → 1.551 token/câu).
  - *Đánh đổi:* abstention trên test **giảm** từ 3/3 xuống 1/3, citation hợp lệ 3/4 → 4/8. Trên dev, abstention của hai cấu hình bằng nhau (2/3).
  - *Giả thuyết cho việc giảm abstention* (chưa kiểm chứng vì không chỉnh trên test): khi query là câu hỏi gốc, context chứa các Điều rất sát chủ đề (Điều 98 về lương làm thêm cho câu hỏi thuế, Điều 127 cho câu hỏi về phạt), khiến model 7B dễ suy diễn thành khẳng định. A1 tự viết query nên context ít liên quan hơn và model từ chối nhiều hơn. Lỗi cùng loại đã xuất hiện ở dev u03 (A3/A4 khẳng định sai về thai sản lao động nam) nhưng chưa được xử lý trước khi chốt.
  - *Hướng tiếp theo* (cần đánh giá trên một bộ test mới): bước kiểm chứng buộc mỗi khẳng định khớp với chunk được trích; model lớn hơn; yêu cầu trích nguyên văn số liệu.
  - *Giới hạn:*
    - Test chỉ có 8 câu; một câu chiếm 12,5–20 điểm phần trăm, nên chênh lệch A1/A3 chưa có ý nghĩa thống kê.
    - Không lượt chạy nào có quá 1 lượt tìm (0/88), nên chưa đo được multi-hop.
    - Ở A1/A2 model tự viết query và tự thêm filter năm cho 12/18 câu dev. Ở A3/A4 model **không tự gọi lượt tìm nào** (0/52): retrieval chỉ là lượt đầu do hệ thống chạy với câu hỏi gốc, model quyết định trả lời hay từ chối. Cấu hình được chọn vì vậy gần với RAG một bước có model kiểm soát việc dừng hơn là agent tự lập kế hoạch tìm kiếm.
    - Nhãn do một người chấm.
    - Câu có đáp án tái sử dụng từ Day 6.
    - Reranker cần tải thêm 2,2 GB và chạy trên MPS.
    - Chi phí $0 chưa tính điện năng và phần cứng.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
