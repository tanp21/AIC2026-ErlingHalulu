# Holistic Query Paraphrasing Benchmark 2026 — PE-Core

## Bối cảnh

Benchmark trước (`docs/reports/visual-query-expansion-benchmark-2026.md`) thử tách 1 caption
thành holistic + aspect (mỗi aspect thêm 1 chi tiết) — **kết quả âm tính**: không cấu hình nào
vượt baseline. Benchmark này thử hướng ngược lại: **giữ nguyên hoàn toàn ngữ nghĩa** của caption
đã duyệt, chỉ đổi cách diễn đạt (paraphrase), rồi fuse nhiều bản diễn đạt của CÙNG một caption.
Giả thuyết: nếu nhiễu retrieval đến từ việc PE-Core nhạy cảm với cách viết câu, thì trung bình
hóa nhiều cách diễn đạt tương đương sẽ làm mượt nhiễu đó.

**Kết quả: TÍCH CỰC, và tái lặp được trên holdout.** Cấu hình tốt nhất (`B*`: trung bình vector
embedding của caption + 4 paraphrase, trọng số caption=0.5) vượt baseline
`caption_approved_shotdedup` trên cả dev (95 case) lẫn holdout (Round 3, 42 case) — đây là lần
đầu tiên trong 2 benchmark liên tiếp có một phương pháp thực sự vượt qua baseline đơn giản.

## Phương pháp

- **Nguồn paraphrase**: chỉ dùng caption tiếng Anh đã duyệt (`caption_en_approved`), KHÔNG dùng
  prompt tiếng Việt gốc — đảm bảo không có chi tiết nào len lỏi trở lại.
- **4 paraphrase/case, thứ tự chiến lược cố định** (subset kích thước n luôn lấy n cái đầu, tránh
  chọn lọc kết quả): `reorder` (đổi trật tự cụm từ/chủ động-bị động), `synonym` (giữ cấu trúc,
  đổi từ đồng nghĩa), `restructure` (đổi cấu trúc ngữ pháp), `combined` (cả 2).
- **Validation nhiều lớp** (`scripts/benchmarks/paraphrase_validation.py`): giữ nguyên số đếm,
  màu sắc, chữ trích dẫn/viết hoa, danh từ riêng; cấm thêm từ khung ("photo", "video"...); kiểm
  tra trùng lặp; **và một lượt Gemini "judge" riêng** xác nhận tương đương ngữ nghĩa tuyệt đối
  (equivalent=true, added/removed/changed đều rỗng). Sai ở bất kỳ bước nào → tự động viết lại tối
  đa 2 lần, vẫn fail thì loại bỏ slot đó.
- **Kết quả sinh**: 547/548 slot pass (99.8%), chỉ 1 slot bị judge loại (thêm "is present" không
  có trong gốc — đúng theo thiết kế), 1/137 case chỉ còn 3 paraphrase. Max 42 token (dưới giới
  hạn PE-Core 72 rất nhiều). **Toàn bộ nội dung paraphrase đã được người dùng duyệt trước khi
  benchmark.**
- **3 họ fusion** thử nghiệm, tất cả K=1000/list (cố định, không tune):
  - **RRF theo shot** (`weighted_shot_rrf`): mỗi list gộp theo shot trước, cộng
    Σ wᵢ/(k+rank), chỉ dùng thứ hạng, không dùng điểm số gốc.
  - **Score fusion theo shot** (`weighted_shot_score`, chỉ ablation): Σ wᵢ·điểm gốc.
  - **Fusion sớm ở tầng embedding** (`combine_vectors`): trung bình có trọng số các vector đã
    chuẩn hoá, chuẩn hoá lại, tìm kiếm FAISS **một lần duy nhất**.
- **10 invariant toán học** cho RRF/score/embedding fusion, tất cả đã unit test (bound theo
  Σw/(k+1), bất biến hoán vị, đơn điệu theo rank, v.v.) — 59 test pass
  (`tests/visual/test_paraphrase_fusion.py`, `test_paraphrase_validation.py`,
  `test_retrieval_cache.py`).
- **Dev/holdout tách biệt nghiêm ngặt**: dev = Test+Round1+Round2 (95 case), holdout = Round 3
  (42 case), **chỉ đọc đúng 1 lần ở bước cuối cùng** (`holdout_done.json` khoá lại, chạy lần 2 sẽ
  bị từ chối).

## Kết quả trên Dev

**Stage 0 sanity**: `weighted_shot_rrf`/`weighted_shot_score`/`combine_vectors` với 1 list (chỉ
caption) đều tái tạo **chính xác tuyệt đối** `caption_approved_shotdedup` trên cả 95 case — xác
nhận harness đúng trước khi tin số liệu.

Từng paraphrase đứng một mình đều **kém hơn** caption gốc (đúng như dự đoán — PE-Core nhạy với
cách diễn đạt): reorder MRR .422, synonym .412, restructure .421, **combined .356 (kém nhất, p=0.001
có ý nghĩa thống kê)**.

**Stage 1** (9 cấu hình): fusion RRF và embedding-mean với `caption + 2 paraphrase` (trọng số
0.5/0.25/0.25) đều **vượt nhẹ** baseline. `E1` (embedding-mean) đạt MRR=.444, R@1=.347 (baseline
.442/.326) — **đầu tiên đạt ngưỡng chấp nhận** (+2 case R@1, không tệ hơn quá 1/3 vòng dev).

**Stage 2** (biến thiên số paraphrase 2→3→4): **RRF càng thêm paraphrase càng tệ** (n=3: MRR
-0.015, n=4: MRR -0.012 so với baseline — bị loại hoàn toàn). **Embedding-mean cải thiện đều đặn**:
n=2 MRR .444, n=4 MRR .448, R@1 .351 — cả 2 đều PASS. `B*` = embedding-mean, `caption + 4
paraphrase`, trọng số cap50 (0.5/0.125×4).

**Stage 3** (ablation trên `B*`): đổi sang RRF cùng list/trọng số → tệ hẳn (bị loại). Đổi trọng số
cap67 (caption=2/3) → gần như không đổi (ΔMRR=.002, không đủ vượt ngưỡng so với cap50) → **giữ
nguyên cap50**.

| Method (dev, n=94-95) | MRR | R@1 | R@10 | Chấp nhận? |
|---|---:|---:|---:|---|
| `caption_approved` | .432 | .326 | .579 | (baseline gốc) |
| `caption_approved_shotdedup` | .442 | .326 | .653 | (baseline chính) |
| RRF, 3 paraphrase, equal, k=10 | .423 | .305 | .653 | fail |
| RRF, caption+2para, cap50, k=10 | .444 | .337 | .684 | fail (dR@1 chỉ +1) |
| Emb-mean, caption+2para, cap50 | .444 | .347 | .663 | **PASS** |
| Emb-mean, caption+4para, cap50 (`B*`) | **.448** | **.351** | .649 | **PASS** |
| `B*` đổi thành RRF | .434 | .319 | .660 | fail |
| `B*` với cap67 | .450 | .351 | .660 | fail (so với `B*`, ΔMRR chỉ .002) |

## Kết quả Holdout (Round 3, 42 case — chỉ chạy 1 lần)

| Method | Holdout MRR | Holdout R@1 | Tổng 3 vòng 2026 MRR | Tổng R@1 |
|---|---:|---:|---:|---:|
| `caption_approved` | .538 | 45.2% | .466 | 37.1% |
| `caption_approved_shotdedup` | .545 | 45.2% | .479 | 37.1% |
| 3 paraphrase, RRF equal | .562 | 47.6% | .462 | 34.3% *(tệ hơn baseline)* |
| caption+2para, RRF cap50 | .549 | 45.2% | .480 | 37.1% |
| **`B*` (emb-mean, caption+4para, cap50)** | **.556** | **47.6%** | **.488** | **39.4%** |

**`B*` thắng trên cả 3 lát cắt** (dev, holdout, tổng 3 vòng) — nhất quán, không phải may mắn trên
1 tập dữ liệu. Trên tổng 105 case cả 3 vòng 2026: R@1 tăng từ 37.1% → 39.4% (+2.4 case), MRR
+0.009. Win/loss trên holdout: **9 thắng / 4 thua / 29 hòa** (p=0.27, chưa đạt ý nghĩa thống kê
nghiêm ngặt do n nhỏ, nhưng tỷ lệ thắng/thua hơn 2 lần và khoảng tin cậy bootstrap của ΔR@1 có
cận dưới = 0.0 — chưa từng quan sát delta âm trong resample).

Chú ý: **RRF 3-paraphrase (biến thể bắt buộc #3 của user) lại TỆ HƠN baseline trên tổng 2026**
(R@1 34.3% vs 37.1%) dù tốt hơn trên riêng holdout — một ví dụ rõ về việc RRF kém ổn định hơn
embedding-mean khi không có caption neo trong list.

## Phân tích rescued/worsened (B* vs baseline, tổng 3 vòng 2026)

- **2 case được cứu** (baseline hạng >10 → fused ≤10): `Round 1-2026/query-p1-16-trake:3` (11→8),
  `Round 3-2026/query-p2-11-kis:1` (12→10).
- **0 case bị làm tệ đi** (baseline ≤10 → fused >10 hoặc miss) — fusion không bao giờ biến 1 case
  đang tốt thành miss hẳn.
- **Top di chuyển hạng**: `query-p1-21-kis` (4→1), `query-p1-5-kis` (2→1),
  `query-p2-30-kis` (2→1) — đều là case mà 1-2 paraphrase (thường là `synonym`, giữ cấu trúc gần
  nguyên nhưng đổi từ, VD "shears" thay "scissors", "sautés" thay "stir-fries") giúp bổ sung tín
  hiệu semantic gần với cách CLIP "hiểu" khái niệm đó hơn. Một số ít case bị lùi nhẹ hạng (VD
  `query-p1-10-kis`: 1→3) nhưng vẫn nằm trong top thấp, không thành miss.

## Kết luận

1. **Paraphrase + embedding-mean fusion là phương pháp đầu tiên vượt được baseline một cách nhất
   quán**, khác hẳn kết quả âm tính của query-expansion (aspect decomposition). Cải thiện khiêm
   tốn nhưng **tái lặp trên cả dev và holdout**: R@1 +2.3 điểm %, MRR +0.009 trên tổng 105 case.
2. **Embedding-mean (fusion sớm) rõ ràng tốt hơn RRF (fusion muộn theo rank)** cho bài toán này —
   RRF càng thêm paraphrase càng tệ, trong khi embedding-mean cải thiện đều. Có thể vì trung bình
   vector giữ được thông tin liên tục (magnitude), còn RRF chỉ dùng thứ hạng rời rạc nên nhạy với
   nhiễu thứ hạng của từng paraphrase riêng lẻ.
3. **Trọng số caption cao (0.5) là cần thiết** — cap50 và cap67 gần tương đương, nhưng cấu hình
   `equal weight` (không ưu tiên caption) luôn kém hơn cap50 ở mọi thử nghiệm.
4. **Khuyến nghị**: cân nhắc triển khai Phase B (`POST /search/expand` kiểu paraphrase, không
   phải aspect) — kiến trúc production đơn giản hơn nhiều so với query-expansion (không cần shot
   fusion phức tạp, chỉ cần: Gemini sinh N paraphrase → encode N+1 vector → trung bình có trọng
   số → 1 lần FAISS search). Tuy nhiên **effect size còn khiêm tốn** (+2.4/105 case) và n holdout
   nhỏ (42) nên p-value win/loss chưa đạt ý nghĩa thống kê nghiêm ngặt — nên xem đây là bằng
   chứng "đủ khuyến khích để thử ở quy mô lớn hơn", không phải "chắc chắn tốt hơn hẳn".
5. **Giới hạn**: mới thử 1 phiên bản prompt paraphrase (chưa cần lặp vì Stage 1 đã pass ngay), 1
   bộ trọng số chính (cap50/cap67), K cố định 1000. Có thể còn dư địa cải thiện thêm (VD trọng số
   không đều giữa các paraphrase, hoặc paraphrase có điều kiện theo độ dài caption).

## Phương pháp & dữ liệu thô

- Script sinh + validate: `scripts/benchmarks/paraphrase_visual_queries.py` +
  `scripts/benchmarks/paraphrase_validation.py`.
- Script benchmark theo giai đoạn: `scripts/benchmarks/bench_holistic_paraphrase.py`
  (`--stage sanity|stage1|stage2|stage3|freeze|holdout`).
- Fusion thuần (không I/O): `backend/subquery_fusion.py` (`weighted_shot_rrf`,
  `weighted_shot_score`, `combine_vectors`) — 20 test mới + 19 test cũ (query-expansion), tất cả
  pass.
- Paraphrase đã duyệt: `artifacts/holistic_paraphrase/v1/paraphrases.json` + `review.txt`.
- Log toàn bộ 18 cấu hình đã thử (stage 1-3): `artifacts/holistic_paraphrase/tuning_log.jsonl`.
- Cấu hình đã đóng băng: `benchmarks/visual/holistic_paraphrase_config.json`.
- Kết quả đầy đủ (dev + holdout, từng case): `benchmarks/visual/holistic_paraphrase_benchmark_2026.json`.
- Đánh dấu holdout đã chạy (khoá, không chạy lại): `artifacts/holistic_paraphrase/holdout_done.json`.
