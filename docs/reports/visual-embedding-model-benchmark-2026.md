# Visual Embedding Benchmark 2026 — PE-Core vs SigLIP2 NaFlex vs SigLIP2 GOPT

## Bối cảnh

Câu hỏi: production hiện đang dùng PE-Core-bigG-14-448 (`backend/core.py`) cho nhánh visual
search. Một benchmark trước đó (`scripts/benchmarks/bench_naflex_ensemble.py`, xem
`benchmarks/naflex/naflex_ensemble_official.json`) đã so PE-Core với SigLIP2 NaFlex
(SO400M) trên P1-2025 + Round 1/2-2026 và thấy NaFlex vượt trội hẳn PE-Core. Gần đây thêm
một ứng viên thứ ba: **SigLIP2 GOPT** (`timm/ViT-gopt-16-SigLIP2-384`, ảnh cố định 384px,
không NaFlex, dim 1536), đã được embed cho toàn bộ corpus "Batch 1 2026" (L21-L30) trên 4
notebook Kaggle song song. Báo cáo này mở rộng benchmark thành 3 chiều, chạy trên **cả 3
vòng thi 2026** (trước đây chỉ có Round 1-2; Round 3 có đáp án đầy đủ trong DB production
nhưng chưa từng được export/benchmark).

## ⚠️ Lỗi phương pháp đã phát hiện và sửa: ngôn ngữ + độ dài query

Bản chạy đầu tiên dùng thẳng prompt tiếng Việt gốc trong DB làm query text cho cả 3 model —
**đây là sai lầm phương pháp nghiêm trọng**, phát hiện sau khi người dùng đặt câu hỏi. Cả 3
text tower đều là CLIP-style, huấn luyện chủ yếu trên caption tiếng Anh, với context window
chỉ **64-72 token** (NaFlex/GOPT: 64, PE-Core: 72) — và **cắt cụt âm thầm** phần vượt quá,
không báo lỗi. Kiểm tra lại:

- Prompt tiếng Việt gốc: **80/95 câu (84%) đã vượt quá 72 token** — phần lớn benchmark ban
  đầu chạy trên query bị cắt cụt giữa chừng, mất đúng phần chi tiết đặc trưng (thường nằm ở
  cuối câu mô tả dài).
- Ngay cả bản dịch tiếng Anh "tự nhiên, đầy đủ chi tiết" đã có sẵn trong repo
  (`artifacts/official_benchmark_queries_en.json`, dùng cho benchmark NaFlex/PE-Core cũ trên
  P1/SOTUYEN1/2) vẫn còn **30/95 câu (32%) vượt 64 token**, vì giữ nguyên khung chỉ dẫn kiểu
  "Find the exact clip where...", "The video describes..." — không giống caption huấn luyện
  CLIP, tốn token vô ích mà không thêm nội dung tìm kiếm được.

**Bằng chứng tác động**: PE-Core (model thuần tiếng Anh, không có tokenizer đa ngôn ngữ) là
model bị ảnh hưởng nặng nhất — số case tìm được đáp án trong top-1000 tăng từ **40/105 (query
tiếng Việt) lên 95/105 (query tiếng Anh ngắn)**. Kết luận "PE-Core kém, nên thay bằng
GOPT/NaFlex" ở bản benchmark đầu tiên **hoàn toàn là hệ quả của lỗi ngôn ngữ/cắt cụt, không
phản ánh chất lượng model thật**.

**Cách sửa**: `scripts/benchmarks/translate_visual_queries.py` — dùng Gemini
(`gemini-3.1-flash-lite`) dịch lại toàn bộ 105 case (Round 1-3, 2026) thành caption ngắn kiểu
CLIP: bỏ hết khung chỉ dẫn/meta-commentary, giữ chi tiết thị giác cụ thể (số lượng, màu sắc,
hành động, chữ trên màn hình, tên riêng), ưu tiên chi tiết hiếm/đặc trưng nếu phải rút gọn,
kiểm tra lại độ dài token qua đúng tokenizer PE-Core sau mỗi lượt dịch (tự động rút gọn thêm
nếu vẫn vượt ngưỡng an toàn 50 token). Kết quả: **0/137 câu vượt 50 token** (trung bình 22,
tối đa 38) — an toàn dưới mọi giới hạn của cả 3 model. Toàn bộ nội dung dịch đã được người
dùng xem và duyệt trước khi chạy benchmark chính thức.

## Dữ liệu & phương pháp

- **Ground truth**: lấy trực tiếp từ DB production đang chạy (`aic_hcmc_collab-data` docker
  volume, `aic-collab.sqlite3` — **không phải** file `artifacts/collaboration.sqlite3` đã
  commit trong repo, file đó chỉ là fixture rỗng dùng cho test). 3 vòng đều đã có đáp án đầy
  đủ: Round 1 - 2026 (25 query → 27 case sau khi tách sự kiện TRAKE), Round 2 - 2026 (30 →
  36), Round 3 - 2026 (36 → 42) — tổng **105 case** trên 3 vòng 2026 (không tính 32 case của
  vòng "Test"/P1-2025, giữ lại trong dữ liệu xuất ra để tham khảo nhưng không đưa vào bảng
  chính). Mỗi câu KIS/QA là 1 case; mỗi câu TRAKE tách thành N case theo từng mốc sự kiện
  (markers không đồng nhất giữa các vòng: "E1", "E1:", "Cảnh 1:" — regex đã bao cả 3 dạng).
  Đáp án mỗi case là **danh sách** (video_id, frame_id) — gồm frame được nộp gốc cộng các
  frame cùng shot được tự động thêm làm "chấp nhận được" (`auto_filled: true`), dùng làm tập
  target để tính rank.
- **Query text**: caption tiếng Anh ngắn (xem mục trên), không phải prompt tiếng Việt gốc.
- **Metric**: giống hệt `bench_naflex_ensemble.py` — một kết quả được tính là "trúng" nếu
  cùng video và lệch ≤10 giây so với bất kỳ frame nào trong tập target (`tolerance_seconds`).
  Báo cáo MRR, Recall@1/5/10/100, số case "hits" (tìm thấy trong top-1000) trên tổng số case.
- **3 profile FAISS Flat cùng row-mapping** (`id_video`/`id_frame` giống hệt nhau theo từng
  vị trí, đã kiểm chứng bằng `np.array_equal`) để RRF fusion hợp lệ:
  - `l2026` — PE-Core-bigG-14-448, dim 1280, có sẵn.
  - `naflex-l2026` — SigLIP2 SO400M NaFlex, dim 1152, có sẵn.
  - `gopt-l2026` (mới build) — SigLIP2 GOPT, dim 1536. Nguồn: 4 notebook Kaggle
    (`taiphamnguyenanh/aic-siglip2-gopt-p1of4`, `.../p4of4-friend`,
    `tiphmnguynanh/aic-siglip2-gopt-{p2,p3}of4`), mỗi notebook embed một phần video rời rạc
    của L21-L30 (289,875 frame thô). Notebook p4of4 gốc chạy lỗi (status ERROR, không có
    output) — phải chạy lại dưới một tài khoản Kaggle khác (`tim46c/aic-siglip2-gopt-p4of4-friend`)
    mới lấy được. 4 phần được gộp lại và **sắp xếp toàn cục theo (video_id, frame_idx)**
    (`scripts/embedding/assemble_gopt_batch1.py`) để khớp đúng thứ tự row mà
    `scripts/indexing/build_flat_profile.py` đã dùng khi build `l2026`/`naflex-l2026` — nếu
    không sắp lại, số lượng vector khớp (268,254/873 video) nhưng thứ tự index không khớp,
    khiến RRF fusion cộng sai điểm giữa các frame khác nhau. Sau khi build với cùng
    `--shot-label-dir artifacts/shot-labels`, số loại trừ khớp chính xác với 2 profile kia
    (21,621 frame: intro/presenter/preview/outro).
- **Encode query text**: mỗi model dùng đúng text tower gốc qua `open_clip` (không qua
  `backend/core.py`, vốn chỉ hard-code PE-Core). `open_clip` bản cài (3.3.0) không còn build
  được full model NaFlex (thiếu field `image_seq_len` trong `CLIPVisionCfg`) và không còn
  local config `ViT-SO400M-16-SigLIP2-naflex` trong registry cũ — text tower được dựng thủ
  công từ `open_clip_config.json` tải về, bỏ qua hoàn toàn vision tower.
- Script benchmark: `scripts/benchmarks/bench_visual_embeddings.py` (mở rộng từ
  `bench_naflex_ensemble.py`, thêm GOPT + Round 3 + breakdown theo từng vòng). Kết quả đầy đủ
  (từng case, từng rank): `benchmarks/visual/visual_embedding_benchmark_2026_en.json` (bản
  chính thức, query tiếng Anh) — file `visual_embedding_benchmark_2026.json` (không hậu tố
  `_en`) là **bản chạy lỗi ban đầu (query tiếng Việt), giữ lại chỉ để đối chiếu**, không dùng
  để ra quyết định.

## Kết quả — 3 model riêng lẻ (query tiếng Anh, đã sửa lỗi)

| Vòng (n case) | Model | MRR | R@1 | R@5 | R@10 | R@100 | Hits/n |
|---|---|---:|---:|---:|---:|---:|---:|
| Round 1 (27) | **PE-Core** | **.307** | **22.2%** | 37.0% | 44.4% | 70.4% | 24/27 |
| | NaFlex | .318 | 22.2% | **44.4%** | 48.1% | **77.8%** | 25/27 |
| | GOPT | .280 | 22.2% | 33.3% | 44.4% | 70.4% | 24/27 |
| Round 2 (36) | **PE-Core** | **.503** | **38.9%** | 63.9% | 63.9% | **86.1%** | 33/36 |
| | NaFlex | .434 | 36.1% | 52.8% | 63.9% | 80.6% | 33/36 |
| | GOPT | .418 | 30.6% | **58.3%** | 63.9% | 88.9% | 33/36 |
| Round 3 (42) | **PE-Core** | **.538** | **45.2%** | 59.5% | **64.3%** | 81.0% | 38/42 |
| | NaFlex | .480 | 42.9% | 52.4% | 59.5% | 78.6% | 38/42 |
| | GOPT | .497 | 42.9% | **59.5%** | 61.9% | 83.3% | 37/42 |
| **Tổng 3 vòng (105)** | **PE-Core** | **.466** | **37.1%** | 55.2% | 59.0% | 80.0% | **95/105** |
| | NaFlex | .423 | 35.2% | 50.5% | 58.1% | 79.0% | 96/105 |
| | GOPT | .414 | 33.3% | 52.4% | 58.1% | **81.9%** | 94/105 |

**PE-Core là model đơn lẻ mạnh nhất** trên tổng 3 vòng (MRR .466, R@1 37.1%) và thắng ở 2/3
vòng (Round 2, Round 3) — đảo ngược hoàn toàn kết luận từ bản benchmark tiếng Việt trước đó
(PE-Core R@1 2.9%). NaFlex và GOPT bám sát phía sau, chênh lệch giữa 3 model giờ nằm trong
biên độ hợp lý (1 case ≈ 1% trên 105 case), không còn khoảng cách "một trời một vực" như bản
lỗi trước. GOPT nhỉnh hơn ở R@100 và R@5 một vài vòng nhưng không nhất quán.

## Kết quả — RRF ensemble (query tiếng Anh)

| Ensemble (tổng 3 vòng, n=105) | MRR | R@1 | R@5 | R@10 | R@100 |
|---|---:|---:|---:|---:|---:|
| PE-Core (riêng lẻ, để so sánh) | .466 | 37.1% | 55.2% | 59.0% | 80.0% |
| RRF PE+NaFlex (75/25 — nghiêng PE) | .481 | **41.9%** | 56.2% | 62.9% | 80.0% |
| RRF PE+NaFlex (50/50) | .483 | 40.0% | 59.0% | 65.7% | **81.9%** |
| **RRF cả 3 model, trọng số bằng nhau** | **.484** | 41.0% | **61.0%** | 64.8% | 81.0% |
| RRF PE+GOPT (25/75 — nghiêng GOPT) | .452 | 37.1% | 55.2% | 60.0% | 83.8% |
| RRF NaFlex+GOPT (bất kỳ trọng số) | .411-.415 | 33.3-34.3% | 49.5-51.4% | 58.1-60.0% | 79.0-82.9% |

**RRF cả 3 model (trọng số bằng nhau) và RRF PE+NaFlex (50/50) là 2 cấu hình tốt nhất**,
gần như ngang nhau và đều vượt PE-Core đơn lẻ trên mọi metric quan trọng (MRR, R@1, R@5,
R@10). Ensemble chỉ NaFlex+GOPT (không có PE-Core) luôn kém hơn PE-Core đơn lẻ — xác nhận
PE-Core vẫn đóng góp tín hiệu quan trọng nhất khi query ở đúng định dạng (tiếng Anh, ngắn).

## Kết luận

1. **Bài học phương pháp quan trọng nhất**: cách viết/dịch query ảnh hưởng đến kết quả nhiều
   hơn hẳn việc chọn model — giống hệt phát hiện trước đó ở nhánh audio
   (`docs/reports/audio-embedding-keyword-query-benchmark.md`). Sai lệch do ngôn ngữ/độ dài
   query (PE-Core R@1 từ 2.9% → 37.1%) lớn hơn nhiều so với chênh lệch thật giữa 3 model
   (37.1% vs 35.2% vs 33.3%).
2. **Không có model nào vượt trội rõ rệt** khi query đúng định dạng — PE-Core, NaFlex, GOPT
   xếp hạng sát nhau (chênh lệch trong biên độ nhiễu ~1-4 điểm % trên 105 case). **Không có
   lý do để thay PE-Core hiện tại chỉ dựa trên benchmark này.**
3. **Nếu muốn cải thiện recall, ưu tiên ensemble hơn đổi model**: RRF cả 3 model (trọng số
   bằng nhau) hoặc RRF PE+NaFlex (50/50) cho MRR/R@1/R@5/R@10 cao nhất, vượt bất kỳ model đơn
   lẻ nào. Hạ tầng cần thiết: chạy song song 2-3 text tower khi encode query (scaffolding
   `naflex_rank`/`gopt_rank` đã có sẵn trong `fuse_search_results()`, `backend/app.py`, chưa
   nối dây thật) và build sẵn cả 2-3 FAISS profile cùng row-mapping (đã có: `l2026`,
   `naflex-l2026`, `gopt-l2026`).
4. **Khuyến nghị vận hành**: nếu muốn cải thiện trải nghiệm search cho đội thi, ưu tiên đảm
   bảo **query nhập vào ở dạng ngắn, tiếng Anh, kiểu caption** (giống cách dịch ở mục trên) —
   hiệu quả hơn nhiều so với đổi hạ tầng model. Cân nhắc thêm bước dịch/rút gọn tự động ở
   frontend cho query tiếng Việt nếu muốn giữ trải nghiệm nhập liệu bằng tiếng Việt.
5. **Giới hạn cần lưu ý**: n nhỏ (27-42/vòng, 105 tổng), nên tái benchmark khi có thêm dữ
   liệu vòng thi thật. Round 3 vừa mới có đáp án đầy đủ tại thời điểm viết báo cáo này.

## Phương pháp & dữ liệu thô

- Script benchmark: `scripts/benchmarks/bench_visual_embeddings.py` (`--export-answers` để
  xuất lại ground truth từ DB).
- Script dịch query: `scripts/benchmarks/translate_visual_queries.py` — dịch caption ngắn,
  kiểm tra token qua tokenizer PE-Core, tự rút gọn nếu còn dài.
- Script gộp embedding GOPT: `scripts/embedding/assemble_gopt_batch1.py`.
- Ground truth (tiếng Việt gốc): `artifacts/official_benchmark_answers_v2.json`.
- Ground truth đã dịch (dùng để benchmark chính thức): `artifacts/official_benchmark_queries_en_v2.json`.
- Kết quả đầy đủ từng case (bản chính thức, tiếng Anh): `benchmarks/visual/visual_embedding_benchmark_2026_en.json`.
- Kết quả bản lỗi (tiếng Việt, chỉ để đối chiếu): `benchmarks/visual/visual_embedding_benchmark_2026.json`.
- Profile FAISS mới: `indexes/profiles/gopt-l2026/` (đăng ký trong
  `indexes/profiles/manifest.json`, chưa đặt làm `default`).
- Ground truth export scope: `["P1", "SOTUYEN1", "SOTUYEN2", "Round 3 - 2026"]`, nguồn DB
  production (`aic_hcmc_collab-data` volume) tại thời điểm 2026-09-16.
