# Caption Retrieval vs PE-Core 2026 — Qwen3-Embedding-4B trên caption CapRL

## Bối cảnh

Câu hỏi: caption cảnh quay (CapRL-Video-4B, tiếng Anh) đã được embed bằng
**Qwen3-Embedding-4B** và nằm trong `data/2026/embeddings/qwen3-4b-fp16/`. Liệu dùng caption
làm kênh retrieval (query tiếng Anh → caption) có giúp ích nhiều cho hệ thống hiện tại — vốn
dùng **PE-Core-bigG-14-448** trên keyframe — hay không? Báo cáo này so sánh trực tiếp hai kênh
trên cả 3 vòng 2026 (kèm vòng Test/P1), rồi thử các cách fusion caption + PE-Core để đo mức
đóng góp thực tế.

## Dữ liệu & phương pháp

- **Query**: 137 case tiếng Anh trong `artifacts/official_benchmark_queries_en_v2.json`
  (`query_language: English (caption-style, meta-commentary stripped)`, dịch bởi
  `gemini-3.1-flash-lite`) — Test/P1 (32), Round 1 (27), Round 2 (36), Round 3 (42). Mỗi câu
  TRAKE đã tách thành từng case theo mốc sự kiện.
- **Caption branch**: 103.086 caption (48.796 shot-window) trên 873 video 2026
  (`data/2026/captions/shot_captions.parquet`), document embed **không prompt**, query embed
  với `Instruct: {INSTRUCTION}\nQuery: `, last-token pooling, L2 normalize — đúng recipe của
  kernel Kaggle (`notebooks/kaggle_qwen3_caption_embed.py`). Tìm kiếm brute-force dot-product
  top-1000.
- **PE-Core branch**: query tiếng Anh, text tower PE-Core-bigG-14-448, FAISS Flat
  `indexes/profiles/l2026/flat.ip` (268.254 keyframe / 873 video), top-1000.
- **Metric** (`tolerance_seconds = 10`, `k = 1000`), giống hệt các benchmark cũ:
  - *moment*: cùng video và khoảnh khắc trúng đích — PE-Core là keyframe lệch ≤10 s; caption là
    target rơi trong `[window_start − 10 s, window_end + 10 s]` của caption được lấy.
  - *video*: video đích (bất kỳ target video nào) xuất hiện lần đầu ở rank nào trong top-1000.
- **Tái lập**: script `scripts/benchmarks/bench_caption_retrieval.py` (chạy
  `uv run python -m scripts.benchmarks.bench_caption_retrieval`), kết quả ghi vào
  `benchmarks/visual/caption_vs_pe_core_2026.json`.
  - PE-Core tái lập **137/137 case khớp tuyệt đối** với
    `benchmarks/visual/visual_embedding_benchmark_2026_en.json`.
  - Caption tái lập **đúng metric tổng** của eval Kaggle (`eval_qwen3-4b-fp16.json`): MRR
    .3337, R@1 25.55%, R@5 41.61%, R@10 50.36%, R@100 71.53%, 131/137 hit. Rank từng case khớp
    93/137 (window) và 107/137 (video); phần lệch là nhiễu bf16 CPU (local) so với fp16
    (Kaggle) giữa các caption gần điểm nhau — đa số lệch 1–6 rank, không đổi kết luận.
- **Không đánh giá Qwen3-Embedding-8B**: `data/2026/embeddings/qwen3-8b-fp16-shard0` mới có
  20.617/103.086 dòng (107/873 video, run shard 0/5) — chưa đủ corpus để so sánh.

## Kết quả

### Caption đơn lẻ thua PE-Core ở tầng khoảnh khắc

Tổng hợp Round 1-3 - 2026 (105 case):

| Kênh | MRR | R@1 | R@5 | R@10 | R@100 | hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PE-Core** (keyframe ±10 s) | **.466** | **37.1%** | **55.2%** | **59.0%** | **80.0%** | 95/105 |
| Qwen3-4B caption (window ±10 s) | .295 | 21.9% | 36.2% | 45.7% | 70.5% | **100/105** |

Caption tìm được *nhiều case hơn* trong top-1000 (100 vs 95) nhưng xếp hạng đúng khoảnh khắc
kém hơn hẳn (MRR .295 vs .466). Theo từng vòng:

| Vòng (case) | PE-Core MRR / R@1 / hits | Caption MRR / R@1 / hits |
| --- | --- | --- |
| Test (32) | .457 / 34.4% / 31 | **.461 / 37.5% / 31** |
| Round 1 (27) | **.307 / 22.2% / 24** | .199 / 14.8% / **26** |
| Round 2 (36) | **.503 / 38.9% / 33** | .307 / 25.0% / 33 |
| Round 3 (42) | **.538 / 45.2% / 38** | .346 / 23.8% / **41** |
| **R1-3 (105)** | **.466 / 37.1% / 95** | .295 / 21.9% / **100** |

Lưu ý metric caption *rộng hơn* PE-Core (cả window được tính là trúng, không chỉ ±10 s quanh
một frame), vậy mà caption vẫn thua — chất lượng xếp hạng thực tế còn kém hơn con số này.

### Ở tầng video, PE-Core cũng xếp hạng tốt hơn, caption chỉ bù recall

| Kênh video-level | MRR | R@1 | R@5 | R@10 | R@100 | hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PE-Core** | **.612** | **53.3%** | **69.5%** | 73.3% | 92.4% | 104/105 |
| Caption | .424 | 33.3% | 52.4% | 61.0% | 92.4% | **105/105** |

PE-Core không chỉ định vị đúng khoảnh khắc mà còn định vị đúng *video* tốt hơn caption
(MRR video .612 vs .424). Caption không thay thế được kênh nào của PE-Core.

### Fusion: RRF giúp recall sâu, filter/boost đều hại

| Biến thể fusion (R1-3) | MRR | R@1 | R@10 | R@100 | hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| PE-Core đơn lẻ (moment) | **.466** | **37.1%** | 59.0% | 80.0% | 95/105 |
| RRF video, PE weight 0.75 | .589 | 48.6% | **83.8%** | **99.0%** | 105/105 |
| RRF video, PE weight 0.25 | .543 | 45.7% | 70.5% | 99.0% | 105/105 |
| RRF video, equal | .540 | 42.9% | 79.0% | 99.0% | 105/105 |
| PE lọc theo top-10 video của caption | .392 | 33.3% | 48.6% | 58.1% | 61/105 |
| PE lọc theo top-20 video của caption | .421 | 35.2% | 52.4% | 66.7% | 72/105 |
| PE + boost điểm top-10 video caption (w=1) | .450 | 37.1% | 59.0% | 80.0% | 95/105 |
| PE + boost điểm top-10 video caption (w=4) | .401 | 34.3% | 50.5% | 71.4% | 95/105 |

- **RRF ở tầng video** (caption + PE) là biến thể duy nhất có lợi, nhưng chỉ ở recall sâu:
  R@10 video tăng 73.3% → 83.8%, R@100 92.4% → 99.0% (cả 105/105 case có video đích trong
  top-100). Đổi lại MRR video giảm nhẹ (.612 → .589) vì thứ tự top-1 bị caption pha loãng.
- **Dùng caption làm prior cứng cho PE** (lọc theo top-N video) mất rất nhiều recall
  (top-10 chỉ còn 61/105 case trúng) — caption định vị video đúng chưa đủ tin cậy để lọc.
- **Boost điểm** gần như trung tính ở w=1 và hại ở w=4; không cải thiện gì so với PE-Core đơn lẻ.

### Bù trừ case-level

| Vòng | chỉ PE-Core trúng | chỉ caption trúng | cả hai | không ai | hợp (union) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test (32) | 1 | 1 | 30 | 0 | 32 (100%) |
| Round 1 (27) | 1 | 3 | 23 | 0 | 27 (100%) |
| Round 2 (36) | 2 | 2 | 31 | 1 | 35 (97.2%) |
| Round 3 (42) | 1 | 4 | 37 | 0 | 42 (100%) |
| **R1-3 (105)** | 4 | 9 | 91 | 1 | **104 (99.0%)** |
| Toàn bộ 137 | 5 | 10 | 121 | 1 | **136 (99.3%)** |

Hai kênh bù trừ tốt ở mức *oracle*: nếu có cách chọn đúng kênh, hợp của hai kênh phủ
136/137 case (chỉ 1 case Round 2 trượt cả hai). Nhưng đây là oracle — không có cách fusion
đơn giản nào ở trên biến 9–10 case "chỉ caption trúng" thành kết quả thực; cần một verifier
(VLM/agent) chấm lại candidate mới khai thác được.

## Vì sao caption không giúp nhiều (phân tích nguyên nhân)

Chạy `uv run python -m scripts.benchmarks.bench_caption_retrieval --diagnose` (kết quả:
`artifacts/caption_vs_pe_core/diagnosis.json`) để tách lỗi thành 2 tầng: **chọn đúng video**
và **chọn đúng khoảnh khắc trong video đó**.

### Nút thắt nằm ở nhận diện video, không phải chi tiết khoảnh khắc

Toàn bộ 137 case:

| Chỉ số | Caption Qwen3-4B | PE-Core |
| --- | ---: | ---: |
| Video đúng ở top-1 | 38.0% | **55.5%** |
| Video đúng ở top-5 | 57.7% | **67.9%** |
| Video đúng ở top-10 | 65.7% | **72.3%** |
| Shot đúng trong chính video đúng, top-1 | 49.6% | **58.4%** |
| Shot đúng trong video, top-5 | **81.8%** | 78.8% |
| Shot đúng trong video, top-10 | **87.6%** | 86.1% |
| Khoảnh khắc top-1 **khi video đã top-1** | **67.3%** (52 case) | 65.8% (76 case) |
| Khoảnh khắc top-5 khi video đã top-1 | **84.6%** | 84.2% |
| Khoảnh khắc top-10 khi video đã top-1 | **90.4%** | 88.2% |

(Hai dòng "Shot đúng trong video" xếp hạng target window chỉ trong số caption của chính video
đích — tức trần trên nếu chọn video hoàn hảo. Ba dòng cuối tính trên các tập case khác nhau:
52 case caption chọn đúng video top-1, 76 case PE chọn đúng video top-1.)

Đọc bảng này: **một khi đã xác định đúng video, caption xếp hạng khoảnh khắc ngang bằng
PE-Core** (thậm chí nhỉnh hơn ở top-1/top-5/top-10 trên tập case tương ứng). Toàn bộ khoảng
cách nằm ở chỗ caption chỉ chọn đúng video top-1 38% so với 55.5% của PE-Core. Nói cách khác,
vấn đề không phải "caption thiếu chi tiết" mà là "chi tiết trong caption không phân biệt được
video này với video khác".

### Caption của các video khác nhau gần như đồng nghĩa

- 85/137 case top-1 caption thuộc **video khác**. Các caption sai đó khớp từ khóa với query
  trung bình **0.519** — gần bằng caption đúng (0.558); 45/85 trường hợp caption sai còn khớp
  *nhiều hơn* caption đúng.
- Mẫu điển hình: query nấu ăn ("asparagus spear chạm dầu") → top-1 là caption của video khác
  cũng tả "chảo dầu đang sôi"; query "năm người chơi gần một con vật vàng" → top-1 là gameshow
  ngoài trời của video khác. Các chương trình truyền hình lặp lại bối cảnh (studio, chợ, lớp
  học, nấu ăn) nên mô tả text gần trùng nhau.
- Hệ quả: thêm chi tiết vào caption **không tự động tăng độ phân biệt** — chi tiết chỉ hữu ích
  nếu nó khác biệt so với các video còn lại.

### Không phải lỗi của embedding model

Chạy thêm baseline **TF-IDF thuần từ khóa** trên đúng bộ caption đó (trong `--diagnose`):

| Kênh (137 case) | moment MRR | moment R@1 | moment R@100 | video MRR | video R@1 | video R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B embedding | **.334** | **25.6%** | **71.5%** | .472 | 38.0% | **92.7%** |
| TF-IDF từ khóa | .296 | 18.3% | 69.3% | **.476** | **39.4%** | 87.6% |

Hai kênh tương đương nhau (video top-1 thậm chí TF-IDF nhỉnh hơn). Nghĩa là trần chất lượng
nằm ở **bản thân bộ caption** (nội dung gần trùng giữa các video), không phải năng lực ngữ
nghĩa của encoder — đổi sang embedding mạnh hơn cũng khó cải thiện.

### Chi tiết bị pha loãng và độ phân giải thô (yếu tố thứ cấp)

- Caption dài trung bình **756 token** (p90 992, tối đa 2049; 54 caption bị cắt ở 2048), query
  trung bình **21 token** — gấp ~36 lần. Một vector 2560 chiều phải "nén" cả đoạn mô tả dài
  gồm bối cảnh, logo, timecode; chi tiết đặc trưng chỉ là 1–2 câu trong đó.
- Coverage từ nội dung của query trong caption đúng chỉ **58.3%** trung bình; chỉ **15.3%**
  case đạt ≥80% (8% case ≤30%) — CapRL thường mô tả cảnh/hoạt động chung chứ không luôn ghi
  lại chi tiết đặc trưng (màu, số lượng, chữ trên màn hình).
- Độ phân giải: **1 caption/shot** (48.8k unit) so với **~5.5 keyframe/shot** của PE-Core
  (268k). Trong cùng video, caption top-1 49.6% so với PE 58.4% — đây là phần giải thích
  khoảng cách top-1 nhỏ còn lại, không phải khoảng cách video.

### Caption thắng ở đâu

Nhìn 10 case chỉ caption tìm được: tập trung ở query có **chữ trên màn hình** (slide địa lý,
bài thơ, câu hỏi đề thi sinh học) và **chuỗi động tác dài** (lion/dragon dance TRAKE — caption
mô tả thứ tự sự kiện tốt hơn keyframe tĩnh). PE-Core thắng ở các **close-up thị giác đặc
trưng** (nấu ăn, thủ công). Đây là lý do union của hai kênh đạt 136/137 — caption bù trừ được
ở nhóm query text-heavy, nhưng cần bước verify mới khai thác được.

## Kết luận

1. **Caption + Qwen3-Embedding-4B không giúp nhiều.** Ở dạng đơn lẻ nó thua PE-Core rõ rệt
   (MRR .295 vs .466; R@1 21.9% vs 37.1%) dù metric còn rộng hơn. Ở tầng video nó cũng thua
   (MRR .424 vs .612). Nguyên nhân chính (mục trên): **chọn sai video** (top-1 38.0% vs
   55.5%) vì caption các video gần như đồng nghĩa; một khi đã đúng video thì caption xếp hạng
   khoảnh khắc ngang PE-Core. Chi tiết nhiều không tạo ra *độ phân biệt*, và TF-IDF thuần từ
   khóa cho kết quả tương đương — nên không phải giới hạn của encoder.
2. **Không nên thay hay lọc PE-Core bằng caption**: mọi biến thể lọc theo caption đều mất
   recall; boost điểm trung tính hoặc hại.
3. **Điểm có ích duy nhất**: RRF video PE+caption (PE weight ~0.75) kéo recall sâu lên
   (R@10 73% → 84%, R@100 92% → 99%, đủ 105/105 case) với giá phải trả là MRR video giảm nhẹ.
   Chỉ đáng cân nhắc nếu mục tiêu là *tập candidate video* cho bước sau (agent/verifier), không
   phải trả kết quả trực tiếp cho người dùng.
4. 10 case chỉ caption tìm thấy khoảnh khắc (R1: 3, R3: 4, R2: 2, Test: 1) cho thấy tín hiệu
   caption là *bù trừ*, không phải vô dụng — nhưng muốn dùng phải có bước verify/rerank bằng
   model khác, ví dụ agent hiện có.

**Khuyến nghị**: giữ PE-Core làm kênh chính cho cả moment lẫn video. Nếu cần tăng recall sâu
ở tầng video, thêm caption như một nhánh RRF weight thấp; còn caption cho moment-level chỉ nên
đưa vào như candidate generator cho agent, không đưa trực tiếp vào kết quả. Muốn caption thực
sự mạnh hơn thì phải tăng *độ phân biệt* chứ không phải độ chi tiết: ví dụ embed theo
câu/segment + MaxSim thay vì 1 vector cho cả shot, hoặc lọc bỏ phần boilerplate (logo,
timecode, "the video opens with...") trước khi embed — còn đổi sang embedding model mạnh hơn
trên cùng cách biểu diễn khó có đột phá, vì TF-IDF đã tương đương.

## Caveat

- Metric caption (window ±10 s) rộng hơn metric PE-Core (keyframe ±10 s) — lợi cho caption,
  nên khoảng cách thực tế có thể còn lớn hơn.
- Local chạy Qwen3-4B bằng **bf16 CPU** (GPU đang bận) trong khi Kaggle chạy fp16; metric tổng
  khớp tuyệt đối nhưng rank từng case lệch nhẹ (xem mục Tái lập).
- Caption chỉ phủ dataset 2026 (873 video); PE-Core `l2026` cũng vậy nên so sánh hợp lệ.
- Qwen3-Embedding-8B chưa so được (mới có 107/873 video); nếu sau này có đủ shard, nên chạy
  lại script với `--caption-dir data/2026/embeddings/qwen3-8b-fp16`.

## Tái lập

```bash
# lần đầu tải Qwen3-Embedding-4B (~7.6 GB) và encode 137 query trên CPU (~17 min)
.venv/bin/python -m scripts.benchmarks.bench_caption_retrieval

# các lần sau dùng cache top-1000 trong artifacts/caption_vs_pe_core/ (gitignored)
.venv/bin/python -m scripts.benchmarks.bench_caption_retrieval --reuse

# phân tích nguyên nhân (video vs moment, token, lexical coverage, TF-IDF baseline)
.venv/bin/python -m scripts.benchmarks.bench_caption_retrieval --diagnose
```

Kết quả: `benchmarks/visual/caption_vs_pe_core_2026.json`; chẩn đoán:
`artifacts/caption_vs_pe_core/diagnosis.json` (gitignored). Self-test thuần logic:
`--self-test`.
