# Caption Cleaning Probe 2026 — Làm caption khác nhau để hết nhầm video

## Câu hỏi

Nút thắt đã biết là **các video trông giống nhau**: caption của video khác nhau mô tả gần như
cùng một bối cảnh (studio, nấu ăn, lớp học...), nên text retrieval chọn nhầm video (top-1 chỉ
38%). Thí nghiệm này hỏi: nếu **viết lại caption để nó khác biệt rõ với các caption gần giống
thuộc video khác**, thì recall có tăng không?

Caption gốc **không bị sửa** — chỉ đọc; bản sạch lưu ở artifact riêng.

## Thiết kế

- **Pool 40 case / 3.345 window** như probe chunking (30 case tệ nhất rank>100 + 10 control
  rank≤5), gồm cả target lẫn các window "na ná" từ video khác.
- **kNN không dùng đáp án**: với mỗi window, lấy 5 caption gần nhất (cosine trên embedding 4B
  sẵn có) thuộc **video khác**.
- **3 biến thể clean bằng DeepSeek flash**:
  - `stripped` — bỏ boilerplate (logo, timecode, tên chương trình, "the video opens with...").
  - `distinctive` — viết lại ≤60 từ, chỉ giữ chi tiết đặc trưng, không bịa.
  - `contrastive` — viết lại ≤60 từ, **nhấn mạnh điểm khác với 5 caption gần giống kia**.
- Embedding **Qwen3-Embedding-0.6B** (screening, cùng model cho mọi biến thể + bản gốc), chấm
  MaxSim trên đúng pool; thêm `first_sentence` làm baseline "ngắn đi" để tách hiệu ứng độ dài.

## Kết quả

### 1. Độ giống nhau cross-video (cosine giữa caption của các video khác nhau)

| Biến thể | mean | cặp >0.85 | cặp >0.9 |
| --- | ---: | ---: | ---: |
| original | .408 | 0.71% | 0.13% |
| first_sentence (ngắn đi) | .488 | 1.20% | 0.43% |
| stripped | .400 | 0.55% | 0.09% |
| distinctive | .442 | 0.07% | 0.02% |
| **contrastive** | .465 | **0.04%** | **0%** |

Ngắn đi làm caption **giống nhau hơn** (chi tiết mất, phần chung lộ ra). `stripped` gần như
không đổi. Chỉ `distinctive`/`contrastive` mới **xoá gần hết cặp gần trùng** (giảm 10–18 lần).

### 2. Moment recall — 30 case tệ (oracle pool)

| Biến thể | MRR | R@1 | R@5 | R@10 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | .041 | 0% | 0% | 10% | 40% | 67% |
| stripped | .041 | 0% | 0% | 10% | 30% | 73% |
| first_sentence | .130 | 3% | 17% | 37% | 63% | 83% |
| distinctive | .138 | 7% | 17% | 17% | 50% | 87% |
| **contrastive** | **.163** | **10%** | **20%** | **27%** | 40% | **90%** |

### 3. Video-level (xếp video theo caption tốt nhất, 30 case tệ)

| Biến thể | MRR | R@1 | R@5 | R@10 | R@25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | .251 | 7% | 37% | 67% | 97% |
| stripped | .236 | 7% | 47% | 63% | 93% |
| first_sentence | .307 | 10% | 53% | 80% | 97% |
| distinctive | .382 | 20% | 60% | 73% | 100% |
| **contrastive** | **.395** | **23%** | **70%** | **87%** | 97% |

**Video top-1 tăng 7% → 23%, top-5 37% → 70%** — đúng cái nút thắt "nhầm video".

### 4. Case dễ (10 control, realistic pool) — giá phải trả

| Biến thể | MRR | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: | ---: |
| original | **.803** | **70%** | **100%** | **100%** |
| stripped | .742 | 60% | 100% | 100% |
| first_sentence | .437 | 30% | 70% | 70% |
| distinctive | .469 | 30% | 60% | 90% |
| contrastive | .443 | 30% | 70% | 80% |

Giống mọi phương pháp trước: đổi representation lấy recall thì mất precision top-1 ở case dễ.
Video-level control của contrastive còn 60% top-1 (gốc 90%).

### 5. Trộn điểm & union candidate (realistic pool, không oracle)

| Phương pháp | BAD MRR | BAD R@1 | BAD R@5 | BAD R@10 | BAD R@50 | CTRL MRR | CTRL R@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | .031 | 0% | 0% | 7% | 60% | **.803** | **70%** |
| contrastive | **.132** | **10%** | **13%** | **17%** | **70%** | .443 | 30% |
| mean(original, contrastive) | .106 | 7% | 10% | 10% | 60% | .610 | 40% |
| 0.7·original + 0.3·contrastive | .054 | 0% | 7% | 10% | 60% | .762 | **60%** |

**Union candidate** (lấy top-N của từng kênh rồi hợp lại — không mất precision của kênh gốc):

| N | original | original+contrastive | cả 3 | control (cả 3) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 0/30 | 4/30 | **5/30** | 10/10 |
| 10 | 2/30 | 6/30 | 6/30 | 10/10 |
| 25 | 9/30 | 13/30 | **14/30** | 10/10 |
| 50 | 18/30 | 21/30 | **22/30** | 10/10 |

## Xác nhận bằng Qwen3-Embedding-4B

Chạy lại đúng pool đó với 4B (fp16, GPU local; `--stage score-4b`, output
`cleaning_probe_4b.json`). Lưu ý phương pháp: **pool được dựng từ chính các case 4B gốc xếp hạng
>100 và top-50 toàn cục theo 4B gốc**, nên cột `original` ở đây là worst-case (mọi đối thủ mạnh
nhất đều có mặt), không so được trực tiếp với con số 0.6B; chỉ nên so *giữa các biến thể* trong
cùng bảng.

| Biến thể (4B) | BAD MRR | BAD R@1 | BAD R@5 | BAD R@10 | BAD R@25 | BAD R@50 | CTRL MRR | CTRL R@1 | CTRL R@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | .017 | 0% | 0% | 0% | 0% | 0% | **.728** | **60%** | **100%** |
| contrastive 60 từ | **.103** | **6.7%** | 6.7% | **10%** | 33.3% | **86.7%** | .382 | 20% | 50% |
| contrastive_long 150 từ | .069 | 3.3% | 3.3% | 6.7% | 33.3% | 80% | .613 | 50% | 80% |
| original + long (mean) | .028 | 0% | 3.3% | 3.3% | 6.7% | 33.3% | .77 | 60% | 100% |

Video-level (30 case tệ): original MRR .164 / R@1 6.7%; contrastive 60w .377 / 20%; long .337 /
**23.3%**. Độ giống nhau cross-video (4B): original nền ~.49, contrastive 60w .508 (>0.85: 0.06%),
long .536 (>0.85: 0.25%).

**Kết luận 4B giống hệt 0.6B**: 60 từ tối đa recall trên case tệ, 150 từ giữ precision ở case dễ
tốt hơn; không có biến thể nào thắng tuyệt đối. Xu hướng method-ranking ổn định qua cả hai model,
nên kết quả 0.6B dùng để screening là đáng tin.

**So sánh trực tiếp 0.6B vs 4B cho `contrastive_long` (150 từ):**

| Metric | 0.6B | 4B |
| --- | ---: | ---: |
| BAD oracle MRR | .074 | .069 |
| BAD oracle R@1 / R@5 / R@10 | 3% / 3% / 7% | 3% / 3% / 7% |
| BAD oracle R@25 / R@50 | 33% / 83% | 33% / 80% |
| CTRL realistic MRR / R@1 | .556 / 40% | .613 / 50% |
| BAD video MRR / R@1 / R@5 | .249 / 10% / 40% | **.337 / 23% / 43%** |
| Case tệ được cứu vào top-25 | 3 | **10** (chỉ trùng 2) |

→ Ở tầng khoảnh khắc hai model gần như trùng khớp; 4B nhỉnh hơn ở **tầng video** và cứu được
nhiều case hơn (chủ yếu nhóm lion-dance TRAKE), nhưng phần lớn lợi thế đó đến từ pool bias (pool
dựng từ lỗi của 4B gốc) — cần chạy full-corpus để có con số công bằng.

## Chi phí clean toàn bộ 103.086 caption (DeepSeek)

Đo thực tế trên 3.345 window (đã gồm retry/fallback): **1.659 input + 302 output token/window**
(batch 4, 5 neighbor, excerpt 600 ký tự).

| Khoản cho 103.086 window | Token | Giá off-peak | Giá peak |
| --- | ---: | ---: | ---: |
| Input (cache miss, $0.15/$0.30 mỗi 1M) | 171,0M | $25.7 | $51.3 |
| Output ($0.60/$1.20 mỗi 1M) | 31,1M | $18.7 | $37.3 |
| **Tổng** | | **≈ $44** | **≈ $89** |

- Off-peak: mọi giờ trừ 01:00–04:00 và 06:00–10:00 UTC (thứ 2–6).
- Giảm chi phí: 3 neighbor hoặc excerpt 300 ký tự → input còn ~125M (~$33 off-peak); prompt
  caching hầu như không giúp vì mỗi call một prompt khác.
- Thời gian: ~26k call; ở 8 worker đo được ~1,5 call/s → ~5h (tăng lên 16–24 worker có thể còn
  2–3h, concurrency limit 2.500).
- Sau khi clean: embed 4B trên Kaggle 2×T4 fp16 cho 103k × ~220 token ≈ 22,7M token ≈ **2,7h**.
- Lưu riêng trong `./data` (không đụng caption gốc): cleaned parquet + thư mục embedding riêng
  (ví dụ `data/2026/captions/cleaned/contrastive_long.parquet`,
  `data/2026/embeddings/qwen3-4b-contrastive-long/`).

## So với PE-Core

### Corpus-level (137 case, top-1000)

| Kênh | moment MRR | R@1 | R@5 | R@10 | R@100 | hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PE-Core** | **.464** | **36.5%** | **54.7%** | **59.9%** | **80.3%** | 126/137 |
| Caption 4B (gốc) | .334 | 25.5% | 41.6% | 50.4% | 71.5% | 131/137 |
| PE ∪ caption (oracle) | — | 99.3% | 99.3% | 99.3% | 99.3% | 136/137 |

| Kênh video-level | MRR | R@1 | R@5 | R@10 | R@100 | hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PE-Core** | **.619** | **55.5%** | **67.9%** | **72.3%** | 92.7% | 136/137 |
| Caption 4B (gốc) | .474 | 38.0% | 57.7% | 65.7% | 92.7% | 137/137 |

Trên 30 case mà caption gốc xếp >100: PE-Core tìm được 23/30 (R@25 33%, R@100 47%); hợp PE ∪
caption phủ 29/30 (only-PE 5, only-caption 6, neither 1).

### Pool-level (cùng 3.345 window, 4B)

| Phương pháp | BAD MRR | BAD R@1 | BAD R@5 | BAD R@10 | BAD R@25 | BAD R@50 | CTRL MRR | CTRL R@1 | video MRR | video R@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PE-Core** | **.270** | **10%** | **46.7%** | **56.7%** | **70%** | **90%** | .598 | 40% | **.572** | **43.3%** |
| Caption gốc 4B | .017 | 0% | 0% | 0% | 0% | 0% | **.728** | **60%** | .164 | 6.7% |
| Caption clean 60 từ | .103 | 6.7% | 6.7% | 10% | 33% | 87% | .382 | 20% | .377 | 20% |
| Caption clean 150 từ | .069 | 3.3% | 3.3% | 6.7% | 33% | 80% | .613 | 50% | .337 | 23.3% |

**Hợp candidate (realistic pool, 30 case tệ):**

| N | PE | PE + clean 60w | PE + clean 150w | control |
| --- | ---: | ---: | ---: | ---: |
| 5 | 10/30 | 10/30 | 10/30 | 10/10 |
| 10 | 12/30 | 12/30 | **13/30** | 10/10 |
| 25 | 14/30 | 15/30 | **17/30** | 10/10 |
| 50 | 19/30 | 21/30 | **22/30** | 10/10 |

**Kết luận so sánh**: PE-Core đơn lẻ vẫn mạnh hơn caption (kể cả sau cleaning) ở mọi mức — trong
pool, PE R@5 46.7% so với 6.7%/3.3% của caption sạch; corpus-level PE R@1 36.5% vs 25.5%.
Cleaning biến caption từ "vô dụng trong pool" (0% top-50) thành kênh **bổ trợ** có giá trị: hợp
PE + clean 150w tăng candidate recall ở N=25 (14→17/30) và N=50 (19→22/30) mà không mất control.
Caption không thay thế được PE; vai trò hợp lý vẫn là candidate/verify bù trừ.

## Kết luận

1. **Giả thuyết được xác nhận**: viết lại caption để khác với các caption gần giống của video
   khác (`contrastive`) cải thiện **cả moment recall lẫn nhận diện video** trên nhóm case tệ —
   video top-1 7%→23%, top-5 37%→70%; moment MRR .041→.163, R@1 0→10%, R@50 67→90%; cặp caption
   gần trùng (>0.85) giảm từ 0.71% xuống 0.04%.
2. **Bỏ boilerplate một mình vô ích** (`stripped` ≈ original) — vấn đề không phải logo/timecode
   mà là **nội dung mô tả chung chung giống nhau giữa các video**.
3. **Ngắn đi không đủ**: `first_sentence` cũng cải thiện (nhờ bỏ bớt phần chung) nhưng kém
   contrastive ở mọi mức recall nhỏ (R@1 3% vs 10%, R@50 83% vs 90%).
4. **Giá phải trả vẫn là precision ở case dễ** (control R@1 70%→30%). Trộn 0.7 gốc + 0.3
   contrastive giữ control gần như nguyên (R@1 60%) mà vẫn nhích được case tệ; **union candidate**
   thì không mất gì: top-25 hợp 3 kênh phủ 14/30 case tệ (gốc 9/30), control vẫn 10/10.

**Hướng tiếp theo hợp lý**:
- Dùng `contrastive` như **kênh candidate/verify bổ sung** (union với caption gốc), không thay
  caption gốc — không mất precision, tăng recall rõ.
- Nếu muốn apply toàn corpus: clean contrastive cho 103k window cần ~26k call DeepSeek
  (ước tính ~100–175M input token, ~10–15h với 8 luồng, vài chục USD) rồi re-embed 4B trên
  Kaggle — đắt hơn chunking nhưng đây là hướng duy nhất tới giờ tấn công đúng nút thắt video.
  Có thể giảm chi phí: chỉ clean các window tham gia candidate (top-K của các query) hoặc dùng
  model rẻ hơn.
- Xác nhận lại bằng Qwen3-4B (Kaggle) trên pool nhỏ trước khi mở rộng.

## Artifacts & chi phí

- Script: `scripts/benchmarks/clean_captions_probe.py` (+`--self-test`), tái dùng
  `SmallEncoder`/pool của `probe_caption_chunking.py`.
- Dữ liệu: `artifacts/caption_vs_pe_core/probe/{caption_neighbors,cleaned_captions_probe,cleaning_probe,cleaning_combo_probe,cleaning_union_probe}.json`
  + vector cache `cleaned_vectors_*.npy` (gitignored).
- DeepSeek: ~4.400 call, 12.5M input + 5.7M output token (main + repair), ước tính <$10.
- Thời gian: kNN ~1', clean ~55', score ~10'.
