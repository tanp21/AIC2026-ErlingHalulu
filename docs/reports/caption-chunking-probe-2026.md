# Caption Chunking Probe 2026 — Caption quá dense có phải nguyên nhân?

## Câu hỏi

Caption CapRL dài trung bình **756 token** (p90 992) nhưng bị nén vào **1 vector** mỗi shot.
Giả thuyết: chi tiết khoảnh khắc bị pha loãng, nên tách caption thành câu / token chunk /
**core caption** (LLM) rồi chấm MaxSim sẽ cứu được các case mà full-caption retrieval xếp sai.

## Thiết kế probe

- **Subset 40 case** (theo `diagnosis.json`): 30 case tệ nhất (moment rank >100, gồm 6 case
  ngoài top-1000) + 10 case control (rank ≤5).
- **Pool/case**: top-50 full-caption 4B + top-20 window của chính video đích + toàn bộ target
  window. "Realistic pool" = 70 window đầu; "oracle pool" = realistic + target window (đo trần).
  Tổng 3.345 window duy nhất.
- **Biến thể document**: `full` (baseline), `sentence`, `chunk128`, `chunk256` (overlap 32),
  `core` = 2–5 core caption tự chứa do **DeepSeek flash** tách (3.345 window, 436 call,
  median 4 core/window, 104 window phải fallback sentence).
- **Scoring**: MaxSim (max cosine) và top-2 mean; model **Qwen3-Embedding-0.6B** (GPU local),
  full baseline tính lại cùng model để so công bằng. Lexical probe chạy thêm trên **toàn bộ
  103.086 caption / 137 case** bằng TF-IDF (không cần model).

## Kết quả

### 1. Lexical (TF-IDF, toàn corpus 137 case): chunking câu làm TỆ ĐI

| Kênh | | MRR | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF full caption | moment | **.296** | **18%** | **43%** | **51%** | **62%** | **67%** | **69%** |
| TF-IDF sentence chunks | moment | .176 | 10% | 26% | 33% | 43% | 54% | 64% |
| TF-IDF full caption | video | **.476** | **39%** | **55%** | **66%** | **77%** | **82%** | **88%** |
| TF-IDF sentence chunks | video | .343 | 25% | 45% | 58% | 68% | 78% | 88% |

Cắt câu rồi khớp từ khóa chỉ tạo ra hàng triệu mảnh ngắn khớp hời hợt — term-level chunking
không phải là câu trả lời. (2.39M câu vì caption vốn nhiều câu ngắn.)

### 2. Dense 0.6B — oracle pool (target chắc chắn có mặt), 30 case tệ

| Biến thể | MRR | R@1 | R@5 | R@10 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | .041 | 0% | 0% | 10% | 40% | 67% |
| `sentence` | **.159** | 3.3% | **33.3%** | **43.3%** | **63.3%** | **90%** |
| `core` | **.159** | **10%** | 16.7% | 26.7% | 60% | **90%** |
| `chunk128` | .062 | 0% | 6.7% | 10% | 43.3% | 80% |
| `chunk256` | .045 | 0% | 3.3% | 13.3% | 40% | 73.3% |

**Khi target nằm trong candidate set, chunk/core-MaxSim nâng R@25 từ 40% → 60–63%, R@50 từ
67% → 90%, MRR gấp ~4 lần.** Đây là bằng chứng trực tiếp cho giả thuyết density: chi tiết có
trong caption nhưng bị vector 756-token pha loãng.

9 case được "cứu" rõ rệt (oracle, full → core): kéo lưới lúc bình minh (48→12), lân đáp
xuống 4 chân (55→20), châm bếp bằng dụng cụ (51→7), hai người nằm trên xe máy với 2 vòng đỏ
(39→20), bánh tím nhân giá đỗ (55→16)… — đều là query chi tiết nhỏ, đúng nhóm bị dilution.

### 3. Dense 0.6B — realistic pool (không oracle)

**30 case tệ (realistic pool):**

| Biến thể | MRR | R@1 | R@5 | R@10 | R@25 | R@50 | hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | .031 | 0% | 0% | 7% | 30% | 60% | 23/30 |
| `sentence` | .069 | 0% | 10% | 20% | **40%** | **70%** | 23/30 |
| `chunk128` | .050 | 0% | 7% | 10% | 27% | 67% | 23/30 |
| `chunk256` | .034 | 0% | 0% | 10% | 33% | 63% | 23/30 |
| `core` | **.098** | **7%** | 7% | 10% | 37% | 67% | 23/30 |

**10 case control (realistic pool):**

| Biến thể | MRR | R@1 | R@5 | R@10 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | **.803** | **70%** | **100%** | **100%** | 100% | 100% |
| `sentence` | .503 | 30% | 80% | 100% | 100% | 100% |
| `chunk128` | .574 | 30% | 90% | 90% | 100% | 100% |
| `chunk256` | .733 | 60% | 100% | 100% | 100% | 100% |
| `core` | .570 | 40% | 80% | 90% | 100% | 100% |

- Trên case tệ: `core` thắng (MRR ×3, R@1 0→7%, rescued 5 case vào top-25, chỉ 3 case tụt),
  `sentence` thắng recall sâu (rescued 6, tụt 3; R@25 30→40%, R@50 60→70%).
- Trên case control: **chunking lấy đi precision top-1** (R@1 70% → 30–40%, R@5 100% → 80–90%).
  Mix điểm full + λ·chunk với λ nhỏ (0.1–0.2) giữ được control nhưng không đủ sức cứu case tệ;
  λ lớn cứu được case tệ nhưng phá control. Trên 40 case này không có một trọng số đơn nào tốt
  cho cả hai nhóm (bảng combo đầy đủ ở Phụ lục).
- `chunk128/256` gần như trung tính → phải cắt **đủ nhỏ (câu/core) + semantic MaxSim**, cắt cơ
  học theo token không giúp.

### 4. Giới hạn candidate generation

7/30 case tệ không có target window ngay trong pool (top-50 full + top-20 video đích), nên mọi
cách re-rank đều vô hiệu với chúng. Muốn R@100 tiến tới 100% thì phải **chunk toàn corpus**
(Phase 2) chứ không chỉ re-rank candidate.

### 5. Thử thêm: retrieval hai tầng video-first (miễn phí, dùng vector 4B cache)

Vì nút thắt chính là chọn sai video, đã thử shortlist video trước rồi mới xếp window trong đó
(không cần model mới):

| Cách xếp video (137 case) | MRR | R@1 | R@5 | R@10 | R@25 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max điểm window trong video | **.486** | **38%** | **60%** | **69%** | **85%** | **96%** |
| Centroid trung bình các window | .333 | 24% | 42% | 49% | 71% | 92% |

| Moment hai tầng (shortlist V video, xếp window bằng full score) | MRR | R@1 | R@5 | R@10 | R@25 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V=1 | .163 | 13% | 21% | 23% | 23% | 23% |
| V=10 | .224 | 18% | 28% | 35% | 38% | 43% |
| V=50 | .306 | 23% | 39% | 46% | 55% | 67% |
| V=100 | .341 | 26% | 43% | 51% | 58% | 70% |
| **Baseline (full window, không tầng)** | .334 | 26% | 42% | 50% | 58% | 72% |

Kết luận: **video-first không tạo thêm giá trị** — max-window vốn đã là aggregator video tối ưu,
shortlist chỉ mất mát (V=100 mới ngang baseline). Centroid trung bình còn tệ hơn vì làm nhoè chi
tiết. Muốn cải thiện video ID phải thêm tín hiệu khác, không phải tái tổ chức ranking hiện có.
(Bảng video-level max-window cho thấy giá trị thật của caption: R@10 69%, R@25 85%, R@100 96%
— rất tốt làm **candidate generator** cho bước sau.)

## Phụ lục — bảng đầy đủ

**Cả 40 case, oracle pool:**

| Biến thể | MRR | R@1 | R@5 | R@10 | R@25 | R@50 | hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | .232 | 18% | 25% | 32% | 55% | 75% | 40/40 |
| `sentence` | .245 | 10% | **45%** | **58%** | **72%** | **92%** | 40/40 |
| `chunk128` | .192 | 8% | 28% | 32% | 58% | 85% | 40/40 |
| `chunk256` | .236 | 18% | 28% | 35% | 55% | 80% | 40/40 |
| `core` | **.262** | **18%** | 32% | 42% | 70% | **92%** | 40/40 |

**Cả 40 case, realistic pool:**

| Biến thể | MRR | R@1 | R@5 | R@10 | R@25 | R@50 | hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | .224 | **18%** | 25% | 30% | 48% | 70% | 33/40 |
| `sentence` | .178 | 8% | 28% | **40%** | **55%** | **78%** | 33/40 |
| `chunk128` | .181 | 8% | 28% | 30% | 45% | 75% | 33/40 |
| `chunk256` | .208 | 15% | 25% | 32% | 50% | 72% | 33/40 |
| `core` | .216 | 15% | 25% | 30% | 52% | 75% | 33/40 |

**Combo/score-fusion (30 case tệ, realistic pool):**

| Phương pháp | MRR | R@1 | R@5 | R@10 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_max` | .031 | 0% | 0% | 7% | 30% | 60% |
| `core_max` | **.098** | **7%** | **7%** | **10%** | **37%** | **67%** |
| `max_full_core` | .094 | 7% | 7% | 10% | 27% | 67% |
| `full+core_rrf` | .035 | 0% | 3% | 7% | 23% | 63% |
| `full+sentence_rrf` | .039 | 0% | 3% | 7% | 27% | 63% |
| `zsum_full+C1.0` | .049 | 0% | 7% | 7% | 27% | 63% |

**Combo/score-fusion (10 case control, realistic pool):**

| Phương pháp | MRR | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: | ---: |
| `full_max` | **.803** | **70%** | **100%** | **100%** |
| `zsum_full+S0.5` | .715 | 60% | 90% | 90% |
| `zsum_full+C0.5` | .687 | 50% | 100% | 100% |
| `full+core_rrf` | .679 | 50% | 90% | 100% |
| `core_max` | .570 | 40% | 80% | 90% |

## Kết luận & hướng đi tiếp theo

**Nhận xét tổng**: caption + Qwen3-4B bị chặn trần bởi *ambiguity văn bản giữa các video*
(TF-IDF ≈ dense; video top-1 chỉ 38%; video-first không giúp). Density là nút thắt thật nhưng
chỉ ở tầng recall sâu, không phải top-1. Vì vậy caption hợp vai **kênh candidate/verify bù trừ**,
không phải kênh xếp hạng chính.

Thứ tự ưu tiên nên thử tiếp (ROI giảm dần):

1. **Đa kênh ở tầng candidate + verifier**: union PE ∪ caption đã phủ 136/137; thêm OCR (chữ
   trên màn hình) và ASR/transcript rồi để agent/Gemini verify top-N. Đây là hướng tăng recall
   *hệ thống* thực tế nhất; caption đóng góp đúng nhóm case text-heavy mà PE/ASR bỏ sót.
2. **Core/sentence caption toàn corpus bằng Qwen3-4B (Kaggle) + MaxSim làm kênh candidate**:
   chỉ đáng nếu mục tiêu là R@100 tiến tới 100% (probe: trong candidate set, R@25 40→60–63%,
   R@50 67→90%; full-corpus còn giải quyết 7/30 case target không có trong pool). Giữ full
   caption cho precision, chunk chỉ mở rộng candidate + verify. Chi phí ~4–6h Kaggle,
   ~$20 DeepSeek cho ~2.39M câu.
3. **Query-side (paraphrase mean, VI gốc, holistic)** gộp vào bước 2: rẻ, tiền lệ +2.4 case R@1
   trên PE, nhưng một mình không phá trần.
4. **Không nên**: chunk token cơ học, TF-IDF chunk, video centroid, video-first caption-only.

## Artifacts

- Script: `scripts/benchmarks/probe_caption_chunking.py` (+`--self-test`),
  `scripts/benchmarks/generate_core_captions.py` (DeepSeek, resumable).
- Dữ liệu: `artifacts/caption_vs_pe_core/probe/{pool,lexical_probe,dense_probe,core_captions_probe,combo_probe,probe_summary}.json`
  (gitignored; vector cache `vectors_*.npy`).
- Chi phí probe: ~1h (lexical 15', DeepSeek 7', dense 0.6B GPU ~20').
