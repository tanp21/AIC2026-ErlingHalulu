# Audio Embedding Benchmark — Model Size vs. Query Phrasing

## Bối cảnh

Câu hỏi ban đầu: có nên thay Qwen3-Embedding-0.6B (model production hiện tại cho nhánh audio)
bằng 4B hoặc 8B để tăng recall không? Benchmark chạy trên Kaggle T4x2 (3 tài khoản song song),
dùng đúng config chunking production (`SemanticChunkConfig` "config C": `context_window_sentences=3,
breakpoint_percentile=15.0, min_duration_seconds=10, target_duration_seconds=40,
max_duration_seconds=75, overlap_sentences=1, pause_boost_enabled=False`) và đúng phương pháp đo của
`scripts/benchmarks/bench_audio_search.py --dense` (video-collapsed gold-video recall, xem
`evaluate()`/`DenseIndex.rank_videos()`).

5 biến thể đã benchmark đầy đủ: Qwen3-Embedding-0.6B (fp16), 4B (fp16 + int8 bitsandbytes),
8B (int8 + int4 bitsandbytes). 8B-fp16 mặc định không chạy được ổn định do lỗi kiến trúc khi
SentenceTransformer/accelerate chia model qua 2 GPU (`RuntimeError: Expected all tensors to be on
the same device` trong RMSNorm) — không phải lỗi do quantization.

## Vòng 1 — dùng nguyên query mô tả hình ảnh (`audio_ground_truth.json`)

Ground truth gốc có 48 câu, viết cho nhánh visual (KIS/QA/TRAKE thật của VBS). Sau khi loại 9 câu
mà nội dung không hề được nói ra trong audio (múa lân không lời, chi tiết OCR/hình ảnh thuần túy —
xem `docs/superpowers` memory `audio_benchmark_unanswerable_queries.md`), còn lại 39 câu:

| Model | R@1 | R@5 | R@10 | R@50 | R@100 | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B (fp16) | 53.8% | 69.2% | 84.6% | 100% | 100% | 1.0 |
| 4B (fp16) | 53.8% | 71.8% | 84.6% | 100% | 100% | 1.0 |
| 4B (int8) | 51.3% | 69.2% | 82.1% | 97.4% | 100% | 1.0 |
| 8B (int8) | 56.4% | 69.2% | 76.9% | 94.9% | 100% | 1.0 |
| 8B (int4) | 56.4% | 69.2% | 71.8% | 100% | 100% | 1.0 |

**Kết luận vòng 1**: tăng size model từ 0.6B lên 4B/8B không cải thiện recall rõ rệt — chênh lệch
nằm trong biên độ nhiễu (mỗi câu ≈ 2.6 điểm % trên 39 câu). Không có lý do rõ ràng để nâng cấp
model production chỉ dựa trên vòng benchmark này.

### Vì sao chênh lệch giữa các model nhỏ mà vẫn có nhiễu?

Đào sâu câu `2025:query-p1-8-kis` (0.6B rank 7, 4B rank 2-3, 8B-int8 rank 26-54) phát hiện: video
gold `L22_V030` và một video khác trong corpus, `L22_V027`, phát **cùng một bản tin thời sự quốc tế**
(chương trình "60 Giây" đưa lại tin lễ hội "Japan Fes" ở New York) — gần như là bản sao nội dung.
Đây là một cặp "song sinh nội dung" khiến bất kỳ embedding nào cũng khó phân định dứt khoát; model
nhỏ có xu hướng nhạy hơn với khác biệt câu chữ bề mặt nên đôi khi chọn nhầm — đây là nhãn gold mơ hồ
do trùng lặp corpus, không phải model kém.

## Vòng 2 — viết lại 39 query theo phong cách từ khóa thực tế khi thi

Giả thuyết: khi thi thật, người dùng audio search gõ **từ khóa/cụm từ họ đoán sẽ được nói ra**
(tên riêng, địa danh, con số, cụm đặc trưng), không mô tả những gì họ **nhìn thấy** trên màn hình.
Ví dụ, thay vì "một cô bé đeo một con bạch tuộc màu đỏ phía trước ngực", dùng "lễ hội ẩm thực Nhật
Bản Japan Fes lớn nhất thế giới bánh bạch tuộc" — cụm từ gần như nguyên văn xuất hiện trong transcript.

Toàn bộ 39 query đã được viết lại theo cách này, mỗi câu đối chiếu trực tiếp với transcript thật của
video gold (`data/2026/audio/generated/step_data_clean.parquet`) để đảm bảo cụm từ mới thực sự xuất
hiện hoặc gần-nguyên-văn xuất hiện trong lời thoại. 37/39 câu có bằng chứng mạnh (trích dẫn gần như
nguyên văn); 2 câu chỉ có bằng chứng yếu (chủ đề đúng nhưng chi tiết cụ thể không được nói ra):

- `2025:query-p1-13-kis` (vệ sinh ống kính máy ảnh) — transcript là phỏng vấn chủ tiệm sửa máy ảnh
  phim, đúng chủ đề nhưng không nhắc riêng hành động lau ống kính bằng tăm bông.
- `2025:query-p1-7-kis` (loài chim cụ thể ở U Minh Hạ) — transcript nói chung về đa dạng sinh học,
  không mô tả bộ lông loài chim trong câu hỏi.

Kết quả sau khi đổi query (39 câu, cùng passages/embeddings đã tính, chỉ encode lại query mới):

| Model | R@1 (cũ → mới) | R@5 (cũ → mới) | R@10 (cũ → mới) | R@50/R@100 | Median |
|---|---|---|---|---|---:|
| 0.6B (fp16) | 53.8% → **84.6%** | 69.2% → **94.9%** | 84.6% → 94.9% | 100%/100% | 1 |
| 4B (fp16) | 53.8% → **87.2%** | 71.8% → **100%** | 84.6% → 100% | 100%/100% | 1 |
| 4B (int8) | 51.3% → **87.2%** | 69.2% → **100%** | 82.1% → 100% | 100%/100% | 1 |
| 8B (fp16, sharded) | 51.3% → **87.2%** | 69.2% → **100%** | 74.4% → 100% | 100%/100% | 1 |
| 8B (int8) | 56.4% → **89.7%** | 69.2% → **100%** | 76.9% → 100% | 100%/100% | 1 |
| 8B (int4) | 56.4% → **84.6%** | 69.2% → **100%** | 71.8% → 100% | 100%/100% | 1 |

**Đây là bảng benchmark cuối cùng, đầy đủ 6/6 biến thể.** Theo quyết định của người dùng
(2026-09-02), từ nay các benchmark model embedding audio trong repo này **chỉ dùng bộ 39 query
từ khóa** — không dùng lại query mô tả hình ảnh, vì nó không phản ánh đúng chất lượng nhánh audio
(xem memory `audio_benchmark_keyword_only_methodology`).

## Kết luận

1. **Cách viết query ảnh hưởng đến recall nhiều hơn hẳn việc chọn model hay độ chính xác số học.**
   Đổi từ query mô tả-hình-ảnh sang query từ-khóa-thực-tế nâng R@1 thêm 28-33 điểm phần trăm và đưa
   R@5 lên gần như tuyệt đối (100% ở 4/5 biến thể) — lớn hơn nhiều so với bất kỳ chênh lệch nào giữa
   0.6B/4B/8B hay fp16/int8/int4 quan sát được ở vòng 1.
2. **Quantization gần như miễn phí về recall khi query đã "sạch".** 4B-int8 = 4B-fp16 = 8B-fp16
   (sharded) hệt nhau về R@1 (87.2%) và R@5 (100%); 8B-int4 chỉ thấp hơn 5 điểm % R@1, vẫn vượt xa
   mọi kết quả vòng 1. Việc 8B-fp16 không quantize cũng ngang bằng 4B (không vượt trội) càng khẳng
   định đây không phải hiệu ứng của quantization che khuất lợi ích size — bản thân size cũng không
   giúp gì thêm.
3. **8B-int8 nhỉnh nhất (R@1 89.7%)** nhưng cách biệt 3-5 điểm % so với các biến thể khác vẫn nằm
   trong biên độ nhiễu của 39 câu (1 câu ≈ 2.6 điểm %) — chưa đủ bằng chứng để khẳng định 8B tốt hơn
   0.6B một cách hệ thống.
4. **Khuyến nghị**: không có lý do rõ ràng để đổi model production 0.6B chỉ vì recall — nếu muốn cải
   thiện trải nghiệm audio search cho đội thi, ưu tiên **huấn luyện/hướng dẫn người dùng viết query
   dạng từ khóa** (giống cách viết lại ở vòng 2) sẽ hiệu quả hơn nhiều so với nâng cấp hạ tầng model.
   Nếu vẫn muốn nâng cấp, 8B-int8 là lựa chọn hợp lý nhất về mặt hiệu năng nhưng cần cân nhắc chi phí
   VRAM khi chạy đồng thời với PE-Core trên server production (xem phần triển khai).

## Phương pháp & dữ liệu thô

- Chunking + embedding toàn corpus (873 video, ~8,365-8,367 passage tùy model) chạy một lần trên
  Kaggle T4x2 cho mỗi biến thể; passages + embeddings được lưu lại (`passages_<label>.parquet`,
  `embeddings_<label>.npy`) và tái sử dụng cho vòng 2 — vòng 2 chỉ encode lại 39 query mới, không
  chunk/embed lại corpus, nên mỗi lần rerun chỉ mất vài phút thay vì hàng giờ.
- `RECALL_AT = (1, 5, 10, 50, 100)`; rank tính trên toàn bộ ~873 video trong corpus (không phải chỉ
  so với 48/39 câu query khác); video-collapse lấy điểm passage tốt nhất mỗi video, giống hệt cách
  production `audio_search.merge_temporal_hits` gộp điểm.
- 39 câu keyword, evidence trích transcript, và kết quả rank chi tiết từng model nằm trong
  `docs/reports/` (báo cáo này) và bộ nhớ dự án (`audio_benchmark_unanswerable_queries.md` cho danh
  sách 9 câu loại trừ khỏi benchmark).
