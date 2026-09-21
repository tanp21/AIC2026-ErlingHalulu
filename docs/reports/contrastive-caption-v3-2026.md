# Contrastive Caption v3 2026 — Xoá mạnh chi tiết trùng, giữ tối đa chi tiết riêng

## Câu hỏi

Tiếp nối session trước: caption CapRL thua PE-Core vì caption của **các video khác nhau gần như
đồng nghĩa** (video top-1 chỉ 38% so với 55.5%). Session trước đã thử viết lại caption cho khác
biệt và thấy có tác dụng. Session này hỏi tiếp: nếu **xoá mạnh** nội dung trùng với các shot/video
khác nhưng **giữ lại tối đa chi tiết riêng**, thì recall tới đâu — và cái đang thực sự tạo ra cải
thiện là gì?

Caption gốc **không bị sửa**: chỉ đọc. Mọi artifact mới đều ghi ra file riêng có run-stamp.
Đã ghi và kiểm tra sha256 của `data/2026/captions/shot_captions.parquet`
(`ddaf6a51d9e1ecb1…`, `v3_originals_hash.json`).

## Phát hiện 0 — v2 của session trước bị lỗi, chưa từng chạy được

`scripts/benchmarks/contrastive_rewrite_v2.py` (viết 07:28 hôm nay, không có artifact nào) có lỗi
hợp đồng dữ liệu: `read_records()` trả `{id: {tag: text}}` nhưng mọi caller dùng như string.

```
surviving(pass1[id])      -> TypeError: expected string or bytes-like object, got 'dict'
rewrite_metrics(..., id)  -> AttributeError: 'dict' object has no attribute 'lower'
```

`--stage all` chết ở repair; `--stage merge` sẽ ghi dict vào trường `rewritten`. Đây là lý do
không có artifact `contrastive_v2_*` nào — đã kiểm chứng, không phải suy đoán.

## Phát hiện 1 — Đổi từ ngữ (chiến lược của v2) không có tác dụng gì

v2 *đổi từ* các cụm dùng chung nhưng giữ nguyên nghĩa ("on the left" → "along the left-hand side of
the frame"). Mô phỏng đúng chiến lược đó ở tầng caption: paraphrase các cụm dùng chung, thay ngược
vào caption, rồi đo.

| | near-dup >0.85 | mean cosine | BAD oracle MRR | CTRL MRR |
|---|---:|---:|---:|---:|
| original | 0.71% | 0.4076 | 0.0414 | 0.8033 |
| paraphrase-shared | **0.70%** | 0.4093 | **0.0393** | 0.7933 |

Cosine tầng cụm từ: mean **0.9014**, 39.6% cặp >0.95 — paraphrase embed gần như y hệt.

**Kết luận:** đổi từ ngữ **không làm giảm độ trùng lặp** (0.71% → 0.70%) và không cải thiện
retrieval. Thước đo `shared_remaining` của v2 là **lexical**, nên nó sẽ báo thành công trong khi
embedding không nhúc nhích — chiến lược của v2 tự "lách" được thước đo của chính nó. Đây là cơ sở
để chuyển sang **xoá** thay vì **đổi từ**.

*(Hạn chế: mới paraphrase 48 cụm hay gặp nhất = 17% số lượt xuất hiện, nhưng đó là nhóm nặng ký
nhất.)*

## Phát hiện 2 — Sửa baseline: near-dup gốc ở 4B là 1.176%, không phải 0.71%

`score_4b_stage` không hề tính `similarity["original"]` (vòng lặp chỉ chạy qua `texts`, không có key
`original`), nên báo cáo 4B **không có dòng gốc**. Con số 0.71% trong báo cáo trước là của **0.6B**
— so lệch encoder. Đo lại trực tiếp từ cache vector 4B trên đúng 5.567.891 cặp cross-video:

| biến thể | 4B mean | 4B >0.85 |
|---|---:|---:|
| **original (đo lại, chưa từng có trong artifact)** | 0.5062 | **1.176%** |
| contrastive (clean60) | 0.5081 | 0.06% |
| contrastive_long (clean150) | 0.5364 | 0.25% |
| contrastive_diff | 0.5293 | 1.17% |
| contrastive_rewrite_local | 0.5058 | 1.00% |
| contrastive_rule | 0.5332 | 1.16% |

**Hệ quả:** so với baseline đúng, `rule` (1.16%) và `diff` (1.17%) **đi ngang**, không tệ hơn; bản
v1 rewrite (1.00%) nhích tốt hơn. Cách đọc đúng là mức giảm gần-trùng bám theo **độ ngắn** của
caption, không phải theo "chất lượng xoá trùng".

## Phát hiện 3 — Độ dài là đòn bẩy chính, không phải contrastive

Các nhánh cắt ngắn tất định, **không LLM**, trên đúng pool đó (0.6B):

| biến thể | từ | BAD MRR | BAD@1 | BAD@50 | CTRL MRR | CTRL@1 |
|---|---:|---:|---:|---:|---:|---:|
| original | 628 | .041 | 0% | 66.7% | **.803** | **70%** |
| stripped *(DeepSeek, không cắt ngắn)* | 553 | .041 | 0% | 73.3% | .742 | 60% |
| prefix_300w | 300 | .088 | 3.3% | 80.0% | .560 | 40% |
| prefix_150w | 150 | .083 | 0% | 76.7% | .542 | 50% |
| lead_3s *(cắt theo câu)* | 94 | .105 | 3.3% | 80.0% | .448 | 30% |
| prefix_60w | 60 | .100 | 3.3% | 76.7% | .380 | 30% |
| **first_sentence** *(không LLM)* | **31** | **.130** | 3.3% | 83.3% | .436 | 30% |
| contrastive_long *(DeepSeek, giữ chi tiết)* | 119 | .074 | 3.3% | 83.3% | .556 | 40% |
| distinctive *(DeepSeek, 60 từ, no-neighbor)* | 55 | .138 | 6.7% | 86.7% | .469 | 30% |
| **contrastive** *(DeepSeek)* | 62 | **.163** | **10%** | **90.0%** | .443 | 30% |

Ba kết luận:

1. **`stripped` giữ nguyên độ dài thì không thay đổi gì** (.041 → .041). Xoá logo/timecode vô ích.
2. **Cắt ngắn miễn phí được phần lớn hiệu ứng**: `first_sentence` (1 câu, 31 từ, không GPU) đạt
   **80%** MRR của contrastive (.130 vs .163).
3. **Giữ nhiều chi tiết thì THUA**, và tăng đơn điệu khi ngắn đi:
   `contrastive_long` (119 từ) .074 < `lead_3s` (94 từ) .105 < `first_sentence` (31 từ) .130.

Phân rã phần đóng góp (cùng họ prompt DeepSeek, cùng ~60 từ):

| bước | BAD MRR | phần tăng |
|---|---:|---:|
| cắt thô 60 từ, không LLM | .100 | — |
| + LLM tóm tắt giữ chi tiết riêng (không có neighbor) — `distinctive` | .138 | +.038 |
| + tín hiệu neighbor/contrastive — `contrastive` | .163 | +.025 |

→ Ở **cùng độ dài**, LLM thắng cắt thô rõ (.163 vs .100), và tín hiệu contrastive đóng góp riêng
khoảng **+.025 MRR (~20% tổng mức tăng)**. Vậy contrastive có giá trị thật, nhưng độ dài mới là
yếu tố lớn nhất.

## Phát hiện 4 — Lỗi matcher: 12.7% cụm từ không khớp lại được caption nguồn

`phrase_pattern` (`clean_captions_local.py:76`) dùng lớp phân cách `[\s,;:/\-–—]+`, **thiếu dấu
nháy và dấu câu cuối câu**. Hệ quả trên 9.929 cụm: chỉ **88.0%** khớp lại được chính caption sinh
ra nó; "chef's" bị tokenize thành `chef s` nên cụm `wearing a white chef s` không bao giờ khớp.
Thêm `'’\.!\?` vào lớp → **94.2%**, không over-match. Điều này âm thầm làm các biến thể `rule`/`diff`
của session trước xoá hụt. v3 dùng matcher đã sửa.

## Phát hiện 5 — Trần mining 25 cụm đang chặn tín hiệu

`shared_phrases_probe.json`: **3.202/3.345 window (95.7%) nằm đúng trần 25 cụm** (median = max = 25),
phủ ~16% độ dài caption. Mining lại **không trần** (cùng ngưỡng ≥2 neighbor):

| | cụm/window | xoá được | độ dài còn lại |
|---|---:|---:|---:|
| rule cũ (trần 25) | 25 | 25 | ~545 từ (87%) |
| prune ≥2 | 57.7 | 51.7 | 444 từ (71%) |
| prune ≥1 | 82.6 | 72.7 | 354 từ (56%) |

## Artifacts

- Script: `scripts/benchmarks/contrastive_rewrite_v3.py` (+`--self-test`).
  Stages: `paraphrase-probe`, `controls`, `prune`, `mine-axes`, `rewrite`, `merge`, `audit`,
  `table`, `verify-originals`.
- Hook chấm điểm: `clean_captions_probe.py` thêm `--variants-json`, `--score-out`,
  `--vector-prefix`, `--cache-suffix`, `--min-coverage` (chặn nếu phủ <99%, tránh âm thầm rơi về
  caption gốc), và **tính `similarity["original"]` thật ở 4B**.
- Dữ liệu: `artifacts/caption_vs_pe_core/probe/{paraphrase_probe_smoke,v3_controls_0b_ctrl0b,
  v3_prune_0b_prune0b,v3_originals_hash}.json` + cache vector `v3_vectors_0b_*.npy`.

## Còn lại

Sinh caption v3 bằng LLM local (3 budget 120/200/300 từ, cả hai trục) → audit → chấm điểm 4B trên
pool → bảng so sánh đầy đủ. Kết quả bổ sung vào báo cáo này.
