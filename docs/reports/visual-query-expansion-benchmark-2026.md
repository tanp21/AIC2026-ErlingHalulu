# Query Expansion Benchmark 2026 — Holistic Caption + Aspect Queries → Shot Fusion

## Bối cảnh

Ý tưởng: chỉ dùng PE-Core, nhưng thay vì 1 caption ngắn/case, Gemini tách prompt tiếng Việt gốc
thành 1 **holistic caption** + 2-5 **aspect query** (mỗi aspect mô tả 1 khía cạnh thị giác), mỗi
query được retrieve riêng, kết quả gộp ở mức **shot** theo công thức:

```
final(shot) = λ · holistic_RRF(shot) + (1-λ) · support(shot) · aspect_fusion(shot)
```

với `aspect_fusion` bị chặn trần bởi 1 query hoàn hảo (không cho phép coverage lấn át holistic),
và một cơ chế "neighbor evidence" (γ) + "soft support penalty" (β) để cứu các case mà holistic bị
miss. Toàn bộ định nghĩa scoring, 10 invariant (I1-I10), và protocol chống overfit (dev/holdout
tách biệt, tuning theo giai đoạn) đã được duyệt trước khi code — xem plan đã lưu, và
`backend/subquery_fusion.py` + `tests/visual/test_subquery_fusion.py` (19 test, tất cả pass).

**Kết luận ngắn gọn: phương pháp này KHÔNG vượt được baseline đơn giản (1 caption + shot-dedup)
trên dev set, kể cả sau khi tune tham số và thử 2 phiên bản prompt khác nhau. Khuyến nghị KHÔNG
triển khai Phase B (tính năng production) dựa trên bằng chứng hiện tại.** Chi tiết bên dưới.

## Dữ liệu & phương pháp

- **Dev** = Test/2025 (32) + Round 1-2026 (27) + Round 2-2026 (36) = 95 case. **Holdout** = Round
  3-2026 (42 case) — **chưa từng được đọc trong toàn bộ quá trình tune** (chỉ dev). Vì kết quả
  dev đã âm tính rõ ràng, holdout **không được dùng** — tránh lãng phí lần đánh giá "chỉ chạy một
  lần" cho một cấu hình mà bằng chứng dev đã cho thấy khó vượt qua bar chấp nhận.
- **Ground truth/metric**: giống hệt benchmark 3-model trước — cùng frame nếu cùng video và lệch
  ≤10s so với bất kỳ target nào (kể cả frame auto-filled cùng shot).
- **Harness**: `scripts/benchmarks/bench_query_expansion.py`. **Stage 0 sanity**: baseline
  `caption_approved` (1 caption đã duyệt, không fusion) tái tạo **chính xác 95/95** rank đã lưu
  trong `visual_embedding_benchmark_2026_en.json` — xác nhận harness đúng 100% trước khi tin bất
  kỳ số nào ở các giai đoạn sau.
- **Shot fusion**: dùng shot boundaries có sẵn cho toàn bộ L21-L30
  (`data/2026/shots/`), map mỗi frame hit về `(video, shot_idx)` qua `ShotIndex.shot_at()`, fallback
  time-bucket 5s cho video không có shot.
- **Prompt V1**: Gemini sinh 2-5 aspect/case theo facet cố định (people/clothing/object/action/
  setting/text/other), không giới hạn "phải hiếm". 137/137 case dịch thành công, 0 bị cắt/drop,
  max 27 token (dưới target 50 rất nhiều) — **đã được bạn duyệt** trước khi tune.

## Giai đoạn 0-1: harness sanity + grid tham số (dev, prompt V1)

Baseline tham chiếu (dev, 95 case):

| Method | MRR | R@1 | R@5 | R@10 | R@100 | hits |
|---|---:|---:|---:|---:|---:|---:|
| `caption_approved` (1 caption, không dedup) | .432 | .326 | .526 | .579 | .800 | 88/95 |
| **`caption_approved_shotdedup`** (1 caption, dedup theo shot) | **.442** | **.326** | .558 | .653 | .821 | 87/95 |
| `holistic_only` (λ=1, caption V1 mới) | .426 | .284 | .600 | .642 | .832 | 85/95 |

**Chỉ riêng việc dedup theo shot (không cần expansion) đã nâng R@10 từ .579 → .653** — đây là cải
tiến thật, độc lập với ý tưởng aspect. `holistic_only` (caption mới do Gemini viết cho mục đích
expansion) lại **kém hơn** caption đã duyệt trên R@1 (.284 vs .326) dù khá hơn ở R@5/R@10.

Grid 24 cấu hình (k∈{10,60} × K∈{500,2000} × agg∈{mean,top_m_mean} × λ∈{0.25,0.5,0.75}), γ=0,
β=1 (không ablation): **cấu hình tốt nhất (S1\*: k=10, K=2000, agg=mean, λ=0.75) đạt MRR=.416,
R@1=.274, R@10=.653 — thấp hơn `caption_approved_shotdedup` trên cả MRR và R@1**, và không vượt
`holistic_only` trên MRR. λ=0.25 (nghiêng aspect) luôn tệ nhất trong mọi tổ hợp (MRR .32-.35).

## Giai đoạn 2: ablation (neighbor γ, soft β, dedup)

Tất cả trên nền S1\*. Không ablation nào đạt ngưỡng chấp nhận (ΔMRR≥0.02 **và** ≥2 case cải
thiện R@1, không tệ hơn ở >1/3 vòng dev):

| Ablation | MRR | R@1 | R@10 |
|---|---:|---:|---:|
| S1\* (nền) | .416 | .274 | .653 |
| γ=0.5 (neighbor, scope=aspects) | .415 | .274 | .653 |
| γ=1.0 (neighbor, scope=aspects) | .420 | .284 | .653 |
| β=0.5 soft (T_h=200) | .416 | .274 | .653 |
| β=0.25 soft (T_h=200) | .416 | .274 | .653 |
| β=0 hard gate (T_h=200) | .416 | .274 | .653 |
| dedup_seconds=2 | .418 | .274 | .642 |

γ=1.0 cho cải thiện nhỏ nhất (MRR +.004, R@1 +1 case) — **dưới ngưỡng chấp nhận cả về MRR lẫn số
case**. β (soft/hard gate) hoàn toàn không đổi kết quả — với K=2000 và λ=0.75, group không có
holistic trực tiếp gần như không bao giờ đủ điểm để lọt top kết quả, nên penalty/gate không có
tác dụng thực tế ở cấu hình này. **Không ablation nào được chấp nhận → S2\* = S1\*.**

## Phân tích thất bại (dev, S1\* vs `caption_approved_shotdedup`)

So từng case: **30 thắng / 33 thua / 32 hòa** (n=95) — gần như 50/50, nghiêng nhẹ về phía thua.

Mẫu hình rõ rệt trong các case thua: khi holistic caption **đã đúng và tự tin** (baseline rank
1-19), thêm aspect thường **kéo rank xuống**, đôi khi rất mạnh (rank 14→155, rank 130→291,
rank 54→170). Ví dụ điển hình:

> `Round 1 - 2026/query-p1-11-kis:1` — holistic "Three cyclists cross the finish line in a race"
> đã đúng ở rank 14, nhưng 2 aspect "cyclist in a yellow jersey and black shorts" / "cyclist in a
> blue jersey and red shorts" lại khớp mạnh với các đoạn đua xe đạp KHÁC trong corpus (rất nhiều
> video có người mặc áo đua nhiều màu), kéo rank xuống 155.

Ngược lại, các case thắng thường là khi holistic **mơ hồ/yếu** (rank 50-170 hoặc miss hẳn) và ít
nhất 1 aspect nêu được chi tiết hiếm (VD "FANA club banner" — tên riêng, hoặc "dragon costume
moving its head" — vật thể hiếm) giúp cứu case.

**Nguyên nhân gốc rễ có khả năng cao nhất**: công thức fusion là **trung bình có trọng số**
(convex combination), không phải cơ chế "chỉ thưởng khi đồng thuận". Khi 1 shot sai nhưng tình cờ
khớp tốt với các aspect *chung chung* (VD "outdoor setting", "cyclist jersey" — khớp hàng chục
video khác), điểm trung bình của nó vẫn tăng đủ để vượt qua shot đúng có holistic mạnh nhưng
aspect yếu. Cơ chế trần `1/(k+1)` (invariant I1) ngăn aspect **áp đảo hoàn toàn**, nhưng không
ngăn được việc **làm nhiễu thứ hạng** khi λ không đủ lớn.

## Giai đoạn 3: 1 lần lặp prompt (V2 — bắt buộc aspect phải "hiếm")

Prompt V2 yêu cầu rõ: chỉ viết aspect nêu chi tiết **hiếm/đặc trưng** (màu/hoa văn bất thường, số
đếm cụ thể, chữ trên màn hình, tên riêng), cho phép 0-1 aspect nếu prompt không có gì đặc biệt
thêm, cấm lặp lại ngữ cảnh chung đã có trong holistic. Kết quả: 23/137 case giờ có 0 aspect,
77 có 1, chỉ 3 có 3+ (so với V1: toàn bộ ≥2).

| Method (dev, V2) | MRR | R@1 | R@10 |
|---|---:|---:|---:|
| `holistic_only` V2 | .415 | .284 | **.695** |
| S2\* (γ=1.0) λ=0.75, V2 | .400 | .263 | .684 |
| S2\* (γ=1.0) λ=0.5, V2 | .365 | .253 | .674 |
| `caption_approved_shotdedup` (tham chiếu) | .442 | .326 | .653 |

V2 **không cải thiện** so với V1 — thậm chí λ=0.75/γ=1.0 ở V2 (MRR .400) còn thấp hơn cấu hình
tương ứng ở V1 (MRR .420). Điểm sáng duy nhất: `holistic_only` V2 đạt R@10=.695, cao nhất trong
toàn bộ thí nghiệm (kể cả vượt baseline .653) — cho thấy bản thân **caption holistic mới** (không
cần aspect) có tiềm năng, nhưng R@1 (.284) vẫn thấp hơn caption đã duyệt (.326). Đây là 1 lần lặp
prompt bị từ chối theo acceptance rule; dừng ở đây thay vì tiếp tục lặp đến V5, vì bằng chứng đủ
nhất quán qua 2 prompt khác nhau để kết luận vấn đề nằm ở **cơ chế fusion**, không phải cách viết
prompt.

## Kết luận

1. **Không triển khai Phase B.** Trên dev set (95 case), không cấu hình fusion nào (37 cấu hình
   đã thử: 24 grid + 7 ablation + 6 so sánh V2) vượt được baseline đơn giản
   `caption_approved_shotdedup` (MRR .442, R@1 .326) trên MRR hay R@1 — chỉ số mà VBS thực sự
   chấm điểm. Một vài cấu hình có R@10/R@100 nhỉnh hơn, nhưng đây không phải điều đội thi ưu
   tiên.
2. **Phát hiện phụ có giá trị**: shot-level dedup của MỘT caption đơn (không cần expansion) đã
   nâng R@10 đáng kể (.579→.653). Đây là một cải tiến độc lập, rẻ, đáng cân nhắc áp dụng ngay cho
   `visual_search` hiện tại — dùng `backend/subquery_fusion.py`'s `shot_ranks()` + dedup để lọc
   trùng shot trong kết quả top-K, không cần Gemini/expansion.
3. **Nguyên nhân gốc rễ**: cơ chế trung bình-có-trọng-số giữa holistic và aspect vốn dễ bị nhiễu
   bởi các aspect "chung chung" khớp nhiều video khác — ngay cả sau khi ép prompt phải nêu chi
   tiết hiếm (V2), vấn đề không biến mất. Muốn thử tiếp, hướng khả thi hơn là cơ chế **lọc cứng**
   (aspect chỉ dùng để re-rank trong tập ứng viên holistic đã chọn sẵn, không bao giờ tự nó đưa
   một shot mới vào) thay vì trộn điểm tuyến tính — nhưng đây là một thiết kế khác, chưa benchmark.
4. Holdout (Round 3) **chưa từng được dùng** — giữ nguyên cho lần benchmark tiếp theo nếu có
   hướng tiếp cận mới đáng thử.

**Dừng lại và chờ quyết định của bạn**, theo đúng cổng quyết định trong plan.

## Phương pháp & dữ liệu thô

- `backend/subquery_fusion.py` (fusion thuần, không I/O) + `tests/visual/test_subquery_fusion.py`
  (19 test, invariant I1-I10 + property-based, tất cả pass).
- `backend/query_expansion.py`: logic gọi Gemini + token policy (target 50, limit 72 token
  PE-Core, chỉ drop khi vượt limit).
- `scripts/benchmarks/expand_visual_queries.py`: sinh expansion (V1, V2).
- `scripts/benchmarks/bench_query_expansion.py`: harness tune theo giai đoạn (`--stage
  sanity|grid|ablation|holdout`).
- Expansion đã dịch: `artifacts/query_expansion/v1/` và `v2/` (mỗi bộ có `expansions.json` +
  `review.txt`).
- Log toàn bộ cấu hình đã thử: `artifacts/query_expansion/tuning_log.jsonl` (31 dòng từ
  stage 1-2; 6 cấu hình so sánh V2 chạy qua script riêng, không log — liệt kê thủ công ở trên).
- Không có file `query_expansion_config.json` được freeze — không có cấu hình nào đủ tốt để
  đóng băng.
