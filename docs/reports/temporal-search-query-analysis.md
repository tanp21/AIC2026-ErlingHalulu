# Temporal Search — Per-query Benchmark Analysis

## Thiết lập

- Ground truth: 7 query TRAKE.
- Cấu hình production: `candidate_k=2000`, `scoring_method=raw_mean`.
- Khoảng cách tối đa giữa hai event liền kề: 120 giây.
- Một sequence được xem là đúng khi cùng video và mọi event nằm trong ±10 giây so với ground truth.
- Audio service không hoạt động trong lúc benchmark, vì vậy kết quả bên dưới sử dụng visual fallback.

## Tổng quan

| Query | Rank sequence đúng | Kết quả |
|---|---:|---|
| `query-p1-4` | Không tìm thấy | Fail |
| `query-p1-16` | Không tìm thấy | Fail |
| `query-p1-18` | 3 | Pass |
| `query-p2-10` | 1 | Pass |
| `query-p2-22` | Không tìm thấy | Fail |
| `query-p3-1` | 3 | Pass |
| `query-p3-24` | 1 | Pass |

Kết quả tổng hợp: 4/7 query có đáp án trong top 10, tương ứng Recall@10 `0.5714` và MRR `0.3810`.

## Phân tích từng query

### `query-p1-4` — không tìm thấy

| Event | Candidate rank tốt nhất | Sai lệch |
|---|---:|---:|
| E1 | 1 | 0.36 giây |
| E2 | 50 | 8.32 giây |
| E3 | 33 | 4.20 giây |
| E4 | 13 | 1.44 giây |

Sequence cùng video gần ground truth nhất đứng rank 1:

```text
Dự đoán: [189.32, 232.28, 233.40, 276.52]
GT:      [188.96, 204.88, 217.40, 234.84]
Sai số:  [0.36,   27.40,  16.00,  41.68]
```

Model tìm được candidate đúng cho từng event riêng lẻ, nhưng DP hiện chỉ giữ một prefix tốt nhất tại mỗi endpoint. Candidate đúng của E2/E3 bị các cảnh nấu tương tự có visual score cao hơn thay thế. `raw_mean` cũng chưa thưởng cho một chuỗi thao tác tập trung trong cùng đoạn. Đây chủ yếu là lỗi sequence assembly/ranking, không phải candidate recall.

### `query-p1-16` — không tìm thấy

| Event | Candidate rank tốt nhất | Sai lệch |
|---|---:|---:|
| E1 | 558 | 8.37 giây |
| E2 | 1517 | 8.70 giây |
| E3 | 11 | 8.57 giây |
| E4 | 1 | 4.33 giây |

Sequence cùng video gần ground truth nhất đứng rank 5:

```text
Dự đoán: [472.17, 473.57, 554.17, 567.77]
GT:      [531.53, 533.53, 545.60, 563.43]
Sai số:  [59.37,  59.97,  8.57,   4.33]
```

E1 và E2 nằm sâu ở rank 558 và 1517. Model nhận diện con lân/rồng tốt nhưng khó phân biệt các hành động cụ thể như bắt đầu xoay bằng hai chân trước và thời điểm bốn chân vừa chạm đất. Candidate đúng của cả bốn event vẫn nằm trong K=2000, nhưng DP chọn prefix ở một cảnh lân trước đó có score cao hơn. Đây là lỗi kết hợp giữa embedding chưa đủ nhạy với hành động và DP chỉ giữ một prefix.

### `query-p1-18` — top 3

| Event | Candidate rank tốt nhất | Sai lệch |
|---|---:|---:|
| E1 | 42 | 7.68 giây |
| E2 | 170 | 0.16 giây |
| E3 | 5 | 6.04 giây |
| E4 | 680 | 2.48 giây |

```text
Dự đoán: [107.32, 125.68, 143.08, 154.80]
GT:      [99.64,  125.52, 137.04, 152.32]
Sai số:  [7.68,   0.16,   6.04,   2.48]
```

Dù E4 chỉ đứng rank 680, ràng buộc cùng video và thứ tự thời gian loại được nhiều false positive. Các thao tác cắt nguyên liệu có đặc trưng hình ảnh tương đối rõ. Audio về tên nguyên liệu có thể cải thiện thứ hạng khi Elasticsearch hoạt động.

### `query-p2-10` — top 1

| Event | Candidate rank tốt nhất | Sai lệch |
|---|---:|---:|
| E1 | 82 | 1.60 giây |
| E2 | 7 | 0.53 giây |
| E3 | 1 | 2.70 giây |

```text
Dự đoán: [739.17, 741.23, 742.23]
GT:      [740.77, 742.70, 744.93]
Sai số:  [1.60,   1.47,   2.70]
```

Bối cảnh dây chuyền lắp ráp rất đặc trưng, E2/E3 có candidate rank cao, và ba event diễn ra gần nhau. Constraint cùng video và thứ tự thời gian loại false positive hiệu quả.

### `query-p2-22` — không tìm thấy

| Event | Candidate rank tốt nhất | Sai lệch |
|---|---:|---:|
| E1 | 107 | 0.04 giây |
| E2 | Không có trong top 2000 | — |
| E3 | 7 | 0.12 giây |

Sequence cùng video gần ground truth nhất đứng rank 8:

```text
Dự đoán: [366.00, 368.16, 406.00]
GT:      [398.04, 401.16, 405.88]
Sai số:  [32.04,  33.00,  0.12]
```

Đây là lỗi candidate recall: E2, cảnh sát chỉ tay vào vị trí cần đứng, không có candidate đúng trong top 2000. Embedding nhận diện được bối cảnh khám xét nhưng động tác chỉ tay nhỏ và ngắn. Khi E2 đúng không tồn tại trong candidate pool, DP không thể tạo sequence đúng.

### `query-p3-1` — top 3

| Event | Candidate rank tốt nhất | Sai lệch |
|---|---:|---:|
| E1 | 1069 | 1.52 giây |
| E2 | 3 | 0.24 giây |

```text
Dự đoán: [191.12, 195.24]
GT:      [189.60, 195.48]
Sai số:  [1.52,   0.24]
```

E1 nằm sâu ở rank 1069 nhưng E2 đứng rank 3. Ràng buộc cùng video và E1 xảy ra trước E2 đã đưa candidate E1 vào sequence top 3. Audio về nước sốt và món cá có khả năng cải thiện thêm.

### `query-p3-24` — top 1

| Event | Candidate rank tốt nhất | Sai lệch |
|---|---:|---:|
| E1 | 1 | 0.64 giây |
| E2 | 1 | 0.08 giây |
| E3 | 5 | 0.20 giây |
| E4 | 29 | 1.40 giây |

```text
Dự đoán: [310.48, 319.92, 324.68, 327.92]
GT:      [309.84, 319.84, 324.48, 329.32]
Sai số:  [0.64,   0.08,   0.20,   1.40]
```

Các địa danh có hình dáng kiến trúc đặc trưng; ba trong bốn event nằm trong top 5. Thứ tự xuất hiện liên tục trong cùng bản tin giúp loại false positive rất hiệu quả. Transcript chứa tên địa danh có thể làm kết quả ổn định hơn nữa.

## Phân loại lỗi

| Nhóm lỗi | Query | Diễn giải |
|---|---|---|
| Candidate recall | `query-p2-22` | E2 đúng không xuất hiện trong top 2000 |
| Sequence assembly/ranking | `query-p1-4`, `query-p1-16` | Candidate đúng tồn tại nhưng DP chọn prefix có visual score cao hơn |

DP hiện tại vẫn cho thấy lợi ích rõ ràng khi đưa candidate sâu lên thứ hạng cao:

- `query-p3-1`: E1 candidate rank 1069 → sequence rank 3.
- `query-p1-18`: E4 candidate rank 680 → sequence rank 3.

## Kết quả thử nghiệm Priority 1 và Priority 2

Đã benchmark 20 biến thể trên cùng 7 query:

- Beam DP: `B ∈ {2, 4, 8, 16}`.
- Candidate temporal dedup: cửa sổ `{1, 2, 3}` giây.
- Dedup lấy thô 5000 candidate rồi giữ tối đa 2000 candidate/event.
- Tất cả tổ hợp beam × dedup.

Toàn bộ 20 biến thể đạt cùng kết quả với baseline:

```text
MRR       = 0.3810
Recall@10 = 0.5714
Ranks     = [—, —, 3, 1, —, 3, 1]
```

Beam tạo thêm alternative prefix nhưng các alternative này overlap với đường có `raw_mean` cao hơn và tiếp tục bị temporal NMS loại. Dedup/backfill thay đổi candidate pool nhưng không đưa sequence đúng mới vào kết quả cuối; riêng event chỉ tay của `query-p2-22` vẫn không tạo được sequence đúng. Vì không có quality gain, production tiếp tục dùng baseline `beam_size=1`, không dedup và K=2000. Báo cáo đầy đủ nằm tại [`../../benchmarks/temporal_improvements.md`](../../benchmarks/temporal_improvements.md).

## Future work

### 1. K-best hoặc beam DP

Beam thuần `B ∈ {2, 4, 8, 16}` đã được thử và không cải thiện metric vì NMS vẫn chọn đường overlap có raw score cao nhất. Nếu tiếp tục hướng này, cần kết hợp beam với reranker hoặc sửa tiêu chí chọn trước NMS; chỉ tăng B không đủ.

### 2. Candidate diversity theo thời gian

- NMS candidate theo video với cửa sổ 1–3 giây.
- Giới hạn số candidate trong cùng shot.
- Phân bổ candidate budget theo video thay vì hoàn toàn toàn cục.
- Mở rộng các frame lân cận quanh candidate mạnh.

Các cửa sổ 1–3 giây với pool thô 5000 đã được thử nhưng không cải thiện metric. Bước tiếp theo nên là shot-aware diversity hoặc action-sensitive retrieval; không nên chỉ tăng cửa sổ dedup.

### 3. Action-sensitive retrieval

- Dùng embedding của clip ngắn thay vì chỉ một keyframe.
- Thử optical flow hoặc motion encoder.
- Đánh giá các video-text model như InternVideo/VideoCLIP tương đương.
- Prompt ensemble cho trạng thái trước, bắt đầu, đang diễn ra và kết thúc hành động.

Hướng này đặc biệt phù hợp với các động tác xoay/chạm đất, chỉ tay, quay thiết bị và hoàn tất đổ nguyên liệu.

### 4. Cải thiện sequence scoring

- Kết hợp mean và min để tránh một event yếu bị các event mạnh che lấp.
- Phạt candidate rank quá sâu hoặc temporal gap bất thường.
- Thưởng scene/shot continuity.
- Học trọng số riêng cho event khó.
- Dùng sequence reranker nhận toàn bộ event score, gap và context liên tục.

### 5. Benchmark audio fusion thật sự

Khi Elasticsearch hoạt động, chạy ba ablation: visual-only, audio evidence và visual+audio RRF. Ưu tiên `query-p1-18`, `query-p3-1` và `query-p3-24`, nơi transcript có thể chứa nguyên liệu, món ăn hoặc tên địa danh.

### 6. Mở rộng ground truth

- Thêm mọi sequence đúng khi nội dung xuất hiện ở nhiều bản tin hoặc đoạn không overlap.
- Mở rộng lên ít nhất 30–50 query temporal.
- Cân bằng query từ 2 đến 6 event.
- Phân nhóm object, action, cooking, surveillance, news và landmark.
- Báo cáo metric theo từng nhóm thay vì chỉ MRR tổng.

### 7. Tự động hóa diagnostic report

Benchmark harness nên lưu sequence rank, candidate rank của từng event, timestamp và sai số, nguyên nhân không có candidate, rank trước/sau audio fusion và NMS. Việc này cho phép tạo tự động failure analysis theo từng query.

## Tài liệu liên quan

- Aggregate benchmark: [`../../benchmarks/temporal_benchmark.md`](../../benchmarks/temporal_benchmark.md)
- Raw benchmark data: [`../../benchmarks/temporal_benchmark.json`](../../benchmarks/temporal_benchmark.json)
- Ground truth: [`../../benchmarks/temporal_ground_truth.json`](../../benchmarks/temporal_ground_truth.json)
- Production config: [`../../benchmarks/temporal_config.json`](../../benchmarks/temporal_config.json)
