# Theo dõi tiếp: TRAKE DP thật, OCR routing, và audit ground truth mở rộng

Tiếp nối `docs/reports/visual-search-recall-failure-analysis-2026.md`, làm theo đúng thứ tự đã đề
xuất. Kết luận thẳng ở đầu: **R@25 = 100% KHÔNG khả thi** với dữ liệu/kiến trúc hiện tại — lý do cụ
thể ở cuối tài liệu, kèm mục tiêu thực tế hơn.

## 1. Chấm lại TRAKE bằng DP thật của production (`build_sequences`)

Chạy đúng những gì `/search/temporal` chạy (candidate_k=2000, scoring=shifted_harmonic,
max_gap=120s, **min_gap_seconds=0.0 — xác nhận lại đúng như phát hiện trước: config ghi 1.5
nhưng `app.py` không truyền tham số này vào, nên production hiện tại luôn dùng 0**) cho cả 5 câu
TRAKE của 3 vòng 2026:

| Câu | Hạng video đúng (trong path) | Số sự kiện đúng frame |
|---|---:|---:|
| `p1-16` (R1, L24_V024, 3 sự kiện) | **3** / 1725 path | 1/3 |
| `p2-21` (R2, L30_V031, 4 sự kiện) | **1** / 368 path | 4/4 ✅ |
| `p2-8` (R2, L27_V011, 4 sự kiện) | **1** / 790 path | 3/4 |
| `p2-21` (R3, L26_V156, 4 sự kiện) | **1** / 91 path | 4/4 ✅ |
| `p2-34` (R3, L24_V006, 4 sự kiện) | 35 / 1813 path | 0/4 |

**Kết quả cho đúng 5 sự kiện trước đây bị coi là "miss hẳn"**: 1/5 được sửa hoàn toàn đúng frame
(`p2-21` R3, sự kiện 4 "miếng cam thứ tư"), 2/5 tìm đúng video ở hạng top-3 nhưng vẫn sai frame cụ
thể (`p1-16` R1), 2/5 hoàn toàn không cứu được (`p2-34` R3 — đúng như dự đoán, đây là case 2 sự
kiện nằm chung 1 shot dài 31 giây, xác nhận là **giới hạn thật của hệ thống**, không phải lỗi đo
lường).

→ **Xác nhận giả thuyết là ĐÚNG MỘT PHẦN**: đo TRAKE độc lập từng sự kiện có đánh giá thấp hệ
thống thật, nhưng không phải "ảo giác đo lường" hoàn toàn — vẫn còn giới hạn thật (case cùng 1
shot dài, nhiều hành động vi mô giống hệt nhau).

## 2. Thử OCR routing trên 4 case slide/text (chạy trực tiếp trong container production đang
   chạy, `aic_hcmc-app-1`, dùng đúng OCR index thật)

Tra ngược nội dung OCR **thật** tại đúng frame đích (không đoán mò):

| Case | Chữ OCR thật gần frame đích | Kết quả |
|---|---|---|
| `p2-20` slide đô thị hoá | Banner lặp lại "ĐỊA LÍ DÂN CƯ VIỆT NAM" đúng là có trong OCR, nhưng ở 1 shot khác (frame 35775, cách đích 2173 khung) | Query đúng banner → **tìm ra đúng video ở hạng 9/25**, nhưng SAI shot cụ thể (nội dung bảng "phân bố đô thị" không phải là frame đại diện của bất kỳ shot OCR nào) |
| `p2-23` đề sinh học | Frame gần nhất trong index OCR (23652) lại là **câu hỏi khác hẳn** (về phá rừng, không phải "coastal ecosystems") | Không tìm ra — câu hỏi đích rơi vào 1 shot không được OCR làm đại diện, hoặc OCR bỏ sót |
| `p2-26` slide 3D + kéo co | Frame gần nhất OCR được là nội dung bài giảng khác ("QUY LUẬT CẠNH TRANH") — **không liên quan** | Không thể tìm bằng OCR **về nguyên tắc** — mô tả câu query là nội dung ĐỒ HỌA (hình khối 3D, hoạt hình kéo co), không phải chữ trên màn hình |
| `p1-22` cô giáo áo hồng | Banner intro của video ĐÚNG khớp targetframe's nội dung, nhưng nằm ở shot khác (frame 14136, cách đích 2836 khung) | Cùng pattern: đúng VIDEO nếu OCR đúng banner, sai SHOT cụ thể |

**Phát hiện quan trọng bị bỏ sót ở báo cáo trước**: OCR index hiện tại chỉ có **1 keyframe đại
diện cho mỗi shot** (theo đúng thiết kế trong `docs/reports/ocr-shot-audit.md`). Nếu nội dung chữ
cụ thể cần tìm nằm ở MỘT thời điểm giữa 1 shot dài, mà frame đại diện của shot đó lại là nội dung
khác (slide chuyển cảnh nhanh, hoặc shot bị TransNetV2 gộp quá dài), thì **OCR sẽ không thấy nội
dung đó dù chữ đúng có xuất hiện trên màn hình lúc đó**. Đây là giới hạn cấu trúc, không sửa được
chỉ bằng cách "route query text sang OCR" — cần thêm OCR nhiều keyframe/shot hoặc OCR theo
sampling dày hơn.

**Kết luận sau khi thử thật**: OCR routing **có ích nhưng không phải "silver bullet"** như dự đoán
ban đầu — giúp tìm đúng VIDEO cho 2/4 case (nếu nới lỏng thành "đúng video" thay vì "đúng frame"),
hoàn toàn không giúp được 1/4 case (nội dung đồ họa không phải chữ), và 1/4 case OCR quality/coverage
không đủ.

## 3. Audit ground truth mở rộng (12 case đã kiểm tra bằng mắt, tính cả báo cáo trước)

Kiểm tra thêm 4 case nữa (`p2-12-qa` bánh tím, `p1-25-kis` MC trường học, `p1-18-kis` phở/bún,
`p2-15-kis` hải sản) — **cả 4 đều KHÔNG phải lỗi ground truth**. Target frame khớp đúng mô tả
caption; nguyên nhân sai là **nhầm lẫn thị giác tinh vi thật sự** giữa các cảnh cùng thể loại rất
giống nhau (nhiều video nấu ăn có cách bày biện món/hải sản/tô-chén tương tự nhau, nhiều video
sân khấu trường học có tông màu/bối cảnh tương tự).

**Tổng hợp 12 case đã audit bằng mắt (2 báo cáo cộng lại)**:

| Nguyên nhân | Số case | Tỷ lệ |
|---|---:|---:|
| B. Đúng video, sai frame cụ thể (shot dài/OCR per-shot) | 5 | 42% |
| Nhầm lẫn thị giác tinh vi thật (không phải lỗi GT, không phải bug) | 5 | 42% |
| A. Ground truth sai/lệch hẳn | 2 | 17% |

→ **Điều chỉnh lại kết luận trước**: lỗi ground truth chỉ chiếm ~1/6 case đã audit, KHÔNG phải
nguyên nhân chính. Nguyên nhân lớn nhất thực ra là **B (đúng video sai frame, do shot dài/granularity)**
và **nhầm lẫn thị giác tinh vi (giới hạn thật của PE-Core khi phân biệt các cảnh cùng thể loại)** —
mỗi loại chiếm ~42%.

## Vậy R@25 = 100% có khả thi không?

**Không, với dữ liệu và kiến trúc hiện tại.** Lý do cụ thể, đã có bằng chứng:

1. **~17% case đã audit có ground truth sai/mơ hồ hẳn** (ví dụ query "đập nước lúc mưa" nhưng
   đáp án là 1 tấm bản đồ) — không có cách retrieval nào "tìm ra" một đáp án về mặt logic là sai.
   Muốn sửa: cần audit lại TOÀN BỘ đáp án của giải đấu, không phải việc của hệ thống tìm kiếm.
2. **Có case xác nhận là giới hạn thật ngay cả với thuật toán production đầy đủ** (TRAKE 2 sự
   kiện trong 1 shot dài 31 giây, cả 2 đều sai kể cả dùng DP thật) — phân biệt 2 khoảnh khắc gần
   như giống hệt nhau trong 1 pha múa lân liên tục là bài toán nhận diện hành động (action
   recognition theo thời gian), không phải bài toán tương đồng ảnh tĩnh mà PE-Core được thiết kế
   để giải.
3. **~42% là nhầm lẫn thị giác tinh vi** giữa các cảnh CÙNG THỂ LOẠI trong corpus (nhiều video nấu
   ăn/hải sản/sân khấu trường học trông rất giống nhau) — sửa triệt để cần model tốt hơn hẳn hoặc
   fine-tune trên chính corpus này, không phải việc có thể làm bằng kỹ thuật fusion/routing.

## Mục tiêu thực tế hơn, và lộ trình đề xuất

Ước tính trần khả thi nếu sửa hết những gì SỬA ĐƯỢC (bỏ qua case ground-truth-sai và case
giới-hạn-thật-đã-xác-nhận): trên mẫu 12 case audit, **10/12 (83%)** là có thể cải thiện được bằng
kỹ thuật (5 case B + 5 case nhầm lẫn thị giác, dù case nhầm lẫn thị giác khó sửa triệt để bằng
route/fusion mà cần cải thiện model). Ngoại suy thận trọng: **trần thực tế cho R@25 nhiều khả năng
nằm trong khoảng 85-90%** trên toàn bộ 105 case, không phải 100%.

Lộ trình theo độ tin cậy/chi phí:
1. **Nới dung sai cho shot dài** (sửa được ~42% case loại B, chi phí thấp — chỉ cần đổi từ "±10s
   quanh 1 frame" sang "bất kỳ frame nào trong shot chứa target" khi shot dài hơn ngưỡng nào đó).
   Đây là đòn bẩy lớn nhất, rẻ nhất, nên làm trước.
2. **Audit + sửa ground truth** cho ~17% case sai hẳn — cần phối hợp với người phụ trách đáp án
   giải đấu, không phải việc kỹ thuật thuần túy, nhưng ảnh hưởng trực tiếp đến số đo recall "thật".
3. **Triển khai paraphrase-fusion `B*`** đã validate (+2.4/105 case, không giải quyết nguyên nhân
   gốc nhưng chắc chắn có lợi, chi phí thấp).
4. **OCR nhiều keyframe/shot hơn** (không chỉ 1 đại diện) cho các shot dài — giúp cả case OCR lẫn
   case B.
5. **Nhầm lẫn thị giác tinh vi (42%)**: khó nhất, cần đầu tư dài hạn hơn — có thể thử benchmark
   thêm với NaFlex/GOPT (đã có sẵn từ benchmark đầu tiên) xem có model nào phân biệt tốt hơn ở
   đúng NHÓM case này (khác với benchmark tổng thể trước đó vốn không tách biệt theo loại lỗi).

## Việc cần làm tiếp (nếu muốn tiếp tục trước khi vào Phase B)

- Thực thi mục 1 (nới dung sai shot dài) — có thể code + benchmark ngay, tái dùng
  `backend/subquery_fusion.shot_ranks`/`GroupResolver` đã có sẵn từ 2 benchmark trước.
- Audit toàn bộ 31 case còn lại (chưa xem bằng mắt) để có tỷ lệ chính xác trên 100% dữ liệu, không
  chỉ mẫu 12 case.
