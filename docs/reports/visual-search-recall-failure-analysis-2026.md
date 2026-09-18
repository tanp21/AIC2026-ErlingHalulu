# Vì sao Recall thấp? Phân tích thủ công như một thí sinh VBS

## Bối cảnh

Theo yêu cầu: đóng vai một thí sinh VBS thật, "gõ" các câu query (bản tiếng Anh đã duyệt) của cả
3 vòng 2026 vào hệ thống (dùng đúng `l2026`/PE-Core, K=1000, giống production), rồi **nhìn bằng
mắt** vào frame đích và các frame bị xếp hạng cao hơn để tìm nguyên nhân thật sự khiến
R@1/R@5/R@10/R@25 thấp — không chỉ nhìn số liệu.

**Baseline dùng để phân tích**: `caption_approved` — đúng những gì production sẽ nhận nếu người
dùng gõ 1 câu tiếng Anh đã duyệt vào ô visual query (chưa có paraphrase-fusion `B*` từ benchmark
trước, vì mục tiêu ở đây là chẩn đoán nguyên nhân gốc, không phải đo hiệu năng của 1 kỹ thuật cụ
thể).

## Recall theo từng ngưỡng (105 case, 3 vòng 2026)

| K | Recall@K | Số case |
|---|---:|---:|
| 1 | 37.1% | 39 |
| 5 | 55.2% | 58 |
| 10 | 59.0% | 62 |
| 25 | 70.5% | 74 |
| 50 | 75.2% | 79 |
| 100 | 80.0% | 84 |
| 1000 | 90.5% | 95 |
| **> 1000 (miss hẳn)** | — | **10** |

Phân bố hạng của gold: hạng 1 (39 case), 2-5 (19), 6-10 (4), 11-25 (12), 26-100 (10), 101-1000
(11), **miss hẳn (10)**.

## Đã kiểm tra bằng mắt 8 case (26-100, 101-1000, và miss) — 4 nguyên nhân gốc

### A. Ground truth sai/lệch frame (đã xác nhận rõ ràng ≥2 case)

**`query-p1-2-kis` — "An aerial view of a dam during a rainstorm"** (target rank: 868, gần miss).
Frame đích (`L28_V018` #3123) **KHÔNG PHẢI cảnh đập nước** — đó là một **bản đồ đồ họa** ("Kênh
Hà Tiên - Rạch Giá") hoàn toàn không liên quan. Đã kiểm tra thêm frame #3225 (kênh đào có cầu) và
các frame lân cận (#3050, #3180) — không frame nào khớp mô tả. Trong khi đó, **kết quả hạng 1 của
PE-Core** (`L21_V003` #1724) là **một cảnh đập xả nước từ trên cao** — khớp gần như hoàn hảo với
câu query. Đây là bằng chứng rõ ràng: **đáp án gốc bị sai/lệch video hoặc lệch frame quá xa**,
không phải lỗi retrieval.

**`query-p1-16-trake:1` — "Two golden dragons spinning in a circle"** (miss hẳn). Frame đích cho
thấy toàn cảnh sân khấu múa lân từ xa (khó thấy rõ "hai con rồng đang xoay"), trong khi hạng 1 là
một cận cảnh đầu rồng vàng-đen rất rõ nét, khớp mô tả tốt hơn nhiều về mặt thị giác thuần túy.

### B. Shot dài/liên tục, frame đích cách xa khoảnh khắc PE-Core tìm được (cùng video, cùng hoạt động)

**`query-p2-16-kis` — "Two women... ten wooden cutting boards..."** (hạng 63). Hạng 1-3 của
PE-Core đều là **CHÍNH XÁC video đích** (`L29_V014`), cùng 2 phụ nữ đan lát y hệt, nhưng ở frame
cách xa 3.488 khung hình (~2 phút, vượt xa ngưỡng dung sai ±10s). Hệ thống **đã tìm đúng video và
đúng hoạt động**, chỉ sai đúng-khoảnh-khắc.

**`query-p1-13-kis` — "A person pulls a fishing net..."** (hạng 155). Y hệt: hạng 4 là đúng video
(`L29_V021`), cùng người, cùng thuyền, cùng động tác kéo lưới, cách nhau 1.224 khung hình.

Kiểm tra shot boundary xác nhận nguyên nhân cấu trúc rõ ràng ở case tương tự
(`query-p2-34-trake:2/3`, múa lân `L24_V006`): **2 mốc sự kiện khác nhau nằm trong CÙNG 1 shot dài
31.4 giây** (784 khung hình liên tục). Trong 1 shot dài như vậy, mọi khung hình trông gần giống
nhau — phân biệt "chân trước nhấc lên" với "quay đầu sang hướng khác" bằng similarity của 1 khung
hình tĩnh là bài toán cực khó, PE-Core không có cách nào làm tốt việc này.

### C. Nội dung nhiều chữ (slide bài giảng, đề thi) — CLIP không đọc được chữ

Trong **10 case miss hẳn**, kiểm tra caption cho thấy **4/10 (40%) là nội dung slide/bài
giảng/đề thi**: "geography lecture slide... table... urban distribution", "Biology exam question
11... graph", "lecture slide showing white 3D figures...", "woman... explains the verb remember"
(video dạy tiếng Anh có slide ngữ pháp). Đã xác nhận bằng mắt case đầu tiên: frame đích đúng là
slide "Phân bố đô thị" cần tìm, nhưng hạng 1 của PE-Core là **1 slide KHÁC trong CÙNG series bài
giảng** (cùng giảng viên, cùng layout, cùng tông màu đỏ-xanh) nhưng nội dung chữ khác hẳn
("Địa lý công nghiệp" thay vì "Phân bố đô thị"). **PE-Core nắm được "đây là slide bài giảng kiểu
này" nhưng không đọc được TEXT cụ thể trên slide để phân biệt** — đúng chính xác bài toán mà
nhánh OCR (`backend/ocr_search.py`) được thiết kế để giải quyết.

### D. Múa lân/rồng chiếm áp đảo trong nhóm miss hẳn — có thể là giới hạn phương pháp đo, không phải giới hạn hệ thống

**5/10 (50%) case miss hẳn đều là TRAKE múa lân/rồng**, chỉ từ **2 video nguồn**
(`L24_V024`, `L24_V006`). Đây là phát hiện quan trọng về **phương pháp benchmark**: cả benchmark
này lẫn 2 benchmark trước đều chấm điểm mỗi sự kiện TRAKE **độc lập** (1 câu → tìm 1 frame), y hệt
KIS. Nhưng **production thật KHÔNG làm vậy** — `/search/temporal`
(`backend/temporal_search.py`) tìm **cả chuỗi sự kiện cùng lúc**, dùng DP để ép các sự kiện phải
xuất hiện **theo đúng thứ tự và gần nhau về thời gian trong cùng 1 video**. Ràng buộc thứ tự +
khoảng cách thời gian này chính là thông tin phân biệt còn thiếu ở mục B — rất có thể đã giúp
loại bỏ phần lớn nhầm lẫn giữa các khung hình múa lân trông giống nhau. **Recall TRAKE đo theo
kiểu KIS-độc-lập trong các benchmark trước có thể đã đánh giá thấp khả năng thật của hệ thống.**

## Tổng hợp nguyên nhân (10 case miss hẳn)

| Nguyên nhân | Số case | Tỷ lệ |
|---|---:|---:|
| D. TRAKE múa lân/rồng (có thể là artifact đo lường) | 5 | 50% |
| C. Nội dung chữ/slide (đúng địa hạt OCR) | 4 | 40% |
| Khác (hành động tinh vi, ví dụ "miếng cam thứ 4") | 1 | 10% |

(Case A — ground truth sai — được phát hiện ở nhóm hạng 100-1000, chưa lượng hóa được tỷ lệ chính
xác trong 10 case miss hẳn vì cần kiểm tra thủ công từng case; xem "Việc cần làm tiếp" bên dưới.)

## Khuyến nghị, theo mức độ tin cậy và chi phí

1. **Route nội dung nhiều chữ sang OCR** (tin cậy cao, chi phí thấp — hạ tầng OCR đã có sẵn). Ví
   dụ: query chứa các từ như "slide", "table", "graph", "exam", "text", "chart", "sign", "board"
   → tăng trọng số OCR trong hybrid fusion, hoặc luôn chạy OCR song song và ưu tiên khi có match
   mạnh. Đây là 40% case miss hẳn — cải thiện tiềm năng lớn nhất, chi phí kỹ thuật thấp nhất.
2. **Sửa phương pháp đo TRAKE**: chấm điểm TRAKE bằng `temporal_visual_paths` +
   `build_sequences` (DP thật của production) thay vì tách từng sự kiện độc lập, trước khi kết
   luận hệ thống "kém" ở múa lân. Cần làm việc này **trước khi** đầu tư sửa gì cho case D — có
   thể recall thật đã tốt hơn số liệu hiện tại.
3. **Audit ground truth** cho các case PE-Core trả về hạng 1 là **video khác hẳn** nhưng khớp mô
   tả rất tốt (như case dam/map) — đây là dấu hiệu mạnh của đáp án sai, nên rà lại thủ công tất cả
   case hạng >100, ưu tiên case nào rank-1 nhìn "quá đúng" so với query.
4. **Nới lỏng dung sai cho shot dài/liên tục** (mục B) — hoặc cân nhắc: nếu 1 shot dài hơn X giây,
   coi TOÀN BỘ shot là chấp nhận được cho KIS (không chỉ ±10s quanh 1 frame cụ thể), vì về mặt
   thực tế thi đấu, tìm đúng video+đúng hoạt động trong 1 shot dài liên tục đã là thành công thực
   sự với thí sinh.
5. **Triển khai paraphrase-fusion `B*`** (đã validate ở benchmark trước, +2.4/105 case) như một
   lớp cải thiện nền tảng, độc lập với các hướng trên — không giải quyết nguyên nhân gốc nhưng là
   cải thiện chắc chắn, chi phí thấp, đã kiểm chứng.

## Việc cần làm tiếp (nếu bạn muốn đào sâu hơn trước Phase B)

- Audit đầy đủ tất cả case hạng 26-1000 + miss hẳn (không chỉ 8 case mẫu) — phân loại theo 4 nhóm
  A-D ở trên, để biết tỷ lệ chính xác toàn bộ, không chỉ trên mẫu nhỏ.
- Chấm lại các case TRAKE bằng `/search/temporal` thật (production DP) để tách bạch "artifact đo
  lường" khỏi "giới hạn hệ thống thật".
- Nếu xác nhận mục 1 (OCR routing) có tiềm năng lớn: benchmark nhanh hybrid visual+OCR trên đúng
  4 case slide/text đã tìm thấy, xem OCR một mình có tìm ra không.
