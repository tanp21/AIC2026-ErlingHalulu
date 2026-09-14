# Kịch bản quay video demo — AIC-HCMC — MMM 2027 (Demo Track)

Tổng thời lượng mục tiêu: **3:00** (giới hạn của MMM 2027 là tối đa 3 phút).

Đã kiểm thử trực tiếp trên hệ thống thật (profile `l2026`, 873 video, 268.254
keyframe) ngày 2026-09-08. Mọi câu query dưới đây đều đã chạy thật và đã xem
lại thumbnail kết quả — không phải suy đoán.

> **Lưu ý về hạn nộp:** Hạn nộp demo track MMM 2027 ghi là *"07 September
> 2026, 23:59 AoE"*. AoE (Anywhere on Earth = UTC-12) quy đổi sang giờ Việt
> Nam (UTC+7) là khoảng **19:00 ngày 08/09/2026** — tức hôm nay, không phải
> hôm qua. Vẫn còn thời gian, nhưng nên chốt xong sớm.

---

## 0:00 – 0:12 · Mở đầu

Cảnh: màn hình search còn trống, thấy rõ 2 ô nhập liệu và nút chuyển mode
(Hybrid / Temporal). Không cần thao tác tìm kiếm.

**Lời dẫn (gợi ý):**
> "Đây là AIC-HCMC — một ô tìm kiếm, năm cách để truy vấn 873 video truyền
> hình tiếng Việt: theo hình ảnh, theo lời thoại, theo chữ trên màn hình, kết
> hợp cả hai, hoặc theo một chuỗi sự kiện theo thứ tự thời gian."

---

## 0:12 – 0:42 · VISUAL — Tìm theo mô tả hình ảnh (tiếng Anh)

Gõ vào ô **"Visual query (English)"**, để trống ô Audio/OCR, bấm tìm. Kết quả
trả về trong ~300ms từ FAISS flat index (exact search).

**Ô "Visual query (English)":**
```
a yellow lion dance costume jumping down from a height, next to a small blue toy model ship
```

**Kết quả xác nhận đúng:** `L24_V044`, khung hình ở giây ~7.5s, **hạng 1**,
điểm số **0.350** — khớp chính xác: đầu lân vàng đang bay xuống, cạnh mô hình
tàu thủy nhỏ màu xanh.

**Lời dẫn:**
> "Chỉ cần mô tả cảnh bằng ngôn ngữ tự nhiên — không cần lướt qua từng khung
> hình — hệ thống tìm đúng khoảnh khắc trong số 268.000 khung hình gần như
> ngay lập tức."

---

## 0:42 – 1:08 · AUDIO — Tìm theo lời thoại (tiếng Việt)

Xoá ô Visual, gõ vào ô **"Transcript query (Vietnamese)"**. Đây là kết hợp
Qwen3 kNN (semantic) + BM25 trong Elasticsearch, fuse bằng reciprocal rank.

**Ô "Transcript query (Vietnamese)":**
```
đậu mùa khỉ vắc-xin khẩn cấp toàn cầu
```
*(nghĩa: "vaccine đậu mùa khỉ, tình trạng khẩn cấp toàn cầu" — bản tin về việc
WHO công bố tình trạng khẩn cấp y tế toàn cầu ngày 14/8 vì bệnh đậu mùa khỉ)*

**Kết quả xác nhận đúng:** `L22_V021`, giây 507.7, **hạng 1**, điểm **0.849**
— đúng đoạn tin về đậu mùa khỉ, có cảnh dây chuyền sản xuất vaccine và dòng
chữ chạy "Thái Lan phát hiện ca bệnh đậu mùa khỉ..." ở dưới màn hình.

**Lời dẫn:**
> "Nhánh audio tìm theo nội dung được nói, không phải hình ảnh — lời thoại
> tiếng Việt được embedding và match BM25, sau đó fuse lại."

---

## 1:08 – 1:36 · HYBRID — Kết hợp cả hai (fuse, không phải fallback)

Điền **cả hai** ô Visual và Transcript cùng lúc. Hai nhánh chạy song song rồi
fuse quanh cửa sổ thời gian chung 30 giây.

**Ô "Visual query (English)":**
```
a cyclist wearing a dark blue jersey crossing the finish line first, raising both arms off the handlebars in celebration
```

**Ô "Transcript query (Vietnamese)":**
```
về đích chiến thắng chặng đua xe đạp
```
*(nghĩa: "về đích, chiến thắng chặng đua")*

**Kết quả xác nhận đúng:** `L21_V017`, giây 875.1 — nguồn **"both"** (khớp cả
hai nhánh), audio rank #1, visual rank #9 — đúng cảnh tay đua ăn mừng băng
qua vạch đích.

**Lời dẫn:**
> "Điền cả hai ô, hai nhánh chạy song song rồi fuse quanh cùng một cửa sổ
> thời gian — bắt được cả những khoảnh khắc mà chỉ một trong hai phương thức
> không đủ để xếp hạng cao."

---

## 1:36 – 2:00 · OCR — Dùng chữ trên màn hình để LỌC kết quả visual

Giữ ô Visual, điền thêm ô **"OCR query"** bên cạnh. Đây không phải chuyển
sang mode OCR riêng — theo kiến trúc hệ thống, hễ ô OCR có nội dung thì nó
chạy song song và **lọc** kết quả visual, chỉ giữ lại những shot có chữ trên
màn hình khớp. Rất hợp để tách hai cảnh có nội dung hình ảnh giống nhau
nhưng khác địa điểm/ngữ cảnh.

**Ô "Visual query (English)":**
```
Bicycle racing
```

**Ô "OCR query":**
```
Quảng Nam
```

**Kết quả xác nhận đúng:** `L23_V013`, giây 318.9, **hạng 1**, điểm visual
0.204 — OCR khớp từ khoá `"Quảng Nam"`, đoạn chữ chạy trên màn hình đọc được
là *"...Chặng 13 - 88,5 km Đà Nẵng - Tam Kỳ (Quảng Nam)..."*. Đúng khung hình
đoàn đua về đích dưới cổng "ĐÍCH", có băng-rôn tên chặng đua ở Quảng Nam.

**Lời dẫn:**
> "Chữ trên màn hình không chỉ tìm được độc lập — nó còn LỌC được kết quả
> hình ảnh: 'đua xe đạp' có hàng chục chặng trong dữ liệu, nhưng thêm chữ
> 'Quảng Nam' thì chỉ còn đúng chặng đua diễn ra ở đó."

---

## 2:00 – 2:42 · TEMPORAL — Một chuỗi sự kiện có thứ tự (TRAKE)

Chuyển mode sang **"Temporal"**, nhập đủ 4 sự kiện theo đúng thứ tự. Đây
**không phải** 4 lần tìm kiếm riêng lẻ — một thuật toán quy hoạch động
(dynamic programming) tìm chuỗi con đúng thứ tự có điểm số cao nhất xuyên
suốt cả video, rồi trả về đúng timestamp của từng sự kiện.

| # | Sự kiện |
|---|---|
| E1 | two women taping and sealing a cardboard carton box together |
| E2 | instant noodle boxes and bread packs neatly stacked and arranged |
| E3 | a man lifting a box of instant noodles and stacking it on top of a pile of boxes |
| E4 | close up of instant noodle boxes stacked on the back of a truck |

**Kết quả xác nhận đúng:** `L30_V031`, chuỗi trải từ giây 35.1 đến 79.3
(44 giây), điểm chuỗi **0.260** — đúng 4 khoảnh khắc liên tiếp: đóng gói
thùng carton → xếp thùng mì tôm → một người xếp thùng lên chồng → cận cảnh
thùng mì chất lên xe tải cứu trợ (một đoạn phóng sự về điểm tập kết nhu yếu
phẩm hỗ trợ đồng bào vùng núi phía Bắc).

**Lời dẫn:**
> "Temporal search không phải là bốn lần tra cứu riêng biệt — đây là một
> thuật toán tìm chuỗi đúng thứ tự có điểm cao nhất trên toàn bộ video, rồi
> trả về chính xác thời điểm của từng sự kiện."

---

## 2:42 – 3:00 · Kết

**Lời dẫn (gợi ý):**
> "Năm phương thức, một tầng fusion, dưới một giây cho 873 video — được xây
> cho một đội 5 người chạy đua với thời gian tại VBS. Cảm ơn đã theo dõi."

Cắt sang title card có tên nhóm / tên hệ thống.

---

## Hai tấm ảnh chụp màn hình (phụ lục paper)

### Ảnh 1 — Hybrid fusion, giữa lúc ra kết quả
- Chụp ngay sau bước Hybrid (1:08–1:36), cả hai ô query vẫn còn hiện rõ nội
  dung đã gõ.
- Mở panel bằng chứng (evidence panel) bên phải trên **một** kết quả để thấy
  rõ nhãn "Hybrid rank" cùng điểm số của cả hai nhánh trong cùng một khung
  hình.
- Đây là tấm ảnh duy nhất chứng minh được việc fuse thật sự, thay vì hai ảnh
  riêng lẻ của từng mode.

### Ảnh 2 — Temporal, đã giải xong 4 sự kiện
- Chụp màn hình kết quả Temporal với đủ 4 thumbnail sự kiện và timestamp
  hiện cùng lúc trên màn hình.
- Đây là tính năng mà hầu hết demo tìm kiếm video khác không có — hãy chọn
  làm tấm ảnh đặc trưng nhất.
- Giữ danh sách 4 sự kiện đã nhập hiện phía trên kết quả để reviewer thấy
  được cả query lẫn câu trả lời trong cùng một ảnh.

---

## Câu hỏi dự phòng (nếu bị yêu cầu demo trực tiếp / ứng biến)

Đã kiểm thử sẵn, có điểm số thật, dùng khi cần thay thế nhanh.

| Mode | Query | Kết quả | Điểm |
|---|---|---|---|
| Visual | `a cyclist wearing a dark blue jersey crossing the finish line first, raising both arms off the handlebars in celebration` | `L23_V012`, giây 394.9 | 0.299 |
| OCR | `sữa đặc có đường` | `L26_V239`, giây 176.2 | 82.6 |
| Audio | `đậu mùa khỉ đã xuất hiện ở châu Á và châu Âu` | `L21_V017`, giây 616.0 | 0.852 |

---

## Ghi chú kỹ thuật

- Tất cả test chạy trên profile `l2026` (mặc định), 873 video, 268.254
  keyframe, ngày 2026-09-08.
- Điểm số visual/hybrid là inner-product FAISS (thang khoảng 0–1, hiếm khi
  vượt 0.4). Điểm OCR là BM25 (thang mở, không chuẩn hoá). Điểm audio là
  fused Qwen3 + BM25 (thang 0–1).
- Nếu index được rebuild trước ngày quay, **nên chạy lại từng câu query một
  lần** để xác nhận thumbnail vẫn đúng trước khi quay chính thức.
