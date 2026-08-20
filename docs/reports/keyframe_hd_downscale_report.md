# Báo cáo: Đánh giá độ an toàn khi downscale keyframes KXX về HD (720p)

**Ngày:** 2026-08-16
**Dữ liệu:** `data/2025/keyframes/K01–K20` (558.246 keyframes PNG, gốc 1920×1080)
**Nguyên tắc:** Không sửa đổi dữ liệu gốc trong `./data`. Toàn bộ thí nghiệm chạy trong `/tmp/opencode`.

---

## 1. Kết luận tổng quan

**An toàn** để downscale toàn bộ keyframes KXX về 720p cho kì thi kiểu Video Browser Showdown (VBS).

| Mặt đánh giá | Kết quả |
|---|---|
| Retrieval / tìm kiếm (embedding) | **Không ảnh hưởng** |
| Hiển thị (thumbnail) | **An toàn** (99.95% frame gần như giống hệt) |
| Hiển thị (full-res zoom, ~0.05% frame tần số cao) | Mất chi tiết mịn nếu soi kỹ |

---

## 2. Retrieval / tìm kiếm — chứng minh không ảnh hưởng

Embed 500 keyframe ngẫu nhiên bằng đúng model pipeline `PE-Core-bigG-14-448`
(open_clip, GPU RTX 3060, fp16) ở hai phiên bản: 1080p gốc và 720p downscale.

| Chỉ số | Giá trị |
|---|---|
| Cosine similarity (1080p vs 720p) | min **0.9909**, mean **0.9979**, median **0.9982** |
| Tự tìm lại chính mình trong 500 frame | **top-1 = 100%**, top-10 = 100%, rank tệ nhất = 0 |

**Giải thích:** model resize mọi ảnh về 448×448 trước khi embed. Vì 1080p và 720p
cùng tỉ lệ 16:9, sau `Resize(448)` cả hai đều thành 796×448 → center-crop 448×448
giống hệt nhau. Do đó downscale 1080→720 gần như không làm thay đổi embedding,
và kết quả tìm kiếm giữ nguyên thứ hạng.

---

## 3. Hiển thị — phân bố sai khác (mẫu 50k + 40k)

So sánh: gốc 1080p → downscale 720p (INTER_AREA) → phóng lại 1080p (LANCZOS4) → đo PSNR/SSIM.

### 3.1 Phân bố PSNR (mẫu 40.000 frame, seed=99)

| Ngưỡng | Tỉ lệ frame | Số frame (ước trên 558k) |
|---|---|---|
| < 25 dB (hỏng nặng, nhìn thấy rõ) | **0.005%** | ~28 |
| < 28 dB | 0.015% | ~84 |
| < 30 dB (thấy khác biệt) | **0.045%** | ~251 |
| < 32 dB | 0.135% | ~750 |
| < 35 dB | 1.578% | ~8.800 |
| < 40 dB | 45.9% | — |
| < 45 dB | 95.0% | — |

- **min ≈ 18.4 dB, mean = 40.3 dB, max = 54.2 dB** (min được cập nhật từ mẫu 100k; mẫu 50k trước đó cho 21 dB — đuôi phân bố dày, giá trị cực tiểu thật trên 558k có thể còn thấp hơn)

### 3.2 Top frame khác nhau nhiều nhất (mẫu 50k, seed=7)

| PSNR (dB) | Path |
|---|---|
| 18.37 | `K17/K17_V010/015481.png` |
| 19.46 | `K19/K19_V014/003736.png` |
| 21.47 | `K19/K19_V023/039321.png` |
| 22.81 | `K16/K16_V004/032328.png` |
| 23.33 | `K17/K17_V012/003111.png` |
| 23.62 | `K17/K17_V021/001848.png` |

(Danh sách đầy đủ 50 frame tệ nhất là artifact local và không được commit.)

**Nhận xét:** frame tệ nhất đều có nội dung tần số cao (chữ nhỏ, lưới, hoa văn dày)
và tập trung ở các video K19, K20, K18, K17, K14, K16. Frame ít chi tiết (edge thấp)
thì PSNR rất cao (>50 dB) — gần như không đổi.

---

## 4. Khuyến nghị

1. **Downscale toàn bộ KXX về 720p là an toàn** cho kì thi VBS-like:
   - Retrieval không đổi (đã chứng minh thực nghiệm).
   - Hiển thị thumbnail (~200–400px) thấp hơn nhiều so với 720p → dư sức.
2. **Rủi ro còn lại:** nếu kì thi yêu cầu đọc chữ nhỏ ở full-res zoom trên ~0.05%
   frame tần số cao, 720p có thể mất nét. Giải pháp thận trọng: giữ riêng ~0.05%
   frame tệ nhất ở 1080p.
3. Tiết kiệm dung lượng dự kiến: **~30–40%**.

---

## 5. File đính kèm (trong `/tmp/opencode/kf_cmp_full/`)

- `worst0_003894.png` — bản gốc 1080p của frame tệ nhất
- `worst0_003894_720.png` — bản 720p
- `worst0_003894_diff5x.png` — bản đồ sai khác (khuếch đại ×5)
- `worst0_side_by_side.png` — ghép cạnh nhau (gốc | 720p | diff)
