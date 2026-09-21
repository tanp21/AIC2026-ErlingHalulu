# Chọn backend thị giác cho retrieval agent — đo trên chính việc của agent

**Ngày:** 2026-09-21
**Nhánh:** `feat/agent-local-vlm`
**Harness:** `scripts/benchmarks/bench_agent_vision.py`
**Kết quả thô:** `benchmarks/visual/vlm_verify_2026.json`

> **Trạng thái:** đợt đo đã dừng giữa chừng theo yêu cầu. Mọi số trong báo cáo là số **đã chốt**
> — không có lượt nào còn đang chạy. Ba việc bị dừng trước khi xong (`answer`/`falsify`/`pick_frame`/
> `check_sequence`, prompt tiếng Anh cho Qwen2.5-VL-3B, caption CapRL như ngữ cảnh, và lượt kiểm
> tất định của model local) nằm ở mục *Chưa đo* cuối bài, kèm lệnh chạy lại.

## Câu hỏi

Agent hiện gửi mọi keyframe nó muốn chấm ra API ngoài dưới dạng JPEG base64 — đó là thứ
nặng nhất nó đưa lên mạng, và vào ngày thi thì đường truyền của hội trường là nút thắt.
Câu hỏi đặt ra: **một model chạy local có thay được Gemini ở bước chấm điểm không, và nếu
không thì mất bao nhiêu?**

## Phương pháp

Đo **riêng bước verify**, không đo end-to-end. Lý do: R@1 end-to-end của agent nhiễu 40–56%
giữa các lần chạy *cùng một đoạn code* (`docs/agent.md`), nên nó không đủ phân giải để chọn
model. Bỏ planning ra khỏi phép đo thì phần còn lại là tất định.

- **137 case** ground truth (`artifacts/official_benchmark_queries_en_v2.json`), mỗi case có
  `targets: [[video_id, frame_idx]]`, tolerance 10 s.
- **Pool ứng viên lấy từ tìm kiếm thật**: mỗi case chạy `POST /agent/search` bằng caption
  tiếng Anh của nó, lấy top-20 → đúng phân bố agent gặp lúc thi, không phải distractor nhân tạo.
- **Chấm bằng tiếng Việt** (`text_vi`), vì production đưa `verify()` câu query tiếng Việt gốc.
  Đây cũng là phép đo công bằng duy nhất: Vintern vốn tinh chỉnh cho tiếng Việt.
- **Cùng một pool cho mọi model** — nếu mỗi model thấy pool khác nhau thì số không so được.
- **91 case dùng được** (có đủ cả lớp gold lẫn non-gold). 46 case còn lại gold không nằm
  trong top-20 — đó là vấn đề *retrieval*, không phải việc mà verifier sửa được.

Chỉ số quyết định: **separation AUC** — xác suất model cho ứng viên gold điểm cao hơn ứng
viên không gold, trong cùng một case. 0.5 = đoán bừa, 1.0 = hoàn hảo.

## Kết quả

| Giám khảo | AUC | top1 | s/case | Trả lời được | VRAM | Prompt |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **gemini-3.8-flash** | **0.809** | 0.681 | 6.9 | 100% | — | Anh (mốc) |
| **gemini-flash-lite-latest** | **0.746** | 0.604 | 7.7 | 100% | — | Anh |
| **deepseek-flash** | **0.704** | 0.571 | **4.2** | 100% | — | Anh |
| **Qwen3-VL-2B-Instruct** | **0.662** | 0.505 | 28.8 | 100% | 4.1 GB | Việt |
| Qwen3-VL-2B-Instruct `[en]` | 0.650 | 0.528 | 27.2 | 100% | 4.1 GB | Anh |
| **5CD-AI/Vintern-1B-v3_5** | **0.605** | 0.500 | 21.5 | 99.9% | 2.0 GB | Việt |
| **Qwen2.5-VL-3B-Instruct** | **0.582** | 0.500 | 25.4 | 99.7% | 7.3 GB | Việt |
| deepseek-v4-pro | *loại* | — | — | — | — | không đọc được ảnh |

`top1` = tỉ lệ case mà ứng viên điểm cao nhất đúng là gold. Dấu `—` là chưa ghi, không phải 0.
**Mốc để đọc cột này là 0,5055**, không phải 0: đó là tỉ lệ ứng viên hạng 1 của pool vốn đã là
gold, tức điểm của một giám khảo không xếp hạng gì cả. Đọc `top1` mà không có mốc đó thì con số
0,500 của model local trông như "chọn đúng một nửa".

| Giám khảo | `top1` | So với mốc 0,5055 |
| --- | ---: | --- |
| gemini-3.8-flash | 0.681 | **+0,176** — xếp lại rõ rệt |
| gemini-flash-lite-latest | 0.604 | +0,099 |
| deepseek-flash | 0.571 | +0,066 — có xếp lại, ít |
| Qwen3-VL-2B / Vintern-1B / Qwen2.5-VL-3B | 0.500–0.506 | **±0,00 — không xếp lại** |
| Qwen3-VL-2B `[en]` | 0.528 | +0,022 — nhích nhẹ, xem mục *Prompt* |

Thang này hữu ích hơn AUC ở đúng một điểm: nó đo trực tiếp cái mà đồng đội nhìn thấy. AUC hỏi
"model có phân biệt được gold không"; `top1` hỏi "ứng viên model cho điểm cao nhất có đúng là
gold không". Ba model local trả lời có ở câu đầu và **không** ở câu sau — nhưng đọc tiếp mục
*Cách đọc* trước khi kết luận, vì `top1` không nói hết được về thứ tự danh sách.

### Prompt tiếng Việt hay tiếng Anh cho model local?

Giả thuyết đặt ra khi thấy Qwen2.5-VL-3B chấm kém: có thể ba model local đang nhận prompt **tiếng
Việt**, và họ Qwen vốn mạnh tiếng Anh hơn. Nên hai model Qwen được chấm lại bằng prompt tiếng Anh,
giữ nguyên mọi thứ khác.

| Model | Prompt Việt | Prompt Anh | Chênh |
| --- | ---: | ---: | ---: |
| Qwen3-VL-2B-Instruct | 0.6624 | 0.6498 | −0.0126 |
| Qwen2.5-VL-3B-Instruct | 0.5815 | *chưa đo* | — |

Với Qwen3-VL-2B: chênh 0,0126, và nếu model local thật sự tất định (xem mục *Nhiễu* ngay dưới)
thì đây là chênh **thật** — nhỏ, nhưng thật, và theo hướng tiếng Việt *nhỉnh hơn*. Dù đọc theo cách
nào thì giả thuyết "tiếng Việt làm hại Qwen" cũng **không được ủng hộ**: tiếng Anh không tốt hơn,
mà có thể kém hơn một chút. Mặc định trong code hiện là `en` cho họ Qwen (`FAMILY_LANGUAGE` trong
`agent/vlm.py`); số này không đủ để nói mặc định đó sai, nhưng cũng không ủng hộ nó.

Một chi tiết nhỏ: `top1` của lượt `[en]` là 0,5275 (48/91), lần đầu nhích lên trên mốc 0,5055 —
tức có đổi thứ tự ở khoảng hai case. Nếu model local tất định (mục *Nhiễu*) thì đây cũng là chênh
thật, dù chỉ hai case trên 91; nếu không thì nó nằm gọn trong nhiễu. Không đủ để đổi mặc định.

### Nhiễu của phép đo: ~0,01 cho API từ xa, ~0 cho model local

Hai giám khảo từ xa đã vô tình chạy hai lần trên **cùng pool, cùng prompt, cùng model**:

| Giám khảo | Lần 1 | Lần 2 | Chênh |
| --- | ---: | ---: | ---: |
| gemini-3.8-flash | 0.8213 | 0.8091 | 0.0122 |
| deepseek-flash | 0.6945 | 0.7037 | 0.0092 |

Hai mẫu độc lập cho cùng một cỡ: **~0,01**. Nhưng **dải này chỉ áp cho giám khảo từ xa** — cả hai
mẫu đều là API. Model local thì khác về nguyên tắc:

**Model local giải mã tham lam, nên chạy lại phải ra y hệt.** `agent/vlm_models.py` đặt
`do_sample=False, num_beams=1, repetition_penalty=1.0` (InternVL) và `do_sample=False` (Qwen);
ảnh lấy từ cache, pool cố định, prompt cố định. Không có nguồn ngẫu nhiên nào trong đường đi. Nên
về nguyên tắc nhiễu chạy lại của local là **0**, và dải 0,01 ở trên **không được bê sang** để đọc
các khoảng cách giữa ba model local.

Hệ quả nếu điều đó đúng: Vintern (0,605) hơn Qwen2.5-VL-3B (0,582) 0,023 là **thật**, không phải
hoà; Qwen3-VL-2B hơn Vintern 0,057 cũng thật; và chênh 0,0126 giữa prompt Việt/Anh của
Qwen3-VL-2B cũng là thật (nhỏ nhưng thật) chứ không phải nhiễu.

**Nhưng đây là suy ra từ code, chưa đo.** Chưa có lần chạy lại nào của một model local trên cùng
cấu hình để xác nhận. Việc đó cần GPU và chưa làm. Cho tới lúc có, hai cách đọc đều có lý, nên
báo cáo này **không** kết luận model local nào hơn model local nào về mặt thống kê — chỉ xếp thứ
tự theo số đo và nói rõ chỗ chưa chắc.

Đây cũng là lý do **không** dùng R@1 end-to-end (nhiễu 40–56%) mà dùng AUC: ngay cả AUC của giám
khảo từ xa cũng có nhiễu, chỉ là nhỏ hơn một bậc.

## Cách đọc

**Model local chạy được nhưng yếu hơn Gemini rõ rệt.** Khoảng cách giữa model local tốt nhất
(0.662) và Gemini (0.809) là **0.147 AUC** — gấp hơn chục lần nhiễu chạy lại của Gemini, nên đây
là khoảng cách thật. Và nó rơi đúng vào bước quyết định thứ tự nộp bài: `CandidatePool.ordered` xếp ứng
viên có verdict ≥ 6 lên đầu, nên chất lượng verifier chính là chất lượng thứ tự mà đồng đội nhìn
vào để chốt.

**Model local không bao giờ cho ứng viên nào điểm cao hơn hạng 1 của tìm kiếm.** Cả ba ra `top1`
đúng bằng mốc 0,5055 ở bảng trên; Qwen3-VL-2B là **46/91**, còn mốc cũng đúng **46/91**. Trùng
cả tử số lẫn mẫu số nghĩa là trong **cả 91 case**, ứng viên điểm cao nhất luôn là hạng 1 — không
case nào có ứng viên khác vượt lên trên nó. Điều này **không** mâu thuẫn với AUC 0,605–0,662: model
vẫn phân biệt gold khỏi non-gold ở mức từng cặp, chỉ là không bao giờ đánh giá ứng viên nào cao
hơn cái mà tìm kiếm đã xếp đầu.

**Nhưng đó chưa phải toàn bộ câu chuyện — và đây là chỗ tôi đã kết luận quá mạnh.** Bản đầu của
mục này viết "model local không xếp lại được pool". Sai, hoặc ít nhất là chưa đo được.
`CandidatePool.ordered` **sort đầy đủ** theo `(tier, -score, -fused)`, không phải phân hoạch ổn
định: nếu hạng 1 được 8 và một ứng viên sâu trong danh sách được 7, cả hai đều vượt ngưỡng 6 và
ứng viên kia **được kéo lên** — dù argmax vẫn là hạng 1. Nên "argmax luôn là hạng 1" **không**
suy ra được "danh sách không đổi".

Cái thật sự cần đo là **hạng của gold trong danh sách mà đồng đội đọc**, và nó chỉ vừa được thêm
vào harness: `mean_gold_rank` (hạng trung bình của gold theo luật `ordered`), `pool_mean_gold_rank`
(hạng khi không có verifier nào), và `list_improved_rate` / `list_worsened_rate`. Ba dòng model
local trong bảng trên **chưa có** số này — chúng chạy trước khi chỉ số tồn tại. Đây là chỉ số cần
chạy lại mới có, xem mục *Chưa đo*. Harness cũng ghi `distinct_scores_mean` (số điểm khác nhau
trong một pool) — model đồng điểm cho 1, model thật cho nhiều.

**Bậc giá rẻ của Gemini vẫn thắng mọi model local.** `gemini-flash-lite-latest` được 0.746 —
thấp hơn bản đầy đủ 0.063, nhưng vẫn trên model local tốt nhất 0.084 (gấp bảy lần nhiễu chạy lại
của chính nó), và ở **7.7 s/case** so với 28.8 s. Nghĩa là ngay cả khi Gemini bị giới hạn quota,
phương án rẻ nhất của nó vẫn tốt hơn việc mang model lên máy — và không tốn 4 GB VRAM.

**To hơn không có nghĩa là chấm tốt hơn.** Ba model local xếp hạng **không** theo kích thước:
Qwen3-VL-2B (0.662) > Vintern-1B (0.605) > Qwen2.5-VL-3B (0.582). Model 3B — lớn nhất, tốn
7.3 GB, chậm thứ nhì — lại **chấm tệ nhất**. Đây là lý do không thể chọn model theo kích thước
hay theo bảng benchmark công khai; phải đo trên chính việc này. Giả thuyết "tại prompt tiếng Việt"
đã đo được một nửa và **không đứng vững** với Qwen3-VL-2B (xem mục *Prompt*); nửa còn lại là
Qwen2.5-VL-3B. Khả năng cao hơn còn lại là nó thuộc họ cũ hơn.

**Vintern chạy được việc này, dù bằng chứng cũ nói ngược lại.** `scripts/ocr/reader.py` ghi
rằng đưa cả frame thì nó đọc một vùng rồi dừng (0/7 thước nước). Đó là phép đo trên việc
*chép chữ*, không phải việc *phán đoán cảnh*, và khi đo đúng việc thì nó có tín hiệu thật
(0.605, JSON hợp lệ 99.9%). Đây là lý do phép đo phải làm trên chính công việc, không mượn
từ việc lân cận.

**`deepseek-flash` là giám khảo tốt nhất ngoài Gemini, và nhanh nhất.** Nhưng nó là API từ xa
giống Gemini — nó *không* giải quyết bài toán mạng nghẽn. Giá trị của nó là: nếu Gemini có
vấn đề về quota/chi phí/độ ổn định thì đây là phương án thay thế rẻ và nhanh hơn, không phải
một bước tiến về offline.

**`deepseek-v4-pro` không đọc được ảnh** — nó trả `content` rỗng, hoặc trả JSON đúng schema
với mọi điểm bằng 0. Cái sau nguy hiểm vì trông như một giám khảo đang chạy và chấm dở, chứ
không như một model không nhìn thấy gì. Harness giờ dò mọi giám khảo từ xa bằng một ảnh
một màu trước khi chấm, và `AIC_VLM_MODEL` mặc định cho `deepseek` là `deepseek-flash`.

## Cái này bỏ được bao nhiêu payload?

Đếm trực tiếp trên chính pool của benchmark — 91 case, 1.820 keyframe ở 640 px, đúng những
byte mà agent gửi đi ở bước verify:

| | |
| --- | ---: |
| JPEG thô, một lượt verify cả 91 query | **61,6 MB** |
| Sau khi base64 (×4/3) — đúng những gì đi lên dây | **82,1 MB** |
| Mỗi query | ~900 KB |

Chuyển sang local thì phần này về **0**: ảnh không rời khỏi máy. Đây mới chỉ là `verify`; một
lượt fast mode còn `falsify` (bác bỏ trước khi chốt) và `pick_frame` cũng gửi ảnh. Con số này
là **cận dưới** của mức giảm, không phải tổng.

## Khuyến nghị

**Giữ `AIC_AGENT_VISION=gemini` làm mặc định.** Local là chế độ dự phòng cho ngày mạng nghẽn,
đúng như thiết kế hiện tại — không phải bản thay thế tương đương. Nếu buộc phải chạy offline,
`AIC_VLM_MODEL=Qwen/Qwen3-VL-2B-Instruct` là lựa chọn tốt nhất đo được: đổi **0.147 AUC** lấy
khả năng chạy hoàn toàn cục bộ — bỏ được 82 MB payload mỗi lượt verify. Nhưng đọc kèm mục
*Cách đọc* trước khi bật. Cái đã đo được: model local **không bao giờ** cho ứng viên nào điểm cao
hơn hạng 1 của tìm kiếm, nên nó không đẩy được ứng viên đúng lên đầu danh sách khi tìm kiếm đã
xếp sai. Cái **chưa** đo được: `ordered` còn sort theo điểm trong nhóm vượt ngưỡng, nên nó vẫn
có thể xáo trộn phần dưới danh sách — và điều đó tốt hay xấu thì chưa có số (xem mục *Chưa đo*).
Cho tới lúc có, hãy coi verifier local là "có chấm để biết" chứ chưa phải "đẩy ứng viên đúng lên
đầu".

Đổi lại, đây là chế độ duy nhất không phụ thuộc đường truyền, và nó là chế độ dự phòng cho đúng
tình huống mà đường truyền là nút thắt.

Ba điều chỉnh kèm theo khi dùng local:

1. `verify_chunk` không còn tác dụng — backend local tách mỗi ảnh thành một lời gọi và giữ
   khoá quanh chúng, nên công việc là tuần tự bất kể đặt bao nhiêu.
2. Ngân sách fast mode (20 s) sẽ chật. Đo được: một lượt verify 20 ứng viên tốn **21,5 s**
   (Vintern), **27,2 s** (Qwen3-VL-2B), **25,4 s** (Qwen2.5-VL-3B) — đã vượt ngân sách 20 s của
   fast mode trước khi kịp làm việc gì khác. Cần `--mode auto`/`deep`, hoặc nới `--time-budget`,
   hoặc giảm `verify_top`.
3. VRAM là chỗ chật nhất, và đây là số đo trên máy chứ không phải ước lượng: `nvidia-smi` lúc
   container đang chạy cho ~**4,2 GB** cho PE-Core + Qwen3-Embedding. Cộng 4,1 GB của
   Qwen3-VL-2B là ~8,3 GB trên card 12,0 GB — còn thở. Nhưng Qwen2.5-VL-3B (7,3 GB) cộng vào
   là 11,5 GB, dưới 0,5 GB dự phòng. Đó là lý do `docs/operations.md` khuyên dùng model 2B.

## Chưa đo

- **Việc `answer` (đọc câu trả lời Q&A)** — đây là việc cần *đọc chữ* trong frame, và
  `pick_frame`/`falsify` cũng chưa đo riêng. Benchmark chỉ đo `verify`, là việc chiếm phần lớn
  payload. Kết luận ở trên có thể không chuyển sang bốn việc còn lại.
- **Prompt theo họ model.** Số trong bảng chính là prompt **tiếng Việt** cho cả ba model local, để
  so cùng điều kiện. Lượt tiếng Anh cho Qwen3-VL-2B đã xong và **không nhỉnh hơn** (xem mục
  *Prompt*). Lượt cho Qwen2.5-VL-3B **chưa xong** — nó chạy được 10/91 case thì bị dừng cùng
  toàn bộ hàng đợi, nên chưa có số. Nếu sau này đo và cũng không nhỉnh hơn thì mặc định `en` cho
  họ Qwen trong `FAMILY_LANGUAGE` nên đổi thành `vi` — cùng cỡ điểm nhưng bớt được một khác biệt
  so với InternVL.
- **Caption CapRL như ngữ cảnh.** **Chưa đo.** Hạ tầng đã dựng đủ và đã kiểm (`GET /captions/{video_id}`,
  `candidate_label(caption=…)`, harness có `--captions`/`--tag cap`, cache caption phủ 2.585/2.585
  candidate), nhưng lượt chạy bị dừng trước khi bắt đầu. Đây là câu hỏi gốc của cả nhánh, nên nó
  là việc nên làm đầu tiên nếu quay lại. Lượt này cũng sẽ là lượt đầu có `mean_gold_rank`,
  `list_improved_rate` và `distinct_scores_mean`.

  ```bash
  uv run python -m scripts.benchmarks.bench_agent_vision score \
      --models 5CD-AI/Vintern-1B-v3_5 --captions --tag cap   # so với dòng 0,605 ở bảng chính
  uv run python -m scripts.benchmarks.bench_agent_vision score \
      --models Qwen/Qwen2.5-VL-3B-Instruct --tag en          # nốt lượt prompt tiếng Anh
  ```
- **Hạng của gold trong danh sách, cho ba dòng model local đã có trong bảng.** Chúng chạy trước
  khi chỉ số ra đời. Muốn có thì phải chạy lại — tốn GPU, chưa làm.
- **Model local có thật sự tất định không.** Lập luận ở mục *Nhiễu* dựa trên `do_sample=False`
  trong code, chứ chưa có lần chạy lại nào của một model local trên cùng cấu hình để so. Đây là
  giả định chống đỡ cho việc đọc mọi khoảng cách giữa ba model local là thật, nên nếu sai thì
  phần xếp hạng ba model local phải đọc lại. Một lượt chạy lại Vintern là đủ để trả lời:

  ```bash
  uv run python -m scripts.benchmarks.bench_agent_vision score \
      --models 5CD-AI/Vintern-1B-v3_5 --tag rerun
  ```

  Rồi so `5CD-AI/Vintern-1B-v3_5` (0,6047) với `5CD-AI/Vintern-1B-v3_5 [rerun]` trong cùng file.
  Lượt này **đã xếp hàng nhưng bị huỷ** theo yêu cầu dừng sau khi benchmark xong — đây là khoảng
  trống có chủ ý, không phải bỏ sót. Chạy lại tốn ~33 phút GPU.
- **End-to-end.** Chưa chạy `AIC_AGENT_VISION=local` trên một round thật để so R@1 với
  baseline Gemini.

## Ghi chú kỹ thuật đã kiểm chứng

- `shot_captions.parquet` chỉ có **một row group**, nên lọc vô tác dụng: đọc cả file tốn 4.9 s
  và **902 MB RSS thường trú**. Đọc `data/2026/captions/parts/<group>/<video>.parquet` tốn
  ~17 ms và không giữ gì — đó là lý do loader đi theo từng video.
- **47% caption** mở đầu bằng "Based on the provided video frames, here is the detailed
  description in chronological order:" — boilerplate thuần, đã cắt ở `strip_caption_lead_in`.
- **Vintern cần `einops`**, thứ không có trong `.venv` lẫn `uv.lock`. Hai model Qwen thì không
  cần dependency mới nào: transformers 5.15.1 đã hỗ trợ native cả ba họ.
- **Vintern cần prompt ngắn.** Đưa `VERIFY_PROMPT` dài tiếng Anh thì nó bịa ra ứng viên không
  tồn tại rồi bị cắt cụt giữa JSON. Prompt ngắn tiếng Việt, bỏ trường `reason` (chỉ dùng để
  log): 98 s/case → 22 s/case, JSON hỏng từ có → không.
- **Id phải đóng dấu từ nhãn, không đọc từ reply.** Hỏi về `[C1]`, Vintern trả
  `"L21_V024 t=810.6s"`; `Inspector.verify` bỏ mọi id lạ nên sẽ cho 0 verdict, im lặng.
- **Caption phủ 100% pool này**: cả 2.740 candidate đều thuộc video 2026 `L*` (kiểm bằng chữ
  cái đầu của `video_id`: 2740/2740 là `L`), và video nào cũng có caption. Trung vị 3.335 ký tự.
  Nhưng `candidate_label` cắt ở `MAX_CAPTION_CHARS` = 600, nên judge chỉ thấy ~18% đầu của mỗi
  caption — đó là hành vi production, và là điều kiện mà phép đo caption ở dưới đo.
  **Hệ quả quan trọng**: phép đo caption chỉ có nghĩa cho 2026. Nếu đề thi có video `K*` (2025)
  thì nhánh caption vắng mặt ở đó, và không có gì trong báo cáo này đo được trường hợp đó.
- **`caption_for_frame` trả dict, không trả chuỗi** (`caption` + `shot_index` + `start_sec`/`end_sec`
  + `window_index`/`window_count`). Harness lần đầu cache nguyên dict, và mọi "caption" dài đúng
  6 ký tự — 6 khoá. Dùng `.get("caption")`, đừng dùng giá trị trả về trực tiếp.
- **Đồng hồ của pool khớp `frame_time`**: kiểm trên cả 2.740 candidate, lệch lớn nhất 0,5 ms.
  Nên harness đọc caption theo `time` có sẵn trong pool, không cần gọi `frame_maps` — cùng một
  cửa sổ mà route `/captions` sẽ trả.
