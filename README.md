# AIC HCMC Search

Ứng dụng tìm keyframe theo Visual (PE-Core + FAISS), Audio (BM25 + Qwen +
Elasticsearch), hoặc Hybrid. Visual query nên viết bằng tiếng Anh; transcript
query viết bằng tiếng Việt.

## Cấu trúc repository

```text
app.py, core.py, ...  Runtime FastAPI và search
frontend/             React/Vite UI
scripts/              Indexing, embedding, extraction, benchmark và remote tools
notebooks/            Pipeline thử nghiệm; đã loại output trước khi commit
docs/                 Tài liệu vận hành, thiết kế và báo cáo Markdown
tests/                Backend regression tests
```

`data/`, `indexes/`, `result/`, model cache và `.env.secure` chỉ tồn tại local,
không được đưa lên Git. Chạy notebook từ repository root bằng
`jupyter lab --notebook-dir=.` để các đường dẫn tương đối tiếp tục trỏ đúng.

Các utility Python chạy dưới dạng module, ví dụ:

```bash
uv run python -m scripts.indexing.build_audio_index --help
uv run python -m scripts.indexing.build_frame_maps --help
uv run python -m scripts.embedding.embed_keyframes --help
```

Để chạy server không qua Docker trong lúc phát triển, build UI trước:

```bash
cd frontend && npm ci && npm run build && cd ..
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Docker tự build frontend trong image nên không cần commit `static/dist`.

## Chạy an toàn cho đội 5 người

Kiến trúc production là:

```text
4 teammate + owner -> Tailscale HTTPS :443 -> Windows localhost:8000
                                            -> HAProxy ingress (không mount dữ liệu)
                                            -> Docker app (GPU, internal network)
                                            -> Docker Elasticsearch
```

Không port-forward router, không dùng Tailscale Funnel. App và Elasticsearch
chỉ bind loopback; teammate chỉ được chạm HTTPS 443. App xác thực thêm email từ
`Tailscale-User-Login`, nên một device share nhầm vẫn chưa đủ để dùng server.

### 1. Chuẩn bị dữ liệu và secret

Yêu cầu Docker Desktop + WSL integration, NVIDIA Container Toolkit hoạt động,
`data/`, `indexes/` và Hugging Face model cache đã có sẵn.

```bash
test -f data/2026/audio/transcription.parquet && echo "data mounted"
nvidia-smi
docker info --format '{{json .Runtimes}}'
test -f .env.secure || cp .env.secure.example .env.secure
chmod 600 .env.secure
export AIC_UID=$(id -u) AIC_GID=$(id -g)
export AIC_HF_HUB_DIR="${HOME}/.cache/huggingface/hub"
```

Các email `example.com` chỉ là placeholder để test local; **phải thay hết trước
khi share server**. Sửa `.env.secure`: đúng 5 email Tailscale, trong đó `AIC_ADMIN_USER` phải thuộc
`AIC_ALLOWED_USERS`. File này đã bị Git và Docker build context loại trừ.
`AIC_HF_HUB_DIR` là bắt buộc để Compose không phụ thuộc username của máy chủ.
Nếu cache model nằm ở vị trí khác, export đường dẫn thật trước khi chạy:

```bash
export AIC_HF_HUB_DIR=/duong/dan/toi/huggingface/hub
```

App cố ý fail startup nếu thiếu allowlist/admin hoặc admin không có trong
allowlist.

### 2. Khởi động Docker

```bash
docker compose up -d --build --wait
docker compose logs -f app
```

Warm-up PE-Core, Qwen, CUDA và Flat FAISS mất khoảng 45–60 giây. Chờ log
`Application startup complete`. Chỉ có một Uvicorn worker để không nhân đôi
VRAM/index. Kiểm tra từ máy chủ:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -i http://127.0.0.1:8000/session   # phải là 401
```

Container app chạy non-root, root filesystem read-only, bỏ toàn bộ Linux
capabilities, không mount Docker socket/home/SSH keys. Code, data, index và model
cache đều read-only; chỉ `/cache` thumbnail và `/tmp` ghi được. Mạng Docker là
internal nên app chỉ ra được Elasticsearch, không ra Internet/LAN. Container
HAProxy tối giản là thành phần duy nhất publish `127.0.0.1:8000`; nó không được
mount data, index, model, secret hay Docker socket.

### 3. Cấu hình Tailscale trên Windows host

Mở PowerShell **Run as Administrator**. Không chạy Funnel.

```powershell
tailscale set --ssh=false --webclient=false
tailscale serve --bg --https=443 http://127.0.0.1:8000
tailscale serve status
```

Lấy URL `https://<machine>.<tailnet>.ts.net` từ output. Tất cả 5 người, kể cả
owner, chỉ dùng URL này; không dùng `localhost` để thao tác app. Tailscale Serve
gỡ header identity giả từ client rồi chèn identity đã xác thực trước khi proxy.

Trong Tailscale Admin Console:

1. Share **chỉ device Windows này** lần lượt cho 4 email teammate; không mời họ
   thành member của toàn tailnet.
2. Tắt Funnel; không bật Tailscale SSH hoặc device web client.
3. Thay default allow-all bằng policy tối thiểu bên dưới. Điền IP Tailscale của
   host và bốn email thật.

```jsonc
{
  "hosts": {
    "aic-server": "100.x.y.z"
  },
  "grants": [
    {
      "src": ["owner@example.com"],
      "dst": ["aic-server"],
      "ip": ["*"]
    },
    {
      "src": ["autogroup:shared"],
      "dst": ["aic-server"],
      "ip": ["tcp:443"]
    }
  ],
  "tests": [
    {
      "src": "teammate1@example.com",
      "accept": ["aic-server:443"],
      "deny": ["aic-server:22", "aic-server:5252", "aic-server:8000", "aic-server:9200"]
    },
    {
      "src": "teammate2@example.com",
      "accept": ["aic-server:443"],
      "deny": ["aic-server:22", "aic-server:5252", "aic-server:8000", "aic-server:9200"]
    }
  ]
}
```

Thêm test tương tự cho teammate 3 và 4 trước khi lưu. Không để lại grant có
`src: ["*"]`, vì nó có thể vô hiệu hóa giới hạn này.

### 4. Kiểm tra từ máy teammate

Sau khi họ accept device share và đăng nhập đúng email:

```bash
curl -I https://<machine>.<tailnet>.ts.net/
curl https://<machine>.<tailnet>.ts.net/session
```

`/session` phải trả đúng email và `is_admin: false`; Compare indexes không xuất
hiện. Truy cập trực tiếp port 22, 5252, 8000 và 9200 phải thất bại. User ngoài
5-email allowlist phải nhận 403; request không qua Serve nhận 401.

## Fair-use và hiệu năng

- Tối đa 5 search chạy toàn hệ thống, tối đa 1 search đang chạy cho mỗi user.
- Khi một user đổi query liên tục, request mới nhất thắng: request cũ đang chờ
  bị loại; kernel GPU đang chạy được hoàn tất an toàn rồi mới chạy query mới.
- Mỗi user tối đa 10 lần submit/10 giây; vượt mức nhận 429 + `Retry-After`.
- Body tối đa 16 KiB, query tối đa 2.000 ký tự, `k` từ 1 đến 200.
- Tối đa 4 FFmpeg subprocess; cùng một thumbnail chỉ sinh một lần.
- Compare chỉ owner dùng vì thao tác này có thể thay index nhiều GB trong RAM.
- IVF/HNSW cũng chỉ owner dùng và chạy độc quyền; 4 teammate dùng Flat đã warm
  để tránh đổi index nhiều GB giữa lúc cả đội đang search.
- Audit log chỉ ghi identity/path/status/latency, không ghi nội dung query; Docker
  tự xoay log (3 file x 10 MiB).

Chạy kiểm thử:

```bash
.venv/bin/python -m unittest discover -s tests -v
cd frontend && npm test -- --run && npm run typecheck && npm run build
```

Kiểm tra tài nguyên và log khi 5 người test Hybrid/video đồng thời:

```bash
docker stats
nvidia-smi -l 1
docker compose logs -f app
```

Mục tiêu: warm Hybrid ≤1,5 giây; 5 query đồng thời mỗi query ≤4 giây, không OOM;
5 video original đồng thời đạt ≥3 Mbps/người khi upload host ≥20 Mbps.

## Thu hồi teammate và rollback khẩn cấp

Khi một người rời đội: revoke device share trong Admin Console ngay. App bắt
buộc đủ đúng 5 identity, vì vậy chỉ recreate sau khi đã **thay** email cũ bằng
email teammate mới trong `.env.secure` (không để danh sách còn 4 người):

```bash
docker compose up -d --force-recreate app
```

Khóa truy cập ngay lập tức nhưng giữ container nội bộ chạy:

```powershell
tailscale serve off
```

Dừng toàn bộ dịch vụ (volume Elasticsearch và thumbnail vẫn được giữ):

```bash
docker compose stop
```

Nếu nghi ngờ lộ máy: `tailscale serve off`, revoke cả 4 shares, kiểm tra audit
log bằng `docker compose logs app`, rồi mới điều tra/re-enable.

## Lỗi thường gặp

- `data/... not found`: mount lại `data/` sau khi restart WSL.
- `permission denied docker.sock`: bật lại Docker Desktop WSL integration.
- Model offline không thấy: sửa `AIC_HF_HUB_DIR`; container không tải Internet.
- Port 8000 bận: dừng Uvicorn cũ; Compose chỉ được bind `127.0.0.1:8000`.
- 401 qua URL Tailscale: kiểm tra đang dùng **Serve**, không phải Funnel/proxy TCP.
- 403: email trong `.env.secure` phải khớp chính xác login ở `/session`.
