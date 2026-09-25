# Prompt: set up the laptop media host

A task brief to paste into a coding agent running **on the laptop**, filling in the
`<<angle-bracket>>` placeholders first. It complements `docs/laptop-media-host.md` rather
than replacing it: that document is the procedure and the source of truth, and the prompt
tells the agent to read it instead of improvising.

The prompt is in Vietnamese because it is written for this team. Everything it points at —
the compose override, the ingress config, the scripts — is documented in English.

---

```
Bối cảnh: repo AIC_HCMC vừa được pull về laptop Windows này. Máy KHÔNG dùng WSL2,
chỉ có Docker Desktop. Nó sẽ làm MEDIA HOST cho team 5 người: phục vụ SPA và
/thumb /frame /video /clip /api/dres từ đĩa local, forward search/agent/collab
sang vast.ai instance qua SSH tunnel. Team nối vào bằng switch LAN + Basic Auth.
WiFi dùng cho internet và nộp DRES.

ĐỌC TRƯỚC, đây là nguồn chân lý — đừng tự nghĩ ra quy trình khác:
  docs/laptop-media-host.md   (toàn bộ quy trình + bảng troubleshooting)
  compose.media.yaml          (những gì bị bỏ khỏi compose gốc và vì sao)
  haproxy.lan.cfg             (ingress Basic Auth)

Thông tin máy này:
  - Repo ở: C:\aic
  - Video đã tải sẵn ở: <<ĐƯỜNG DẪN>>   ← phải khớp layout data/<year>/videos/<group>/
  - IP Ethernet trên switch: <<IP>>      (ipconfig; KHÔNG phải IP WiFi)
  - Instance: <<host>>:<<port>>, user root, key <<đường dẫn .ssh>>
  - 5 login + admin: tôi sẽ tự tạo .env.secure, đừng đoán
  - Payload copy từ desktop sang bằng: <<CÁCH>>

LÀM THEO THỨ TỰ, dừng lại hỏi tôi nếu thiếu thông tin:
  1. `docker info` — xác nhận Docker Desktop đang chạy Linux container
  2. Kiểm tra layout data/ khớp data/<year>/videos/<group>/<ID>.mp4|mov
  3. Tạo .env (từ .env.laptop.example), .env.secure, .env.dres
  4. Copy từ desktop: indexes/frame_maps/ (CẢ manifest.json), indexes/profiles/
     (BỎ file *.ip), indexes/{video_meta.parquet,id_video.npy,id_frame.npy},
     data/<year>/shots/, data/2026/keyframe_thumbs/ (22GB, chỉ cần cho nhóm N),
     data/<year>/videos_faststart/ (tùy chọn). Dùng `robocopy /COPY:DAT /DCOPY:DAT`
     hoặc `rsync -a` để giữ nguyên timestamp.
  5. `python -m scripts.indexing.build_frame_maps`
     → BẮT BUỘC, nếu không DRES từ chối mọi lượt nộp. Không có ffmpeg thì dùng
       `--resign-only --data-dir <đường dẫn>`; nó từ chối video đã đổi kích thước.
  6. `python -m scripts.media.check_media_host`  → phải in "media host is ready"
  7. `python -m scripts.media.make_lan_users`    → sinh haproxy-users.cfg
  8. Thêm firewall rule cho TCP 8000, giới hạn RemoteAddress về subnet LAN
  9. `docker compose -f compose.yaml -f compose.media.yaml up -d --build --wait`
 10. Verify: /healthz từ chính máy; một ảnh JPEG thật từ /thumb/L21_V001/0?w=320
     trong container; /healthz qua core-tunnel; và từ một máy khác trong LAN mở
     http://<<IP>>:8000/ phải hiện prompt mật khẩu rồi vào được SPA.

QUY TẮC:
  - Không đoán mật khẩu, API key, hay login. Hỏi tôi.
  - Không commit .env*, haproxy-users.cfg, hay bất cứ gì trong data/ indexes/.
  - Nếu /data đọc không được (app báo 404 "video file not found"), báo tôi ngay:
    đó là vấn đề phân quyền bind mount của Docker Desktop, cách sửa nằm ở
    compose.media.yaml chứ không phải ở code.
  - Báo cáo lại nguyên văn output của check_media_host và mọi bước lỗi.
```
