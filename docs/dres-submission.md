# Official DRES submission (2026 final)

The organiser's final-round document defines the API and payload format. This adapter is separate from the collaboration CSV/ZIP exporter. It never submits automatically: a teammate selects a frame, previews the exact payload, and clicks **Submit to DRES**.

## Before the contest

1. Rebuild original-video PTS maps after any in-place video remux or replacement:

   ```bash
   .venv/bin/python -m scripts.indexing.build_frame_maps --workers 8
   ```

   The builder updates `indexes/frame_maps/*.npz`, `indexes/frame_maps/manifest.json`, and `id_time.npy` in the base index and every available profile. For a changed file it reuses an old decoded map only when every packet PTS and the time base exactly match that map; otherwise it decodes the video again. Restart the app after this offline step so cached search metadata uses the refreshed times. A submission is refused when the source video size/mtime does not match the manifest. `frame_idx` maps to original-video PTS; no FPS arithmetic, shot boundary, player time, or preview clip time is sent. The builder may take a long time on the full corpus. Do not start a full rebuild during an active round. For a newly mounted Batch 2 group, `scripts.indexing.build_batch2_frame_maps` also records provenance; refresh its profile `id_time.npy` before relying on search timestamps.

   The organizer's `data/2026/map-keyframes/<video>.csv` is used for indexed keyframes when it contains the selected `frame_idx`: its `pts_time` supplies the original timeline in milliseconds. This is required for Batch 2 N videos, where `frame_idx` is a relative label because the video is VFR. If an N frame is absent from the organizer map, KIS/Q&A preparation is rejected rather than interpreting that label as a decoded-frame index. Other exact video-playback frames use the verified original-video PTS map.
2. Copy `.env.dres.example` to `.env.dres` and set `AIC_DRES_BASE_URL`, `AIC_DRES_USERNAME`, and `AIC_DRES_PASSWORD`. Keep `.env.dres` off Git and restrict its filesystem permissions; only the `app` service receives it. The owner never enters credentials or a session ID in the web UI. The backend logs in when **Connect** is clicked, then keeps the returned session in runtime memory.
3. Run local tests only. The test transport uses `httpx.MockTransport`; it never contacts the contest server:

   ```bash
   .venv/bin/python -m unittest tests.dres.test_payload tests.dres.test_api -v
   cd frontend && npm test -- --run src/dresPanel.test.tsx
   ```

4. In the Search workspace, choose **KIS**, **Q&A**, or **TRAKE** from the compact task selector at the right end of the navigation bar. Click the adjacent short DRES status to open settings. The owner clicks **Connect**. The backend logs in through `POST /api/v2/login` and keeps `sessionId` in memory. A single `ACTIVE` `SYNCHRONOUS` evaluation is selected automatically. When several are active, the backend matches task type to a unique token in the evaluation name; if that is ambiguous, the settings popup offers a manual fallback. **Every submit refreshes the evaluation list immediately before posting** and validates any manual choice against that list. **Re-login** fetches a fresh session. **Reset session** clears the in-memory session; server-side credentials remain in environment configuration. The right panel is reserved for ranking and team results; DRES settings stay in the navbar popup.
5. The BTC API does not expose the current question or a DRES task ID. Verify the question against DRES (the organiser PDF says to use the eye icon on the DRES website), then select its kind in the navbar. For TRAKE, set the required event count in DRES settings. The internal task label is never sent to DRES. The evaluation ID is never treated as a query ID.

## Search and submit

- **KIS:** Open a keyframe's video, pause at the intended frame, and submit from the compact control below the player. The player requests the original file (`/video/{id}?original=1`) so a separate faststart copy cannot shift the frame timeline. The server resolves the original video file, checks frame-map provenance, converts the selected frame's PTS delta to nearest integer millisecond with integer arithmetic, and builds `start = end = selected_frame_original_timestamp_ms` as **strings**. Shot start/end are never arguments to the payload builder.
- **Q&A:** Select the exact frame and enter the answer. It is normalized on the way into the payload: uppercased, its whitespace closed up, and Vietnamese folded to ASCII (`Năm người` → `NAMNGUOI`, `Đội bóng đá` → `DOIBONGDA`). An English answer is only closed up and uppercased -- nothing is translated, and punctuation stays. The adapter previews the exact `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>` it will send.
- **TRAKE:** An event is the **source-video frame of the moment**, not the organizer keyframe nearest to it — those sit up to several seconds apart, so a snapped keyframe would answer with a different moment than the one on screen. **Add E…** therefore sends the frame the player is showing (`resolveFrameId` from the original-video PTS map; while the video plays, `frameIdAt` resolves the playhead to the frame being displayed) and names that same number and time on the button. The stepper's ±1/±10 walk source frames like every other task, and the strip of the organizer's `map-keyframes` stays a navigation aid: its tiles are labelled with the source frame each keyframe resolves to, so the number in the strip is the number that would be submitted from there. Add, remove, reorder, and inspect events in the player or the bar above search results; submit the sequence explicitly when ready. The event count is chosen by the number of frames added, up to 100. Preparation checks each ID against the video's own frame count and nothing else: the organizers' `map-keyframes` supplies KIS/Q&A timestamps and the strip's navigation, but it is not the answer space, so an event may sit between two of their keyframes. All events must belong to one video and have distinct FRAME_IDs.

All five trusted team users may submit. Only the owner may connect or reset the session. DRES credentials and session stay in backend memory/config. They are never returned by the internal API or stored in the ledger. The app is configured for one worker; each worker has its own session state.

## Submission safety

`POST /api/v2/submit/{evaluationID}?session=...` uses a dedicated `httpx` client with redirects and retries disabled. Local validation and connection failures before sending are recorded as `not_sent`. Explicit HTTP 4xx and structured negative responses are `rejected`. A lost response, write/read timeout, HTTP 5xx, or malformed successful response is `uncertain`. A 2xx JSON response without explicit rejection is `accepted` at transport level; this does **not** mean the answer scored.

The SQLite ledger in `AIC_COLLAB_DB` stores the evaluation, local task revision/label, task type, exact canonical payload and fingerprint, video/frame, actor, time, status, HTTP status, latency, and sanitized server message. It stores no session or password. A pending attempt blocks another POST. On app restart, pending attempts become uncertain. The same exact payload is blocked again within an evaluation, even when the local task label changes. After an uncertain attempt, check the DRES website and use the explicit retry control only if appropriate. Never rely on a browser timeout as proof that the answer was not submitted.

The DRES tab's **Recent attempts** list helps reconcile lost browser responses. If a browser loses its response and cannot read the ledger, its normal Submit button is disabled until an operator makes an explicit decision. Server logs include host, evaluation, task type, video/frame, status, HTTP code, and latency; they omit password and full session ID.

## Development

Use an HTTP loopback mock URL only in local development. Configure the mock in `.env.dres`; do not test against the organiser host. HTTPS is required for non-loopback URLs. The DRES client makes no request on page mount or app startup. Only **Connect**, **Refresh evaluations**, **Re-login**, and an explicit **Submit to DRES** click can contact DRES.
