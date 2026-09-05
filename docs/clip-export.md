# Shot Clip Export

The **Cắt clip** tab is a shot-data dashboard as well as an MP4 exporter. Pick a video to
inspect its shot timeline, indexed keyframes and transcript, scrub to the moment you care
about, pause, and export the shot under the playhead.

Keyframe images do not have to remain under `data/<year>/keyframes/`. The dashboard reads
frame IDs from `indexes/id_{video,frame}.npy` or a selected `indexes/profiles/<name>/`
source, then lazily extracts thumbnails from the MP4 through `/thumb`. **Copy ảnh** extracts
the full frame through `/frame` and writes a PNG bitmap to the browser clipboard.

## How a shot is defined

Shot boundaries come from TransNetV2, stored as one JSON per video:

```
data/<year>/shots/<group>/<video_id>.json
{"fps": 30.0, "shots": [[0, 48], [49, 459], ...], "shots_sec": [[0.0, 1.6], ...]}
```

`shots` holds inclusive `[start_frame, end_frame]` pairs in **decode order**, the same frame
identity `indexes/frame_maps` uses. That is what lets the server turn a shot into an exact PTS
window without fps arithmetic.

The 2025 (`K*`) dataset already ships these files for all 605 videos. The 2026 (`L*`) dataset
does not — generate them first:

```bash
uv run python -m scripts.extraction.detect_shots --year 2026              # all 873 videos
uv run python -m scripts.extraction.detect_shots --year 2026 --groups L21 # one group
uv run python -m scripts.extraction.detect_shots --year 2026 --workers 3  # 3 GPU processes
```

The script needs TensorFlow and the `transnetv2` package, both dev-only dependencies — it never
runs inside the production image, which is offline and read-only. It skips videos that already
have JSON, so it is safe to stop and resume, and the UI works incrementally: `GET /videos`
reports `has_shots` per video and the list marks the ones still missing.

Detection uses the same parameters as the 2025 pipeline: threshold `0.5`, minimum shot length
15 frames, cut placed at the peak of each above-threshold run.

## Datasets outside the search index

Clip export is not limited to the embedded profiles. `backend.shots.media_catalogue()` scans
`data/*/videos/*/*.mp4` and `resolve()` falls back to it whenever a video id is absent from
`video_meta.parquet`, so every media route — `/video`, `/thumb`, `/frame`, `/frame-map`,
`/shots`, `/clip` — serves the 2025 `K*` videos even though no profile embeds them. They are
listed with `indexed: false` and `fps: null` (fps lives in the index; exact timing comes from
the frame map either way) and the UI greys them so it is clear search cannot reach them.

The catalogue is cached for the process lifetime — videos do not appear at runtime. `has_shots`
is stat'ed per request, so JSON produced by a running `detect_shots` shows up on a page refresh
with no restart.

## API

| Route | Purpose |
| --- | --- |
| `GET /videos` | Every video on the data disk with `group`, `year`, `fps`, `indexed`, `has_shots` |
| `GET /shots/{video_id}?keyframe_source=<name>` | Shot bounds, available frame sources, and indexed keyframes grouped into each shot |
| `GET /clip/{video_id}/{shot_index}` | The shot as an MP4 attachment |

`/shots` trims shot bounds to the frame map before returning them: TransNetV2 counts frames
through cv2 while `FrameMap` counts PTS entries, and the two can disagree by a frame or two at
the tail of a file.

When the requested keyframe profile does not contain a video, `/shots` falls back to the
original full index when available. Unknown source names return `400`. Frame IDs outside the
exact frame map or in a gap between shots are ignored. Transcript cues come from the existing
`/transcript/{video_id}` route; the dashboard shows both cues overlapping the active shot and
a whole-video panel that follows playback.

`/clip` re-encodes rather than stream-copying, so the clip starts and ends on the exact shot
boundary instead of the nearest keyframe:

```
ffmpeg -y -copyts -ss <start> -i VIDEO -to <end> \
       -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
       -c:a aac -b:a 128k -avoid_negative_ts make_zero -movflags +faststart OUT.mp4
```

`start` and `end` come from `FrameMap.source_time`, placed **half a frame outside the shot on
both edges**. Landing exactly on a frame's PTS makes inclusion depend on how `%.6f` happens to
round that timestamp, which silently adds or drops a boundary frame; half a frame of margin is
far larger than the rounding error and still well short of the neighbouring shot. Verified on
`L21_V001`: shots 0, 12, 100 and 328 all export exactly `endFrame - startFrame + 1` frames.

Audio is kept for later multimodal models that listen.

The downloaded file is named `<video_id>_shot<NNNN>_f<start>-<end>.mp4`, e.g.
`L21_V001_shot0012_f000450-000612.mp4`, so the shot it came from stays recoverable from the
filename alone.

Two clips encode concurrently (`backend.shots._clip_slots`), separate from the four ffmpeg slots
`backend/thumb.py` uses for keyframes. Encoding writes to a temp file that is deleted once the response
has been sent.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `AIC_CLIP_MAX_SECONDS` | `120` | Shots longer than this are refused with 422 rather than tying up an encoder slot |
| `AIC_DATA_DIR` | `./data` | Where `<year>/shots/` is looked up, same as videos |

No new mounts are needed: `./data:/data:ro` already covers the shot JSON, and the clip is
streamed to the browser rather than written to a host folder.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Video listed as "chưa có shot" | `detect_shots` has not run for it yet |
| `503 shot boundaries unavailable` | Shot JSON missing or unreadable for that video |
| `503 exact frame map unavailable` | `indexes/frame_maps/<video_id>.npz` missing — run `scripts.indexing.build_frame_maps` |
| Export button stays disabled | The video is playing, still seeking, or the playhead sits in a gap between shots |
| `422 shot is longer than 120s` | A detection miss produced one huge shot; raise `AIC_CLIP_MAX_SECONDS` or re-run detection |
