# Report: Server Performance Optimization (Post Competition-Day Slowdown)

**Date:** 2026-09-05 to 2026-09-06

**Trigger:** The team competed in a live round and found `/search` slow under 5 concurrent users, and clicking play on a video took 3–5 minutes to start.

**Scope:** This covers everything found and fixed in that investigation — infrastructure/config fixes identified up front, plus a second wave of algorithmic bugs that only surfaced once the fixes were load-tested end to end with 5 real accounts.

## 1. Executive summary

Two independent problems were conflated in the original complaint:

- **Video-open latency** (3–5 min) was caused by a HAProxy bandwidth limiter sharing one bucket across all users instead of one per user, compounded by the 2026 video corpus lacking "faststart" (its `moov` atom sat at the end of the file, forcing an extra round trip before any byte could play).
- **Search latency under concurrency** turned out to have five separate causes, only two of which were anticipated going in (FAISS thread oversubscription, per-request thread pool churn). The other three — an undersized cache, a missing GPU batching path, and an O(n×m) fusion loop — were invisible until the system was actually driven by 5 concurrent accounts issuing realistic hybrid queries pooling the full `SEARCH_RESULT_LIMIT` (10,000 results). Fixing them dropped **5-concurrent hybrid search p50 from 8.8s to 2.9s**, inside the documented 4s target.

| Metric (5 concurrent users, hybrid search) | Before | After |
| --- | --- | --- |
| Search p50 | 8823 ms | **2902 ms** |
| Search p95 | 10727 ms | **3262 ms** |
| Video time-to-first-byte p50 | 357 ms | **73 ms** |
| Video time-to-first-byte p95 | 2861 ms | **797 ms** |
| Original complaint (video open) | 3–5 min | **< 1 s (p95)** |

## 2. Video-loading fixes

### 2.1 HAProxy shared-bucket bandwidth limiter

`haproxy.cfg`'s `teammate_video` backend rate-limited video responses using a stick-table keyed on the constant `int(1)` — every non-admin user's video traffic drew from **one shared 24 Mbps bucket**, so one person streaming could starve the other four. Measured home upload was ~49–300 Mbps, far above the cap, so this was a self-inflicted bottleneck, not a real network limit.

**Fix:** keyed the limiter on `src` (client IP) instead, with its own per-user 20 Mbps allowance:

```diff
- stick-table type integer size 1 expire 1m store bytes_out_rate(1s)
- filter bwlim-out teammate_video_limit limit 24m key int(1) table teammate_video
+ stick-table type ip size 10k expire 1m store bytes_out_rate(1s)
+ filter bwlim-out teammate_video_limit limit 20m key src table teammate_video
```

### 2.2 Missing "faststart" on the 2026 corpus

Verified via `ffprobe -v trace` that 2026 (`L*`) videos had `mdat` before `moov` — a player must read to the tail of a 90–190 MB file before it can start decoding anything. The 2025 (`K*`) corpus was checked too and found to **already** be faststart (a false assumption early on wasted a large remux batch on the 2025 videos before this was caught — see §5).

**Fix:** replaced all 873 2026 videos in place with a lossless `ffmpeg -c copy -movflags +faststart` remux. Verified byte-identical decoded output at first/middle/last frame on both H.264 and AV1 samples before trusting the batch, and full-corpus duration + spot-check integrity after.

### 2.3 Thumbnail/frame extraction throughput

`backend/thumb.py`'s `ffmpeg_command()` didn't pin `-threads`, so each call auto-threaded across all visible cores; N concurrent extractions oversubscribed the CPU by a further factor of N (load average measured >27 on a 12-thread host running only 8 concurrent extractions). Pinning `-threads 1` per call and raising the semaphore from 4 → 12 (now `AIC_THUMB_FFMPEG_SLOTS`) let concurrency scale close to linearly up to one call per logical core (~35 req/s ceiling, up from ~19–27 req/s oversubscribed).

The disk cache cap was also raised 5 GiB → 20 GiB (`AIC_THUMB_CACHE_MAX_BYTES`) to hold a full corpus pre-warm without evicting itself (see §3).

## 3. Thumbnail pre-generation + the keyframes/ cleanup

The team's plan to switch `embed_keyframes.py` from reading pre-extracted `data/<year>/keyframes/*.png` to pulling frames live by `frame_id` raised a real question: is live ffmpeg extraction fast enough for 5–10 people scrolling a 10,000-result page?

Measured: bulk concurrent extraction plateaus at **~35 req/s** on this host regardless of concurrency past ~12 (confirmed via testing up to fully unbounded — no thrashing, just a flat CPU-bound ceiling). At that rate, a cold 10,000-thumbnail scroll takes minutes. The fix chosen: **pre-generate** thumbnails for every indexed keyframe once, offline (`scripts/media/pregenerate_thumbnails.py`, new), populating the same on-disk cache `/thumb` already reads — no request-path code change needed, just a warm cache.

- Full corpus: 855,632 indexed keyframes, ~8.3 hours wall time, **0 failures** (after fixing the two corrupted source videos in §5), 12 GB of cache.
- `data/<year>/keyframes/*.png` (604 GB total, confirmed unused by any live-serving code path) was deleted after confirming captioning (`scripts/captioning/`) and shot-exclusion review (`scripts/indexing/suggest_shot_exclusions.py`) — the only other consumers — will re-extract from MP4 on demand going forward, same as the new embedding plan.

### 3.1 Bug: `_prune_cache()` was O(n²) over a long run

`backend/thumb.py`'s cache-eviction helper called `CACHE_DIR.glob("*.jpg")` — a full directory listing — **unconditionally on every cache write**, before checking whether the 20 GB cap had even been reached. With the cache growing past tens of thousands of files during the pre-generation run, this made every write cost O(current cache size), and the whole run O(n²). Measured impact: throughput decayed from ~18/s to ~7/s over the first hour with CPU load *normal* (not oversubscribed) — a different signature from the ffmpeg-threading bug in §2.3, initially mistaken for a recurrence of it.

**Fix:** only glob the directory when the running byte count actually exceeds the cap (needed only for the eviction pass itself); the running total is otherwise tracked incrementally.

## 4. Search-concurrency fixes

### 4.1 FAISS OpenMP thread oversubscription (anticipated)

No code capped FAISS's thread pool; each concurrent flat-index search spawned threads across all cores, and `LatestWinsCoordinator` permits 5 concurrent searches — up to 5× oversubscription on a 6-core host. Capped via `faiss.omp_set_num_threads()` at import (`AIC_FAISS_OMP_THREADS`, default 2).

### 4.2 Per-request `ThreadPoolExecutor` churn (anticipated)

`backend/app.py`'s hybrid/OCR/temporal fan-out (visual+audio in parallel) created a fresh `ThreadPoolExecutor(max_workers=2)` per request. Replaced with one shared, module-level `_SEARCH_FANOUT_EXECUTOR` (10 workers) for the process lifetime.

### 4.3 `frame_maps.load()` cache thrashing (found via load test)

**This was the single largest contributor**, and was invisible in isolated single-request testing. `backend/frame_maps.py`'s `load()` was `@lru_cache(maxsize=32)`. Both `visual_vector_search` and `audio_search_response` build their **full** result pool eagerly — up to `SEARCH_RESULT_LIMIT` (10,000) — calling `frame_maps.load(video_id)` once per result. A ranked pool of 10,000 hits routinely spans hundreds of distinct videos, so the 32-entry cache was evicted and reloaded on nearly every call.

Measured: a solo warm hybrid query took **8–12 seconds**, of which profiling attributed ~8s to the visual branch alone despite FAISS search itself taking only ~100ms. Isolated benchmark: 200 distinct cache-thrashing loads cost ~1.6ms/call — scaled to 10,000 calls, consistent with the observed multi-second cost.

**Fix:** the corpus is ~1,500 videos at ~30KB/map (47.7MB total) — there is no reason to bound this cache at all.

```diff
- @lru_cache(maxsize=32)
+ @lru_cache(maxsize=None)
  def load(video_id):
```

Effect: solo warm hybrid latency's visual component dropped from **8094ms → 393ms** (≈20×).

### 4.4 Qwen audio embedding had no GPU batching (found via load test)

`backend/core.py` already routes PE-Core (visual) text encoding through `TextEmbeddingBatcher` — a single dedicated thread that coalesces concurrent encode calls into one batched GPU forward pass (10ms window). `backend/audio_search.py`'s `encode_query()` for Qwen had no equivalent: 5 concurrent hybrid searches each called `.encode()` independently, and un-batched concurrent GPU inference calls don't overlap efficiently.

Measured: audio-only search at 973ms solo vs. **3.6–3.8s at 5 concurrent** (~4× degradation) despite Elasticsearch itself sitting at <1% CPU during the test (ruling out the ES side).

**Fix:** reused the existing `TextEmbeddingBatcher` class (already generic — it takes an encoder callback) for Qwen, mirroring the PE-Core lifecycle (`start_audio_text_batcher()` / `close_audio_text_batcher()` hooked into the FastAPI lifespan). New tests: `tests/test_audio_text_batcher.py`.

Effect (combined with §4.5): audio-only at 5 concurrent dropped to **1.66–1.84s**.

### 4.5 `nearest_frame()` reloading index metadata per hit (found via profiling)

`backend/core.py`'s `nearest_frame()` — called once per audio hit inside `audio_search_response`'s loop (again, up to 10,000 times per request) — called `load_meta(profile)` internally on **every** invocation, even though the caller already had that same data from a call one line above the loop. `load_meta()` itself is a cheap dict lookup once cached, but repeating the resolution logic ~10,000 times per request, with the GIL held on every call, was profiled as **the single largest chunk of per-request time** for audio search once §4.3 and §4.4 were fixed (`cProfile`: `nearest_frame`/`load_meta` combined ~0.3s of a 0.725s solo request).

**Fix:** `nearest_frame()` now accepts an optional pre-loaded `meta` tuple; `audio_search_response` loads it once before the loop and passes it down.

### 4.6 `fuse_search_results()` was O(n×m) (found via profiling — the largest single win)

For every visual hit (up to 10,000), the hybrid fusion step re-scanned **all** audio hits (up to 10,000) via a list comprehension filtering on `video_id` equality — up to **100 million comparisons per hybrid request** in the worst case. This is why hybrid mode (6.6–11s at 5 concurrent) was consistently slower than either branch alone (visual ~0.9–1.15s, audio ~1.7–1.8s at 5 concurrent after the fixes above) — the gap wasn't explained by "the slower of the two branches," it was this fusion step.

**Fix:** group audio hits by `video_id` once (`O(m)`), then each visual hit only scans the (typically short) list of audio hits for its own video, turning the match into `O(n + m)`. Output is provably identical — the grouping doesn't change which candidates are considered, only how fast they're found; all 21 existing hybrid-fusion tests pass unchanged.

Effect: solo warm hybrid request **0.76s** (down from 1.3–1.8s after §4.3+§4.4 alone); 5-concurrent hybrid search **p50 8823ms → 2902ms, p95 10727ms → 3262ms**.

## 5. Data-integrity incident (unrelated to the above, discovered mid-session)

While remuxing 2025 videos under the (incorrect) assumption they also lacked faststart, the host's WSL2 virtual disk filled up (creating full duplicate faststart copies of the entire 230GB corpus, on top of the pre-existing 604GB `keyframes/`, exceeded the underlying Windows drive's real free space) and started throwing genuine `EXT4-fs I/O error` / `Buffer I/O error` messages, forcing an emergency read-only remount.

**Resolution:**
- Confirmed (after a host restart) this was disk-space exhaustion, not failing hardware.
- Switched strategy to **in-place replacement** for the (genuinely faststart-lacking) 2026 videos instead of parallel copies — zero net extra disk space instead of +230GB.
- A fast-scan discovered the 2025 corpus was **already** faststart end-to-end (all 605 files, including groups never touched by the crashed remux job), so no 2025 remux was actually needed at all.
- Full-decode integrity verification on the completed batch surfaced **two genuinely corrupted files** unrelated to any of this session's writes: `K09_V017.mp4` (AV1 bitstream corruption ~64% into the file) and `K09_V019.mp4` (AAC decode error). Both were re-downloaded via a targeted **HTTP Range request against the source ZIP's central directory** (parsing the ZIP64 EOCD/local-file-header by hand to fetch only the ~420MB needed, instead of the full 9.24GB archive), verified with a full decode pass (0 errors) and a byte-for-byte match against the existing `frame_maps` at the specific frame indices that had been failing, then swapped in.
- `scripts/media/pregenerate_thumbnails.py` gained a `--video-ids` flag specifically so this "redownload one corrupted file → backfill its cache" workflow is a single repeatable command going forward.

**Lesson applied:** a lightweight `is_faststart()` check (peeking at the first ~64KB of a file's top-level MP4 boxes, no `ffprobe` needed) should be run **before** any future full-corpus remux, rather than assuming a whole year's corpus is uniformly encoded from one verified sample.

## 6. Files changed

| File | Change |
| --- | --- |
| `haproxy.cfg` | Per-user (`key src`) video bandwidth limit instead of shared global bucket |
| `backend/core.py` | FAISS thread cap (`AIC_FAISS_OMP_THREADS`); `nearest_frame()` accepts pre-loaded `meta` |
| `backend/frame_maps.py` | `load()` cache unbounded (was `maxsize=32`) |
| `backend/audio_search.py` | Qwen encoding routed through `TextEmbeddingBatcher` (new `start/close_audio_text_batcher`) |
| `backend/app.py` | Shared `_SEARCH_FANOUT_EXECUTOR`; `fuse_search_results()` O(n+m) via `video_id` grouping; `audio_search_response` loads meta once; device-confirmation logging in `warmup_search()` |
| `backend/thumb.py` | `ffmpeg -threads 1` pin; semaphore 4→12; cache cap 5GB→20GB; fixed O(n²) `_prune_cache` |
| `backend/shots.py` | `streaming_path()` helper preferring a faststart remux, falling back to the original (thumbnails/shots/frame_maps never touch it — only `/video` does) |
| `compose.yaml` | `cpus:` allocation for `elasticsearch` (3) and `app` (8) |
| `scripts/media/faststart_remux.py` | New: lossless per-video faststart remux, `--video-ids`-free (whole-year batch) |
| `scripts/media/pregenerate_thumbnails.py` | New: offline thumbnail cache warm for every indexed keyframe; `--video-ids` for targeted re-runs |
| `scripts/loadtest/simulate_five_users.py` | New: 5-account concurrent hybrid-search + video-TTFB benchmark |
| `tests/test_runtime_mounts.py` | Updated HAProxy assertions; new `streaming_path()` test |
| `tests/test_audio_text_batcher.py` | New: mirrors `tests/test_text_batcher.py` for the audio batcher |

## 7. Verification

- `.venv/bin/python -m unittest discover -s tests -t .`: 712 tests, same 3 pre-existing failures as on `main` before this work (unrelated: `tests/extraction/test_clip_export.py` — confirmed via `git stash` diff, not introduced here).
- `docker compose up -d --build --wait`: all three containers healthy; `core.DEVICE == "cuda"` and `nvidia-smi` confirmed inside the container.
- `haproxy -c` config check: valid.
- End-to-end 5-real-account load test (`scripts/loadtest/simulate_five_users.py`) against the rebuilt stack: results in §1.

## 8. Open follow-ups (not done in this session)

- Elasticsearch query itself is now the largest remaining chunk of a solo audio request (~250ms of ~530ms profiled) — not investigated further since it was already within budget once the two Python-level bugs above were fixed.
- The lossy "preview" proxy-video tier from the original plan (separate, downscaled variant for bandwidth-constrained scenarios) was not needed — faststart + the per-user HAProxy fix were sufficient, and it was explicitly deprioritized pending exactly this outcome.
