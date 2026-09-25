# Object position search

Object search and position filters use Polars over Parquet data aligned to indexed
FAISS keyframes. Clip Studio uses a separate playback Parquet partition containing
every detector sample, including frames with no objects. The original detector
files stay on the build host; the app only needs the generated data under `indexes/`.

## Build and activate the data

Run from the repository root on the host with the four detector files:

```bash
.venv/bin/python -m scripts.indexing.build_object_index \
  --source L=/mnt/c/Users/sodfi/Downloads/L_obj.parquet \
  --source M=/mnt/c/Users/sodfi/Downloads/M_obj.parquet \
  --source S=/mnt/c/Users/sodfi/Downloads/S_obj.parquet \
  --source N=/mnt/c/Users/sodfi/Downloads/N_obj.parquet
```

The builder writes a new `indexes/objects/polars-*` generation, then atomically
switches `indexes/objects/current` to it. Restart the app after a rebuild so its
in-memory Polars frames use the new generation. `AIC_OBJECT_DATA_DIR` overrides
the default `AIC_INDEX_DIR/objects/current`. The old `v1.sqlite` file is unused
by the Polars backend and can be kept for rollback.

The generated store has a manifest, one keyframe metadata file, one flat
detection file per class, and playback files partitioned by video. Keyframe
metadata records profile membership without duplicating detections for each
profile. A keyframe can be matched when its video is in a FAISS profile *and*
its group is in `--search-prefixes` (default `L,M,S,N`). Every source sample
lands in its video's playback partition either way, which is what lets a video
be looked at before it is searchable; the N groups were playback-only until
`l2026-v3`/`naflex-l2026-v3` carried their videos.

The generation records which profiles a keyframe belongs to, so it is stale
whenever a profile's keyframe set or the profile list changes — rebuild it after
building or extending a profile, not only after a new detector run.

## Query behavior

Each object clause specifies a COCO class, a normalized rectangle, and a minimum
instance count. A detection matches when its bounding-box center is inside the
rectangle and confidence meets the threshold. Requested instances must use
distinct detections on the same indexed keyframe. The builder attaches the
nearest same-video sample within 0.25 seconds of the frame-map timestamp and
converts source `(y0,x0,y1,x1)` boxes to API `(x0,y0,x1,y1)` boxes. It also
normalizes the S source's hyphenated video IDs.

The object store loads its keyframe and class data into Polars on the compute
host during startup. `POST /search` accepts standalone object queries and
position filters on other search modes; `POST /search/image` accepts the same
filter. `POST /search/temporal/image` accepts object events and an optional
object-position filter on each non-object event. Results retain the matching
class, confidence, box and source time offset. Existing search snapshots handle
pagination. An absent or unreadable object store makes object operations return
503 without blocking unrelated search.

`GET /objects/{video_id}/detections` reads only that video's playback Parquet
partition through Polars. The browser chooses the nearest sample within 1.5
seconds, displays its boxes, and hides stale boxes across longer gaps. Empty
detection samples clear the overlay. In split-host mode the teammate app calls
the owner's `/core/object/*` routes, so it does not need local object data.
