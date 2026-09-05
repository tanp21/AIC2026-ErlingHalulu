"""Detect shot boundaries with TransNetV2 and write them next to the videos.

    uv run python -m scripts.extraction.detect_shots --year 2026
    uv run python -m scripts.extraction.detect_shots --year 2026 --groups L21,L22

Output is `data/<year>/shots/<group>/<video_id>.json`, the format the 2025 pipeline
already produced and the format `shots.py` reads:

    {"fps": 30.0, "shots": [[0, 48], [49, 459], ...], "shots_sec": [[0.0, 1.6], ...]}

Frame index means decode order, matching how `indexes/frame_maps` maps frames to time,
so the server can turn a shot into an exact PTS window without fps arithmetic.

Needs TensorFlow and the `transnetv2` package, which are dev-only dependencies: this
never runs inside the production image.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CUDNN_WORKSPACE_LIMIT_IN_MB", "256")  # ~1.3GB VRAM/worker, not ~4.4GB

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))
SHOT_THRESHOLD = 0.5
MIN_SHOT_LEN = 15

_model = None


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def video_fps(video_path):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps > 0 else 30.0


def detect_shots(model, video_path, threshold=SHOT_THRESHOLD, min_shot_len=MIN_SHOT_LEN):
    """Cut at the peak of each run of frames above the transition threshold."""
    _, predictions, _ = model.predict_video(str(video_path))
    predictions = predictions.squeeze()
    frame_count = len(predictions)

    above = predictions > threshold
    cuts = [0]
    position = 0
    while position < frame_count:
        if not above[position]:
            position += 1
            continue
        end = position + 1
        while end < frame_count and above[end]:
            end += 1
        peak = position + int(np.argmax(predictions[position:end]))
        if peak - cuts[-1] >= min_shot_len:
            cuts.append(peak)
        position = end

    shots = [(cuts[i], cuts[i + 1] - 1) for i in range(len(cuts) - 1)]
    if cuts[-1] < frame_count - 1:
        shots.append((cuts[-1], frame_count - 1))
    return [(start, end) for start, end in shots if end - start >= min_shot_len]


def init_worker():
    """Load TransNetV2 once per worker process."""
    global _model
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    from transnetv2 import TransNetV2

    _model = TransNetV2()


def process_video(job):
    video_path, output_path = Path(job[0]), Path(job[1])
    started = time.perf_counter()
    shots = detect_shots(_model, video_path)
    fps = video_fps(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "fps": fps,
        "shots": [[int(start), int(end)] for start, end in shots],
        "shots_sec": [[start / fps, end / fps] for start, end in shots],
    }))
    return {"video": video_path.name, "shots": len(shots),
            "seconds": time.perf_counter() - started}


def collect_jobs(videos_dir, shots_dir, groups, force):
    jobs = []
    for video in sorted(videos_dir.rglob("*.mp4")):
        group = video.parent.name
        if groups and group not in groups:
            continue
        output = shots_dir / group / f"{video.stem}.json"
        if output.exists() and not force:
            continue
        jobs.append((str(video), str(output)))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2026", help="dataset year under the data directory")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. L21,L22")
    parser.add_argument("--workers", type=int, default=2,
                        help="TransNetV2 processes; each holds ~1.3GB VRAM")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--force", action="store_true", help="redo videos that already have JSON")
    args = parser.parse_args(argv)

    videos_dir = args.data_dir / args.year / "videos"
    shots_dir = args.data_dir / args.year / "shots"
    if not videos_dir.is_dir():
        raise SystemExit(f"no videos under {videos_dir}")
    groups = {name.strip() for name in args.groups.split(",") if name.strip()}

    jobs = collect_jobs(videos_dir, shots_dir, groups, args.force)
    if not jobs:
        log(f"nothing to do: every video under {videos_dir} already has shots")
        return 0
    log(f"{len(jobs)} videos -> {shots_dir} ({args.workers} workers)")

    started = time.perf_counter()
    done = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = {pool.submit(process_video, job): job for job in jobs}
        for future in as_completed(futures):
            name = Path(futures[future][0]).name
            try:
                result = future.result()
            except Exception as error:  # one bad file must not sink the batch
                failures.append((name, error))
                log(f"FAIL {name}: {error}")
                continue
            done += 1
            log(f"{done}/{len(jobs)} {name}: {result['shots']} shots "
                f"in {result['seconds']:.1f}s")

    elapsed = time.perf_counter() - started
    log(f"done: {done} videos in {elapsed / 60:.1f} min, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
