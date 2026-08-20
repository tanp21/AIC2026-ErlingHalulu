"""Exact PTS-based thumbnail extraction via ffmpeg + disk cache."""
import hashlib
import os
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("AIC_THUMB_CACHE_DIR", ROOT / ".thumb_cache"))
MAX_CACHE_BYTES = int(os.environ.get("AIC_THUMB_CACHE_MAX_BYTES", 5 * 1024**3))
_ffmpeg_slots = threading.BoundedSemaphore(4)
_key_locks = tuple(threading.Lock() for _ in range(256))
_cache_guard = threading.Lock()
_known_cache_bytes = None


def _key_lock(key):
    return _key_locks[int(key[:8], 16) % len(_key_locks)]


def _prune_cache(current):
    global _known_cache_bytes
    with _cache_guard:
        files = list(CACHE_DIR.glob("*.jpg"))
        if _known_cache_bytes is None:
            _known_cache_bytes = sum(path.stat().st_size for path in files if path.exists())
        else:
            _known_cache_bytes += current.stat().st_size
        if _known_cache_bytes <= MAX_CACHE_BYTES:
            return
        for path in sorted(files, key=lambda item: item.stat().st_mtime):
            if path == current or not path.exists():
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            _known_cache_bytes -= size
            if _known_cache_bytes <= MAX_CACHE_BYTES:
                break


def ffmpeg_command(video_path, frame_idx, mapping, output, w=320):
    pts = int(mapping.pts[frame_idx])
    timestamp = mapping.frame_time(frame_idx)
    filters = [f"select=eq(pts\\,{pts})"]
    if w > 0:
        filters.append(f"scale={w}:-2")
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-copyts", "-ss", f"{timestamp:.15f}",
        "-i", str(video_path), "-an", "-vf", ",".join(filters),
        "-frames:v", "1", "-vsync", "0", "-q:v", "4", str(output),
    ]


def extract(video_path, frame_idx, mapping, w=320):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pts = int(mapping.pts[frame_idx])
    key = hashlib.sha1(
        f"pts-v2:{video_path}:{pts}:{mapping.time_base_num}/{mapping.time_base_den}:{w}".encode()
    ).hexdigest()
    out = CACHE_DIR / f"{key}.jpg"
    with _key_lock(key):
        created = False
        if not out.exists():
            temporary = CACHE_DIR / f"{key}.tmp.jpg"
            try:
                with _ffmpeg_slots:
                    subprocess.run(
                        ffmpeg_command(video_path, frame_idx, mapping, temporary, w),
                        check=True,
                        timeout=30,
                    )
                temporary.replace(out)
                created = True
            finally:
                temporary.unlink(missing_ok=True)
        data = out.read_bytes()
        if created:
            _prune_cache(out)
        return data
