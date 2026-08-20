"""Shared runtime: CLIP text encoder + index/id_map loading."""
import gc
import json
import os
import threading
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
import open_clip
import frame_maps

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.environ.get("AIC_INDEX_DIR", ROOT / "indexes"))
MODEL = "hf-hub:timm/PE-Core-bigG-14-448"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

INDEX_NAMES = ["flat", "ivf", "hnsw"]

_model = None
_tokenizer = None
_meta = None
_config = None
_loaded = {}
_frames_by_video = {}


_model_lock = threading.Lock()


def mounted_video_path(row, data_dir=DATA):
    return Path(data_dir) / str(row["year"]) / "videos" / row["group"] / f"{row['video_id']}.mp4"


def load_model():
    global _model, _tokenizer
    if _model is None:
        with _model_lock:
            if _model is None:
                model, _, _ = open_clip.create_model_and_transforms(
                    MODEL, precision="fp16" if USE_AMP else "fp32", device=DEVICE
                )
                model.eval()
                _tokenizer = open_clip.get_tokenizer(MODEL)
                _model = model
                gc.collect()
                torch.cuda.empty_cache()
    return _model, _tokenizer


def encode_text(text):
    model, tok = load_model()
    amp = torch.autocast(DEVICE, dtype=torch.float16) if USE_AMP else _null()
    with torch.inference_mode(), amp:
        f = model.encode_text(tok([text]).to(DEVICE), normalize=True)
    return f.float().cpu().numpy().astype("float32")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def load_meta():
    global _meta
    if _meta is not None:
        return _meta
    rows = pq.read_table(INDEX_DIR / "video_meta.parquet").to_pylist()
    rows = [{**row, "path": str(mounted_video_path(row))} for row in rows]
    video_meta = {r["video_idx"]: r for r in rows}
    video_lookup = {r["video_id"]: r for r in rows}
    id_video = np.load(INDEX_DIR / "id_video.npy")
    id_frame = np.load(INDEX_DIR / "id_frame.npy")
    _meta = (id_video, id_frame, video_meta, video_lookup)
    return _meta


def load_config():
    global _config
    if _config is None:
        _config = json.loads((INDEX_DIR / "info.json").read_text())
    return _config


def get_index(name):
    if name not in _loaded:
        for k in list(_loaded):
            del _loaded[k]
        gc.collect()
        _loaded[name] = faiss.read_index(str(INDEX_DIR / f"{name}.ip"))
    return _loaded[name]


def index_info(name):
    return load_config()["indexes"][name]


def nearest_frame(video_id, time_seconds):
    id_video, id_frame, _, video_lookup = load_meta()
    row = video_lookup.get(video_id)
    if row is None:
        raise KeyError(video_id)
    video_idx = row["video_idx"]
    if video_idx not in _frames_by_video:
        _frames_by_video[video_idx] = np.sort(id_frame[id_video == video_idx])
    frames = _frames_by_video[video_idx]
    if len(frames) == 0:
        raise KeyError(f"no indexed frames for {video_id}")
    return frame_maps.load(video_id).nearest_frame(float(time_seconds), frames)


def frame_time(video_id, frame_id):
    return frame_maps.load(video_id).frame_time(frame_id)
