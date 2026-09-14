"""Shared runtime: CLIP text encoder + index/id_map loading."""
import gc
import io
import json
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
import open_clip
from PIL import Image, ImageOps, UnidentifiedImageError
from . import frame_maps

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.environ.get("AIC_INDEX_DIR", ROOT / "indexes"))
MODEL = "hf-hub:timm/PE-Core-bigG-14-448"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# Unbounded, FAISS's default OpenMP pool spans every visible core per search call.
# LatestWinsCoordinator allows up to 5 concurrent flat-index searches, so on a
# 6-core host that oversubscribes the CPU badly under full team load.
faiss.omp_set_num_threads(int(os.environ.get("AIC_FAISS_OMP_THREADS", "2")))

INDEX_NAMES = ["flat", "ivf", "hnsw"]

_model = None
_tokenizer = None
_preprocess = None
_frames_by_video = {}


_model_lock = threading.Lock()
_text_batcher = None
_text_batcher_running = True
_text_batcher_lock = threading.Lock()


def mounted_video_path(row, data_dir=DATA):
    return Path(data_dir) / str(row["year"]) / "videos" / row["group"] / f"{row['video_id']}.mp4"


def load_model():
    global _model, _tokenizer, _preprocess
    if _model is None:
        with _model_lock:
            if _model is None:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    MODEL, precision="fp16" if USE_AMP else "fp32", device=DEVICE
                )
                model.eval()
                _tokenizer = open_clip.get_tokenizer(MODEL)
                _model = model
                _preprocess = preprocess
                gc.collect()
                torch.cuda.empty_cache()
    return _model, _tokenizer


class TextEmbeddingBatcher:
    def __init__(self, encoder, window_seconds=0.01, max_requests=5):
        self._encoder = encoder
        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._condition = threading.Condition()
        self._queue = deque()
        self._closed = False
        self._scheduled = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="text-batch")

    def encode(self, texts):
        future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("text embedding batcher is closed")
            self._queue.append((tuple(texts), future))
            if not self._scheduled:
                self._scheduled = True
                self._executor.submit(self._run)
            self._condition.notify_all()
        return future.result()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=True)

    def _run(self):
        wait_for_more = True
        while True:
            with self._condition:
                if not self._queue:
                    self._scheduled = False
                    return

                if wait_for_more and not self._closed:
                    deadline = time.monotonic() + self._window_seconds
                    while len(self._queue) < self._max_requests:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or self._closed:
                            break
                        self._condition.wait(remaining)

                batch = [
                    self._queue.popleft()
                    for _ in range(min(self._max_requests, len(self._queue)))
                ]
                wait_for_more = False

            flat_texts = [text for texts, _ in batch for text in texts]
            try:
                encoded = np.asarray(self._encoder(flat_texts), dtype="float32")
                if len(encoded) != len(flat_texts):
                    raise RuntimeError("text encoder returned the wrong number of rows")
            except BaseException as exc:
                for _, future in batch:
                    future.set_exception(exc)
                continue

            offset = 0
            for texts, future in batch:
                end = offset + len(texts)
                future.set_result(encoded[offset:end])
                offset = end


def _encode_text_batch(texts):
    model, tok = load_model()
    amp = torch.autocast(DEVICE, dtype=torch.float16) if USE_AMP else _null()
    with torch.inference_mode(), amp:
        f = model.encode_text(tok(list(texts)).to(DEVICE), normalize=True)
        f = torch.nn.functional.normalize(f.float(), dim=-1)
    return f.cpu().numpy().astype("float32")


def _get_text_batcher():
    global _text_batcher
    with _text_batcher_lock:
        if not _text_batcher_running:
            raise RuntimeError("text embedding batcher is not running")
        if _text_batcher is None:
            _text_batcher = TextEmbeddingBatcher(_encode_text_batch)
        return _text_batcher


def start_text_batcher():
    global _text_batcher_running
    with _text_batcher_lock:
        _text_batcher_running = True


def close_text_batcher():
    global _text_batcher, _text_batcher_running
    with _text_batcher_lock:
        _text_batcher_running = False
        batcher = _text_batcher
        if batcher is not None:
            batcher.close()
        _text_batcher = None


def encode_text(text):
    return encode_texts([text])


def encode_texts(texts):
    return _get_text_batcher().encode(texts)


def encode_image(image):
    model, _ = load_model()
    tensor = _preprocess(image).unsqueeze(0).to(DEVICE)
    amp = torch.autocast(DEVICE, dtype=torch.float16) if USE_AMP else _null()
    with torch.inference_mode(), amp:
        features = model.encode_image(tensor, normalize=True)
        features = torch.nn.functional.normalize(features.float(), dim=-1)
    return features.cpu().numpy().astype("float32")


def _normalize(vector):
    vector = np.asarray(vector, dtype="float32").reshape(1, -1)
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("query embedding has zero norm")
    return (vector / norm).astype("float32")


def fuse_query_vectors(text_vector, image_vector, image_weight):
    if not 0 <= image_weight <= 1:
        raise ValueError("image weight must be between 0 and 1")
    image_vector = _normalize(image_vector)
    if text_vector is None:
        return image_vector
    return _normalize((1 - image_weight) * _normalize(text_vector) + image_weight * image_vector)


def decode_query_image(data, max_pixels=20_000_000):
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
                raise ValueError("query image must be a valid JPEG, PNG, or WebP")
            if image.width * image.height > max_pixels:
                raise ValueError("query image must not exceed 20 megapixels")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("query image must be a valid JPEG, PNG, or WebP") from exc


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ProfileStore:
    def __init__(self, index_dir=INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.profile_root = self.index_dir / "profiles"
        manifest_path = self.profile_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self.default = manifest["default"]
            self.cache_size = max(1, int(manifest.get("cache_size", 1)))
            self.profiles = manifest["profiles"]
        else:
            self.default = "default"
            self.cache_size = 1
            self.profiles = {"default": {"label": "Full dataset", "directory": "."}}
            self.profile_root = self.index_dir
        if self.default not in self.profiles:
            raise ValueError("default index profile is not defined")
        self._meta = {}
        self._times = {}
        self._config = {}
        self._indexes = OrderedDict()
        self._legacy_meta = None
        self._lock = threading.Lock()

    def directory(self, profile=None):
        name = profile or self.default
        if name not in self.profiles:
            raise KeyError(f"unknown profile {name}")
        directory = (self.profile_root / self.profiles[name]["directory"]).resolve()
        if directory != self.profile_root.resolve() and self.profile_root.resolve() not in directory.parents:
            raise ValueError(f"profile directory escapes profile root: {name}")
        return name, directory

    def list_profiles(self):
        return [
            {"name": name, "label": item["label"], "is_default": name == self.default,
             "count": item.get("count"), "video_count": item.get("video_count"),
             "indexes": list(self.load_config(name)["indexes"])}
            for name, item in self.profiles.items()
        ]

    def load_meta(self, profile=None):
        name, directory = self.directory(profile)
        if name not in self._meta:
            rows = pq.read_table(directory / "video_meta.parquet").to_pylist()
            rows = [{**row, "path": str(mounted_video_path(row))} for row in rows]
            self._meta[name] = (
                np.load(directory / "id_video.npy"), np.load(directory / "id_frame.npy"),
                {r["video_idx"]: r for r in rows}, {r["video_id"]: r for r in rows},
            )
        return self._meta[name]

    def list_frame_sources(self, video_id):
        """Embedding metadata sources containing keyframe ids for one video."""
        sources = []
        for name, item in self.profiles.items():
            if video_id in self.load_meta(name)[3]:
                sources.append({"name": name, "label": item["label"]})
        # With a profile manifest active, the original full index remains useful for
        # K videos that the L-only search profiles intentionally exclude.
        if self.profile_root != self.index_dir and (self.index_dir / "id_frame.npy").exists():
            self._load_legacy_meta()
            if video_id in self._legacy_meta[3]:
                sources.append({"name": "legacy", "label": "Index gốc"})
        return sources

    def _load_legacy_meta(self):
        if self._legacy_meta is None:
            rows = pq.read_table(self.index_dir / "video_meta.parquet").to_pylist()
            self._legacy_meta = (
                np.load(self.index_dir / "id_video.npy"),
                np.load(self.index_dir / "id_frame.npy"),
                {r["video_idx"]: r for r in rows},
                {r["video_id"]: r for r in rows},
            )
        return self._legacy_meta

    def frame_ids(self, source, video_id):
        if source == "legacy":
            id_video, id_frame, _, lookup = self._load_legacy_meta()
        else:
            id_video, id_frame, _, lookup = self.load_meta(source)
        row = lookup.get(video_id)
        if row is None:
            raise KeyError(f"{video_id} is unavailable in frame source {source}")
        return np.unique(id_frame[id_video == int(row["video_idx"])]).astype(np.int64)

    def load_times(self, profile=None):
        name, directory = self.directory(profile)
        if name not in self._times:
            self._times[name] = np.load(directory / "id_time.npy", mmap_mode="r")
        return self._times[name]

    def load_config(self, profile=None):
        name, directory = self.directory(profile)
        if name not in self._config:
            self._config[name] = json.loads((directory / "info.json").read_text())
        return self._config[name]

    def get_index(self, index_name, profile=None):
        name, directory = self.directory(profile)
        if index_name not in self.load_config(name)["indexes"]:
            raise KeyError(f"unknown index {index_name} for profile {name}")
        key = (name, index_name)
        with self._lock:
            if key not in self._indexes:
                self._indexes[key] = faiss.read_index(str(directory / f"{index_name}.ip"))
                while len(self._indexes) > self.cache_size:
                    self._indexes.popitem(last=False)
                    gc.collect()
            else:
                self._indexes.move_to_end(key)
            return self._indexes[key]


PROFILE_STORE = ProfileStore()


def list_profiles():
    return PROFILE_STORE.list_profiles()


def list_frame_sources(video_id):
    return PROFILE_STORE.list_frame_sources(video_id)


def frame_ids(source, video_id):
    return PROFILE_STORE.frame_ids(source, video_id)


def default_profile():
    return PROFILE_STORE.default


def available_indexes(profile=None):
    return list(PROFILE_STORE.load_config(profile)["indexes"])


def load_meta(profile=None):
    return PROFILE_STORE.load_meta(profile)


def load_times(profile=None):
    return PROFILE_STORE.load_times(profile)


def load_config(profile=None):
    return PROFILE_STORE.load_config(profile)


def get_index(name, profile=None):
    return PROFILE_STORE.get_index(name, profile)


def index_info(name, profile=None):
    try:
        return load_config(profile)["indexes"][name]
    except KeyError as exc:
        raise KeyError(f"unknown index {name} for profile {profile or default_profile()}") from exc


def indexed_frame_id(video_id, frame_idx, profile=None):
    id_video, id_frame, _, video_lookup = load_meta(profile)
    row = video_lookup.get(video_id)
    if row is None:
        raise KeyError(video_id)
    matches = np.flatnonzero(
        (id_video == int(row["video_idx"])) & (id_frame == int(frame_idx))
    )
    if len(matches) == 0:
        raise KeyError(f"no indexed frame {frame_idx} for {video_id}")
    return int(matches[0])


def indexed_frame_embedding(index_name, video_id, frame_idx, profile=None):
    global_id = indexed_frame_id(video_id, frame_idx, profile)
    return _normalize(get_index(index_name, profile).reconstruct(global_id)), global_id


def nearest_frame(video_id, time_seconds, profile=None, meta=None):
    profile = profile or default_profile()
    # meta lets a caller looping over many hits (audio search pools up to
    # SEARCH_RESULT_LIMIT) load it once instead of once per hit -- load_meta()
    # itself is a cheap dict lookup once cached, but repeating it per hit measured
    # as the single largest chunk of per-request time under concurrency, since
    # every call point contends for the GIL.
    id_video, id_frame, _, video_lookup = meta if meta is not None else load_meta(profile)
    row = video_lookup.get(video_id)
    if row is None:
        raise KeyError(video_id)
    video_idx = row["video_idx"]
    key = (profile, video_idx)
    if key not in _frames_by_video:
        _frames_by_video[key] = np.sort(id_frame[id_video == video_idx])
    frames = _frames_by_video[key]
    if len(frames) == 0:
        raise KeyError(f"no indexed frames for {video_id}")
    return frame_maps.load(video_id).nearest_frame(float(time_seconds), frames)


def frame_time(video_id, frame_id):
    return frame_maps.load(video_id).frame_time(frame_id)
