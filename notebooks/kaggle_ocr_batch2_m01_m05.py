#!/usr/bin/env python3
"""OCR every keyframe of Batch 2 2026's M01-M05 ("tin tuc" news videos) on Kaggle.

Self-contained on purpose, matching ``notebooks/kaggle_pe_core_n2026_2s.py``: paste this file
into a Kaggle script notebook, enable internet, and attach two datasets:

  1. The raw videos (``yunsoo21/74512abf7b0d2c9bbf44-part1``), layout
     ``Videos_M0N/videos/M0N_V0NN.mp4``.
  2. The batch-2 frame manifest, ``batch2_2026.json`` (``tommr211/aic-keyframes``) --
     already covers all of batch 2 (M01-M10, S01, 316 videos); the same 3-per-shot frame ids
     this job's own ``scripts.ocr.build_ocr_kaggle_manifest`` independently produces for
     M01-M05 (verified byte-for-byte equal on those groups before trusting this source).
     ``filter_manifest_groups`` narrows it to ``GROUPS`` at runtime, since the attached video
     dataset only has M01-M05. Searched for under ``/kaggle/input`` by ``MANIFEST_NAME``.

Everything else this job needs travels inside this one file, or is fetched at runtime, rather
than as a Kaggle Dataset input:

  * ``scripts/ocr/engines.py`` plus ``vietocr_configs/base.yml`` and ``vietocr_configs/
    vgg-seq2seq.yml`` (``postprocess.py`` and ``vgg-transformer.yml`` are not needed -- the
    ``run()`` path this job calls never touches either) are embedded below as gzip+base64
    blobs, built mechanically by a script from the real files (never hand-transcribed -- a
    past incident hand-copying a script into ``save_notebook`` silently introduced a syntax
    bug) and decoded to disk once by ``stage_engine_code``.
  * VietOCR's ``vgg_seq2seq.pth`` checkpoint is downloaded at runtime from VietOCR's own
    GitHub release (valid cert, not the expired-cert ``vocr.vn`` -- see ``docs/reports/
    ocr-shot-audit.md``) and verified against a hardcoded sha256 by ``download_checkpoint``.

(Embedding the engine code and downloading the checkpoint at runtime, rather than shipping
both as Kaggle Dataset inputs like the manifest, is a deliberate adaptation: this account's
Kaggle MCP tools can add a file to an *existing* dataset (``upload_dataset_file`` +
``update_dataset_metadata``) but expose no way to *create* one, and the manifest -- at ~450KB
compressed -- is too large to push reliably through this session's own tooling as an inline
string anyway, so it needed an existing dataset to live in regardless; ``batch2_2026.json``
already being available on ``tommr211/aic-keyframes`` meant one never had to be created.)

Output: one append-only JSONL shard per (part, rank) under ``/kaggle/working/ocr/parts/``,
``{"key": "<group>/<video_id>/<frame_idx>", "lines": [{"text", "box", "conf"}, ...]}`` per
line -- replacing the old ``run_corpus.py`` pattern of rewriting one giant JSON dict on every
flush (quadratic over ~135,000 keyframes) with O(1) appends, resumable by construction. At the
end of a run, ``merge_and_verify`` reassembles every shard into the flat
``{"<key>": {"lines": [...]}}`` dict schema ``scripts/indexing/build_ocr_index.py`` already
reads -- nothing downstream changes.

Two OCR-specific gotchas this template's PE-Core sibling does not have to deal with:

* ``rapidocr_onnxruntime``'s CUDA execution provider hardcodes ``device_id: 0`` -- it ignores
  whatever device index you hand VietOCR/torch. Each spawned rank therefore sets
  ``CUDA_VISIBLE_DEVICES`` to its own rank as the *first* line of ``run_rank``, before any
  torch/cv2/rapidocr/vietocr import in that process, and uses ``cuda:0`` for everything inside
  it (never ``cuda:{rank}``) since that process now sees only one GPU. Do not "simplify" this
  back to per-rank device indices -- it silently piles every rank's detector onto physical GPU
  0.
* RapidOCR falls back to CPU on a CUDA provider failure with a log warning, not an exception.
  ``run_rank`` asserts ``pipeline.detector.text_det.infer.use_cuda`` after first use, so a
  broken GPU setup fails the canary loudly instead of quietly running 100x slower.
* Getting that CUDA provider to exist at all takes deliberate pip ordering, version choice and
  loader-path work -- ``rapidocr_onnxruntime`` declares ``onnxruntime`` (the CPU wheel) as a
  dependency, so a naive install lets it overwrite ``onnxruntime-gpu``'s libraries; cuDNN 9 is
  not on the loader path even though torch bundles it; and since 1.27.0 the PyPI
  ``onnxruntime-gpu`` wheels are built for CUDA 13 while Kaggle's image is CUDA 12.8. See the
  "CUDA runtime for onnxruntime-gpu" block above ``ensure_ocr_deps`` before changing that
  order, the version pin, or the preload.
* **Check the outcome, never the intent.** ``ort.get_available_providers()`` names every
  provider the build was *compiled with* -- it still lists ``CUDAExecutionProvider`` on a
  mismatched build, right up until a session is created, the provider .so fails to dlopen and
  ORT silently runs on CPU. RapidOCR's ``infer.use_cuda`` is derived from that same list, so it
  is True in exactly the case it is supposed to catch. The load probe in ``ORT_PROVIDER_CHECK``
  and the session check in ``run_rank`` both exist for this; an earlier version of this
  notebook trusted the flag and spent a canary run detecting text on CPU at 0.5 fps while every
  check reported success.

A full part runs on its own: ``PART_COUNT=1``. If a run does not fit one Kaggle session's
runtime/quota, set ``PART_COUNT`` and give each session a distinct ``PART_INDEX`` -- the
partition is a deterministic greedy bin-packing over whole videos (identical to the PE-Core
template), so every session computes the same disjoint split with no coordination. Merge every
part's shards afterward with ``scripts/ocr/merge_ocr_kaggle_shards.py``.
"""
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# CONFIG -- teammates forking the notebook change only this block.
# --------------------------------------------------------------------------------------
GROUPS = ("M01", "M02", "M03", "M04", "M05")
MANIFEST_NAME = "batch2_2026.json"  # searched for under /kaggle/input; covers all of batch2
                                    # (M01-M10, S01) -- filter_manifest_groups narrows to GROUPS

CKPT_URL = "https://github.com/pbcquoc/vietocr/releases/download/v0.3.2/vgg-seq2seq.pth"
CKPT_SHA256 = "0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db"

DET_USE_CUDA = True
SMALL_RESCAN = False  # matches run_corpus_remote.py's already-established full-corpus default
DET_MODEL = "v4"      # unchanged from the shipped default, for consistency with batch 1
REC_BATCH_SIZE = 32   # halved once, automatically, on an OOM at startup

STAGE_REPORT_SECONDS = 30  # how often a rank prints its per-stage ms/frame breakdown

PART_COUNT = 1   # every worker/session must use the same value
PART_INDEX = 0   # this session's slice, 0 .. PART_COUNT-1
CANARY = False   # True: OCR only the first 1-2 videos, as a smoke test. Off for this run
                 # because the failure a canary exists to catch -- a detector that silently
                 # lands on CPU -- is now rejected by the pre-flight provider check *before*
                 # any OCR starts, and that check is verified in both directions. Throughput
                 # is visible in the log within the first few minutes regardless.
ONLY_VIDEOS = ()  # (): this part's whole selection. A tuple of video ids narrows to just those
                  # -- how a follow-up run redoes videos an earlier run had to skip.

OCR_REQUIREMENTS = (
    "rapidocr_onnxruntime==1.4.4",
    "vietocr==0.3.13",
    "opencv-python-headless==5.0.0.93",
    "pyyaml",
)
# onnxruntime-gpu is deliberately absent from this list and installed on its own, last, by
# ensure_ocr_deps: rapidocr_onnxruntime declares `onnxruntime>=1.7.0`, so listing both here
# would put the plain CPU wheel and the GPU build in one pip transaction, where neither is
# reliably the last writer to site-packages/onnxruntime/.
#
# It is pinned, and the pin is load-bearing. From 1.27.0 the wheels published as
# `onnxruntime-gpu` are built against CUDA 13 and link libcudart.so.13; Kaggle's image is
# CUDA 12.8 (torch bundles libnvrtc.so.12.8). Installing the newest build there produces an
# ORT whose CUDA provider .so cannot dlopen, so ORT silently runs the detector on CPU -- one
# whole canary run was spent on that. 1.26.0 is the last CUDA-12 line.
ORT_GPU_VERSION = "1.26.0"
ORT_GPU_FALLBACKS = ("1.22.0",)  # still CUDA 12, if the newest CUDA-12 build ever stops loading
# --------------------------------------------------------------------------------------

WORK = Path("/kaggle/working")
CKPT_TARGET = Path("/tmp/vgg_seq2seq.pth")  # VietOCR's own cache-filename derivation (see
                                             # stage_checkpoint's docstring) requires this exact
                                             # underscored name, not the hyphenated release asset.
FRAME_DIR = Path("/tmp/ocr_frames")  # Extracted PNGs live on the local disk, not under
                                     # /kaggle/working -- that path is the notebook's *output*
                                     # volume, and thousands of 1080p PNGs written and read
                                     # back through it cost far more than the same traffic
                                     # against local disk. Only the final merged JSON is
                                     # written to WORK.

# -- embedded engine code (gzip+base64) -------------------------------------------------------
# Built mechanically from the real files by a one-shot local script, never hand-typed:
# scripts/ocr/{engines.py,vietocr_configs/*.yml}, gzip -9'd then base64-encoded, decoded to
# disk once by stage_engine_code. The frame manifest is NOT embedded here -- at ~450KB
# compressed it is too large to push through this MCP session's own tooling reliably -- so it
# ships as a small Kaggle Dataset instead (see load_manifest / MANIFEST_NAME above).
ENGINE_PY_GZ_B64 = (
    "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"
)
VIETOCR_BASE_YML_GZ_B64 = (
    "H4sIAG2ytGoC/3VVZ3sbRRD+fr9iwYApxpblQjh6C72HXo7V3VpafHcr7+1JdihPCiUJARIgdIhLip3EaY5Tdgh8WMf/Y/4Ju3eS7OQBfdCjmXmn7jujPhI2aFpnRAkS80w5UWZETJAZkUsSUUUzpoiQJM8YidgEzWNFWpyplCbMqgq81xIhrfmknz5iZs0O1Auo582C2W3mzE7Uc6hn13av7UJ9DvVZ1Cuoz6NeRX0B9RnUp1FfQn3RzBuLOI76GOol1IuoT6I+gfoo6iOol1Gfqj0aPhY9vnZw7QB7wiyaPagBtUZ9FfVfZsnsRX0F9WVzwuxD2ImwA2E3wi6EzxE+Q/0P6r8RvkT4YmJr/cnGU/xpc8rsR9iLsGdtaW3RLJuvEb5C2PfhM5PPxs8lz6cviBfNefMdwrcI35hVc8ismO8RLGy/uWB+QLCCtR5CsL9/QvgR4SDCAYRfEH5en1ufRfgD4XeEwwh/IswhWM1vCL8iLCDMN1+aelm+kr2qtuWvmSvGOh1HOHZt6dqi0cZijiIcWT+7fgbhJMIJhGWEUwhWPI2whLCIcA7hbOv19hvTb868hbCCcB7hEsJFhCsIl81VY9OvIlzY/nZluDoyOjZ+z5Z7b7q575Zbb+vvv/2OO+8auHtwyL/v/gcefOjhd9597/3gg48+/uRT0u95fSRs5gMkzCNafvsVL2ItHjK/K3oZmwoSEbGYp3WfKEnTbELIhElv02+fEI/YT1RCfVIdGy8UaYPRyCdbSiFPApaGFiGDmM4wmflkvGeJ2H9ZIp4EE4xFNk+bShuqWhktoyV0OnDFxSytq4ZPhivV0cLQFFkQSdEUufJJZXC4UBbFXq/2RFPxhG+35fcCxtLZKpXKSNlQM1RBpqjsutgwPO061KgKG0FmI/hkpFriJU9VwFpMzrhSK4WyRWMedZWjlY6Wq6LN4Uqlq+kj7QaTxZZmtMWI281insR2b0OziIeKi7QAs+mmcGUNDrUZrzdUNrTpPUQoB5uqUQDDBgsnm8LW5cAb0g34YMPSc41F3fkUTQ9aoVNlmsexKzLiGa3F9jqIpJlTxQgNw1zScGbAnZKNk2Nft8akOzcZTZpxoWNp4VqMhlPXlW2eO5tLZrlWPglTkod2Si6l53UOle9M3VrsgbrxkJWUsga/0HSQTaoaLjNNU6HKjDSNCE9onZVUs9hAClHMiSf1oS5xeBpsOPmbAgTlZNS02vTO/wO1xiBLaBz34H1EMkeesgRX2kiVNIrHHCAxlXU7s1IkbW4nztNQMttfb8wljZxzUOJ6NCyViS28zSO3HNfrLdE7+rHhajnWWNCoy2u3jW0hJwt+jpS8tqESlgjH4G0yZ55H87q/KaYT7axy1lmlbJJFdpXTel5kLK9CYfc6RBadbH09HlnSdGhRYzTJGJVho8OKDfIPFP9RmwFuNjVGsli07VUqFrNn9clWGmc26VTOHXEKiXj/AlqtUJQRBwAA"
)
VIETOCR_SEQ2SEQ_YML_GZ_B64 = (
    "H4sIAG2ytGoC/7VR20rEMBB9z1cM7IuCbrcFBfsrIiFNZtuwzcXJtIpf77RbYVFRfDAQyJwzc85kZgfHRMBkfFSZcH20MDDn0lbVnCzt51g5w6aaPbLE1dz3uuBzI3efeVBqBxONIDJjsmaEbHiAq0VWBJ23fK1e0PcDl78Kd8aeuhSxBaHqB91FZWNsFcjZQU5p9LGHwuQdQvFvuDKlnDOWcwuPzQ00T78B9c9AvQAfMXzyPyFFHFf/M3X6pwY2X+cDpCOkKU8MRzQ8EUIweaUH7xzKBpu7e6VkkDokh0ubLWyjVbLjWGQ9AencJ0YrSaQvaxfc4fe4D722g4ny66/JGDqUdHfBUMpp4hYO+1qplNkHGdXmHcyrHmnhDod6RbJlXdjQVvAOppF4j6ECAAA="
)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def stage_summary(stages, frames):
    """Per-frame milliseconds for each stage, for a rank's progress line."""
    frames = max(frames, 1)
    return "ms/frame " + " ".join(f"{name}={seconds * 1000 / frames:.0f}"
                                  for name, seconds in stages.items())


# -- dependency + input discovery -------------------------------------------------------

# -- CUDA runtime for onnxruntime-gpu ---------------------------------------------------------
# onnxruntime-gpu's CUDA provider lives in its own .so and dlopen()s the CUDA/cuDNN runtime
# when a session is created. If that dlopen fails, ORT does not raise -- it silently drops
# CUDAExecutionProvider from get_available_providers(), RapidOCR logs one warning and runs the
# detector on CPU, and the run is ~100x slower with nothing in the output to say so. Two
# independent things cause exactly that on Kaggle's image, and this job has to survive both:
#
#   1. rapidocr_onnxruntime declares ``onnxruntime>=1.7.0``, so installing it *after*
#      onnxruntime-gpu makes pip pull the plain CPU wheel and overwrite the GPU build's shared
#      libraries under site-packages/onnxruntime/ -- both distributions own that tree and pip
#      does not treat either as replacing the other. The GPU build therefore has to be
#      installed last and alone; see ensure_ocr_deps.
#   2. cuDNN 9 / CUDA 12 are not on the dynamic loader's default search path, even though the
#      image's preinstalled torch bundles them under torch/lib. A library already loaded into
#      the process (RTLD_GLOBAL) satisfies a later dlopen by bare soname, and LD_LIBRARY_PATH
#      covers processes started afterwards -- preload_cuda_libs and cuda_library_dirs do those
#      two jobs respectively.
#
# A run that cannot get CUDAExecutionProvider stops with a full diagnosis instead of
# continuing: a silent CPU detector is the "extremely slow, nothing explains why" outcome
# this whole notebook exists to avoid.

ORT_PROVIDER_CHECK = r'''
import ctypes, os, site, sys
from pathlib import Path

dirs = []
try:
    import torch
    dirs.append(Path(torch.__file__).parent / "lib")
except Exception as exc:
    print("torch unavailable: %s" % exc, file=sys.stderr)
for site_dir in site.getsitepackages():
    dirs += sorted(Path(site_dir).glob("nvidia/*/lib"))

def load_all(paths):
    loaded, failed = [], []
    for path in paths:
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            loaded.append(path.name)
        except OSError as exc:
            failed.append((path, exc))
    return loaded, failed

paths = [p for directory in dirs for pattern in ("libcu*.so*", "libnv*.so*")
         for p in sorted(directory.glob(pattern))]
preloaded, failed = load_all(paths)
# A cuDNN split library can need a versioned symbol from a sibling whose name sorts later
# (libcudnn_cnn before libcudnn_graph, say). A second pass, once everything else in the set
# is resident, resolves those -- without it, a lib that would have loaded fine is reported
# as a failure and left out of the process.
if failed:
    again, failed = load_all([p for p, _ in failed])
    preloaded += again
failed = ["%s: %s" % (p.name, exc) for p, exc in failed]

print("preloaded " + ",".join(preloaded))
for line in failed:
    print("preload_failed " + line, file=sys.stderr)
import onnxruntime as ort
print("ort_version " + ort.__version__)
print("ort_providers " + ",".join(ort.get_available_providers()))
print("ort_device " + ort.get_device())

# get_available_providers() lists what this ORT build was COMPILED with, not what loaded --
# the CUDA provider lives in its own .so, dlopen'd when a session is created. That dlopen is
# the step that fails on a CUDA-version mismatch, and it fails silently: ORT drops to CPU and
# get_available_providers() still names CUDA. So load the library here, where the failure is
# visible and carries ORT's own reason (the missing soname).
provider_lib = os.path.join(os.path.dirname(ort.__file__), "capi",
                            "libonnxruntime_providers_cuda.so")
try:
    ctypes.CDLL(provider_lib, mode=ctypes.RTLD_GLOBAL)
    print("ort_cuda_provider loaded")
except OSError as exc:
    print("ort_cuda_provider FAILED " + str(exc), file=sys.stderr)
'''


def cuda_library_dirs():
    """Directories holding CUDA/cuDNN runtime libs onnxruntime-gpu may need to dlopen.

    Resolved with importlib.util.find_spec rather than ``import torch``: this runs in the
    parent process, which must reach ``mp.spawn`` without having initialised a CUDA context
    of its own.
    """
    dirs = []
    spec = importlib.util.find_spec("torch")
    if spec is not None and spec.origin:
        torch_lib = Path(spec.origin).parent / "lib"
        if torch_lib.is_dir():
            dirs.append(str(torch_lib))
    for entry in sys.path:
        for path in sorted(Path(entry).glob("nvidia/*/lib")):
            if path.is_dir():
                dirs.append(str(path))
    return dirs


def preload_cuda_libs():
    """RTLD_GLOBAL-load every CUDA/cuDNN lib cuda_library_dirs() can find into this process.

    Called at the top of run_rank, before onnxruntime is imported there, so the dlopen ORT's
    CUDA provider performs at session-creation time resolves by soname against what is already
    loaded rather than against a search path that does not carry these libraries.
    """
    import ctypes

    def load_all(paths):
        loaded, gone = [], []
        for path in paths:
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(path.name)
            except OSError as exc:
                gone.append((path, exc))
        return loaded, gone

    paths = [p for directory in cuda_library_dirs()
             for pattern in ("libcu*.so*", "libnv*.so*")
             for p in sorted(Path(directory).glob(pattern))]
    loaded, failed = load_all(paths)
    # See the same two-pass note in ORT_PROVIDER_CHECK above: a cuDNN split library can need a
    # versioned symbol from a sibling that sorts later.
    if failed:
        again, failed = load_all([p for p, _ in failed])
        loaded += again
    return loaded, [f"{path.name}: {exc}" for path, exc in failed]


def check_cuda_provider():
    """Run ORT_PROVIDER_CHECK in a fresh interpreter.

    Returns ``(providers, cuda_loads, stdout, stderr)``. A subprocess rather than an
    in-process import, for two reasons: LD_LIBRARY_PATH only affects a process from exec time
    onwards, and ORT prints the actual dlopen failure (the missing soname) to *the child's*
    stderr -- the one place the real reason ever appears.
    """
    env = dict(os.environ)
    dirs = cuda_library_dirs()
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            dirs + [p for p in [env.get("LD_LIBRARY_PATH", "")] if p])
    proc = subprocess.run([sys.executable, "-c", ORT_PROVIDER_CHECK],
                          capture_output=True, text=True, env=env)
    providers = []
    for line in proc.stdout.splitlines():
        if line.startswith("ort_providers "):
            providers = [p for p in line.split(" ", 1)[1].split(",") if p]
    cuda_loads = "ort_cuda_provider loaded" in proc.stdout
    return providers, cuda_loads, proc.stdout.strip(), proc.stderr.strip()


def cuda_provider_ready(providers, cuda_loads):
    """Whether ORT can actually *use* CUDA -- compiled in AND its provider library loaded.

    Both halves matter, and checking only the first is the mistake this function exists to
    stop: ``get_available_providers()`` still names CUDAExecutionProvider on a mismatched
    build, right up until a session is created and quietly runs on CPU instead.
    """
    return "CUDAExecutionProvider" in providers and cuda_loads


def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)


def describe_cuda_environment():
    """This image's CUDA stack, in full, for a failure report."""
    lines = []
    for cmd in (["nvidia-smi"], ["bash", "-lc", "nvcc --version"]):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            lines.append(f"$ {' '.join(cmd)}\n{(proc.stdout or proc.stderr).strip()}")
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never mask the failure
            lines.append(f"$ {' '.join(cmd)} -> {exc}")
    for pattern in ("usr/local/cuda*", "usr/lib/x86_64-linux-gnu/libcudnn*",
                    "usr/local/cuda*/lib64/libcudnn*", "usr/lib/x86_64-linux-gnu/libcublas*"):
        found = sorted(str(p) for p in Path("/").glob(pattern))
        lines.append(f"/{pattern} -> {found}")
    lines.append(f"cuda_library_dirs() -> {cuda_library_dirs()}")
    return "\n".join(lines)


def ensure_ocr_deps():
    """Install the OCR stack in the one order, and at the one version, that keeps CUDA working.

    The order is the whole point (see the CUDA block above): requirements first -- that is
    what drags the plain CPU onnxruntime in as rapidocr_onnxruntime's declared dependency --
    then remove *both* flavours, then install onnxruntime-gpu last so it is the only writer to
    site-packages/onnxruntime. Earlier attempts installed onnxruntime-gpu first and let
    rapidocr replace it, which is why CUDAExecutionProvider kept vanishing with no error.

    The version matters just as much, and the check is on *loading*, not on what the build
    advertises -- see cuda_provider_ready.
    """
    try:
        import rapidocr_onnxruntime  # noqa: F401
        import vietocr  # noqa: F401
    except ImportError:
        deps_ready = False
    else:
        deps_ready = True

    if deps_ready and cuda_provider_ready(*check_cuda_provider()[:2]):
        log("OCR dependencies present with a loadable CUDAExecutionProvider")
        return

    log("installing OCR dependencies (requirements, then onnxruntime-gpu alone and last)")
    _pip("install", "-q", *OCR_REQUIREMENTS)
    _pip("uninstall", "-y", "-q", "onnxruntime", "onnxruntime-gpu")

    last = None
    for version in (ORT_GPU_VERSION, *ORT_GPU_FALLBACKS):
        log(f"installing onnxruntime-gpu=={version}")
        _pip("install", "-q", f"onnxruntime-gpu=={version}")
        providers, cuda_loads, out, err = check_cuda_provider()
        log(f"provider check (onnxruntime-gpu=={version}):\n{out}")
        if err:
            log(f"provider check stderr:\n{err}")
        if cuda_provider_ready(providers, cuda_loads):
            return
        last = (version, providers, out, err)
        log(f"onnxruntime-gpu=={version}: no usable CUDA provider, trying the next build")
        _pip("uninstall", "-y", "-q", "onnxruntime-gpu")

    # cuDNN 9 may genuinely be missing from the image -- pip's onnxruntime-gpu does not vendor
    # it. The nvidia-* wheels put a matching cuDNN/cuBLAS under site-packages/nvidia/*/lib,
    # which cuda_library_dirs() and the provider check both already search.
    version, providers, out, err = last
    log("every candidate build lacked a usable CUDA provider; installing cuDNN/cuBLAS wheels")
    _pip("install", "-q", f"onnxruntime-gpu=={version}", "nvidia-cudnn-cu12", "nvidia-cublas-cu12")
    providers, cuda_loads, out, err = check_cuda_provider()
    log(f"provider check after cuDNN wheels:\n{out}")
    if err:
        log(f"provider check stderr:\n{err}")
    if cuda_provider_ready(providers, cuda_loads):
        return

    raise SystemExit(
        "No onnxruntime-gpu build gave a CUDA provider that actually loads, so RapidOCR's "
        "detector would run on CPU -- roughly 100x slower, with nothing in the output to say "
        "so. Stopping here rather than reporting frames OCR'd at that speed.\n\n"
        f"--- providers seen ---\n{providers}\n\n"
        f"--- provider check stdout ---\n{out}\n\n"
        f"--- provider check stderr ---\n{err}\n\n"
        f"--- image CUDA environment ---\n{describe_cuda_environment()}"
    )


def find_under_input(filename, override_env):
    override = os.environ.get(override_env)
    if override:
        return Path(override)
    root = Path(os.environ.get("AIC_KAGGLE_INPUT", "/kaggle/input"))
    matches = sorted(root.rglob(filename))
    if not matches:
        raise SystemExit(f"{filename} not found under {root} -- is the dataset attached? "
                         f"(or set {override_env})")
    return matches[0]


def find_manifest_path():
    return find_under_input(MANIFEST_NAME, "AIC_MANIFEST_PATH")


def load_manifest():
    path = find_manifest_path()
    manifest = json.loads(path.read_text())
    log(f"manifest: {path} -- {manifest['video_count']} videos, "
        f"{manifest['keyframe_count']:,} keyframes")
    return manifest


def video_group(video_id):
    """The repo-wide convention (see CLAUDE.md): a video's group is the id's prefix before
    the first underscore. Neither this job's own manifest nor ``batch2_2026.json``
    (``tommr211/aic-keyframes``) carries a "group" field on each video entry -- it has to be
    derived from the id every time it's needed, not read off the manifest.
    """
    return video_id.split("_", 1)[0]


def filter_manifest_groups(manifest, groups):
    """Keep only videos whose group prefix is in ``groups``.

    The attached manifest may cover more than this job's scope -- ``batch2_2026.json``
    (``tommr211/aic-keyframes``) lists all of batch 2 (M01-M10, S01, 316 videos), but the
    attached raw-video dataset only has M01-M05 -- so every other video would otherwise be
    recorded as "not found" noise, or worse, silently attempted against whatever partial
    match a suffix search turns up.
    """
    groups = set(groups)
    videos = {vid: entry for vid, entry in manifest["videos"].items()
              if video_group(vid) in groups}
    return {
        **manifest,
        "video_count": len(videos),
        "keyframe_count": sum(len(v["frames"]) for v in videos.values()),
        "videos": videos,
    }


def stage_engine_code(base_dir):
    """Decode the embedded engines.py + vietocr_configs bundle to disk once.

    Embedded rather than shipped as a Kaggle dataset input: this account's Kaggle tools can
    add a file to an existing dataset but cannot create a new one, and engines.py compresses
    to ~15KB, small enough that embedding it here avoids that dependency entirely while still
    shipping the byte-identical file (built by a script, never hand-transcribed).
    """
    import gzip

    base_dir = Path(base_dir)
    (base_dir / "vietocr_configs").mkdir(parents=True, exist_ok=True)
    payloads = {
        "engines.py": ENGINE_PY_GZ_B64,
        "vietocr_configs/base.yml": VIETOCR_BASE_YML_GZ_B64,
        "vietocr_configs/vgg-seq2seq.yml": VIETOCR_SEQ2SEQ_YML_GZ_B64,
    }
    for rel, blob in payloads.items():
        target = base_dir / rel
        content = gzip.decompress(base64.b64decode(blob))
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
    return base_dir


def download_checkpoint(base_dir, url=CKPT_URL, expected_sha256=CKPT_SHA256):
    """Fetch VietOCR's vgg_seq2seq.pth from its GitHub release, verified against a hardcoded
    sha256 -- a valid cert, unlike vocr.vn's expired one (see docs/reports/ocr-shot-audit.md).
    Downloaded once (by main(), before ranks spawn) into a location every rank can copy from;
    see stage_checkpoint for why each rank still copies it into /tmp under its own name.
    """
    import urllib.request

    target = Path(base_dir) / "vgg_seq2seq.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        return target
    log(f"downloading VietOCR checkpoint from {url}")
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    urllib.request.urlretrieve(url, tmp)
    actual = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if actual != expected_sha256:
        tmp.unlink()
        raise RuntimeError(f"{url}: sha256 {actual} != expected {expected_sha256}")
    tmp.replace(target)
    return target


def resolve_video_path(root, video_id, rel):
    """The manifest's ``rel`` first, then a suffix-match fallback if the mount nests deeper
    or shallower than expected -- Kaggle's exact top-level mount folder name is metadata-
    dependent and should not be trusted blindly (same caution the PE-Core template takes).
    """
    direct = root / rel
    if direct.exists():
        return direct
    matches = sorted(root.rglob(f"{video_id}.mp4"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"{video_id}: ambiguous, {len(matches)} matches under {root}")
    raise FileNotFoundError(f"{video_id}: not found under {root} (expected {rel})")


def find_video_root(manifest, root_input=None):
    override = os.environ.get("AIC_VIDEO_ROOT")
    if override:
        return Path(override)
    sample_id, sample_entry = next(iter(sorted(manifest["videos"].items())))
    sample_rel = Path(sample_entry["rel"])
    root_input = Path(root_input) if root_input is not None else Path("/kaggle/input")
    for candidate in sorted(p for p in root_input.iterdir() if p.is_dir()):
        if (candidate / sample_rel).exists():
            return candidate
    # Fall back to a suffix search for the sample video anywhere under /kaggle/input. The
    # root is the top-level /kaggle/input child that contains the match -- not a depth guess
    # from sample_rel's own part count, which breaks when the manifest's rel layout (e.g. a
    # flat "<group>/<video_id>.mp4" built against a meta.parquet) doesn't match how deep the
    # attached dataset actually nests its videos (e.g. "Videos_<group>/videos/<video_id>.mp4").
    # A too-deep root here would still resolve the sample video via resolve_video_path's own
    # rglob fallback, but would silently break every *other* group living under a sibling
    # folder outside that too-narrow root.
    matches = sorted(root_input.rglob(f"{sample_id}.mp4"))
    if matches:
        return root_input / matches[0].relative_to(root_input).parts[0]
    raise SystemExit(f"cannot find the video dataset under {root_input} holding "
                     f"{sample_rel} -- is it attached? (or set AIC_VIDEO_ROOT)")


# -- partitioning (identical shape to kaggle_pe_core_n2026_2s.py's split_videos) ---------

def split_videos(manifest, part_index, part_count):
    """Assign whole videos to balanced, deterministic partitions.

    Longest video first into the currently lightest bin, so every session computes the same
    disjoint split with no coordination.
    """
    if not 0 <= part_index < part_count:
        raise ValueError(f"part index {part_index} is outside 0..{part_count - 1}")
    bins = [[] for _ in range(part_count)]
    totals = [0] * part_count
    videos = sorted(manifest["videos"].items(),
                    key=lambda item: (-len(item[1]["frames"]), item[0]))
    for video_id, entry in videos:
        target = min(range(part_count), key=lambda index: (totals[index], index))
        bins[target].append(video_id)
        totals[target] += len(entry["frames"])
    return set(bins[part_index]), totals


# -- exact-frame ffmpeg extraction (copied from kaggle_pe_core_n2026_2s.py, seeded from this
# job's manifest instead of a stride -- see that file for the full rationale of every choice
# below; kept paste-portable rather than imported, per this repo's Kaggle-notebook convention) --

def chunk_ranges(count, limit=100):
    if count <= 0:
        return []
    chunks = -(-count // limit)
    size = -(-count // chunks)
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def ffmpeg_extract(video, ids, fps, batch, threads):
    if not ids:
        raise ValueError("ffmpeg_extract needs at least one frame id")
    relative = [frame_id - ids[0] for frame_id in ids]
    select = "+".join(f"eq(n\\,{frame_id})" for frame_id in relative)
    shutil.rmtree(batch, ignore_errors=True)
    batch.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-threads", str(threads),
        "-ss", f"{ids[0] / float(fps):.6f}",
        "-t", f"{(ids[-1] - ids[0] + 1) / float(fps):.6f}",
        "-i", str(video),
        "-vf", f"select='{select}'", "-vsync", "0",
        "-c:v", "png", "-pix_fmt", "rgb24", "-compression_level", "1",
        "-start_number", "0", str(batch / "f%06d.png"),
    ]
    result = subprocess.run(command, capture_output=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            f"{Path(video).name}: ffmpeg exited {result.returncode} for ids "
            f"{ids[0]}..{ids[-1]}\n  {' '.join(command)}\n  "
            + result.stderr.decode("utf-8", "replace").strip())
    return sorted(batch.glob("f*.png"))


def move_frames(ids, files, output):
    for frame_id, frame in zip(ids, files):
        frame.replace(output / f"{frame_id:06d}.png")


def extract_chunk(video, ids, fps, batch, threads, output, tail_retry=True):
    if not ids:
        return []
    try:
        files = ffmpeg_extract(video, ids, fps, batch, threads)
    except RuntimeError:
        files = None
    if files is not None and len(files) == len(ids):
        move_frames(ids, files, output)
        return list(ids)
    short = None if files is None else len(ids) - len(files)
    if tail_retry and short is not None and 0 < short < len(ids):
        try:
            retry = ffmpeg_extract(video, ids[:-short], fps, batch, threads)
        except RuntimeError:
            retry = None
        if retry is not None and len(retry) == len(ids) - short:
            log(f"{Path(video).name}: {short} frame(s) {ids[-short]}..{ids[-1]} have no "
                f"decodable frame; dropping them")
            move_frames(ids[:-short], retry, output)
            return list(ids[:-short])
    if len(ids) == 1:
        return []
    middle = len(ids) // 2
    return (extract_chunk(video, ids[:middle], fps, batch, threads, output, tail_retry=False)
            + extract_chunk(video, ids[middle:], fps, batch, threads, output, tail_retry=False))


def extract_exact_frames(video, frame_ids, output, fps, threads=2):
    output.mkdir(parents=True, exist_ok=True)
    requested = sorted({int(frame_id) for frame_id in frame_ids})
    batch = output / ".batch"
    written = []
    for first, last in chunk_ranges(len(requested)):
        written.extend(extract_chunk(video, requested[first:last], fps, batch, threads, output))
    shutil.rmtree(batch, ignore_errors=True)
    return written


# -- output writer ------------------------------------------------------------------------

class OcrShardWriter:
    """Append-only JSONL, one line per keyframe: ``{"key": ..., "lines": [...]}``.

    Replaces the whole-dict-rewrite-on-every-flush pattern ``run_corpus.py`` uses -- O(1) per
    write instead of O(n) -- and is trivially resumable: the ``done`` set is rebuilt from
    whatever the file already holds, so a re-run just skips what is already there.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            with open(self.path) as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self.done.add(json.loads(line)["key"])
        self.handle = open(self.path, "a", buffering=1)

    def write(self, key, lines):
        if key in self.done:
            return
        self.handle.write(json.dumps({"key": key, "lines": lines}) + "\n")
        self.done.add(key)

    def close(self):
        self.handle.close()


# -- VietOCR checkpoint staging -------------------------------------------------------------

def stage_checkpoint(ckpt_path, expected_sha256):
    """Copy the already-downloaded checkpoint to /tmp/vgg_seq2seq.pth, verified.

    VietOCR's own weight loader (``vietocr.tool.utils.download``) derives its cache filename
    from the *configured* weights URL's tail -- ``vietocr_configs/vgg-seq2seq.yml`` still
    points at ``.../vgg_seq2seq.pth`` (underscore) -- and skips downloading entirely if that
    exact path already exists. Placing the file there first is what keeps this pipeline off
    the network (and off vocr.vn's expired cert) at runtime. One rank's copy racing another's
    is handled by staging to a pid-suffixed temp name and an atomic rename.
    """
    def matches(path):
        return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256

    if matches(CKPT_TARGET):
        return CKPT_TARGET
    tmp = CKPT_TARGET.with_name(f"{CKPT_TARGET.name}.tmp.{os.getpid()}")
    shutil.copy(ckpt_path, tmp)
    if not matches(tmp):
        tmp.unlink()
        raise RuntimeError(f"{ckpt_path}: staged copy does not match expected sha256")
    tmp.replace(CKPT_TARGET)
    return CKPT_TARGET


# -- per-rank OCR loop --------------------------------------------------------------------

def run_rank(rank, nprocs, config, manifest, selected_video_ids):
    # FIRST LINE, before any torch/cv2/rapidocr/vietocr import in this process --
    # rapidocr_onnxruntime hardcodes CUDA device_id=0 internally, ignoring whatever device
    # index is passed elsewhere, so remapping visibility per-rank is the only way to actually
    # spread ranks across GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # Also before onnxruntime is imported here: ORT's CUDA provider dlopens cuDNN/cuBLAS by
    # bare soname, which only resolves if they are already in this process (see the CUDA
    # block above the dependency helpers).
    loaded, failed = preload_cuda_libs()
    log(f"GPU {rank}: preloaded {len(loaded)} CUDA libs"
        + (f"; FAILED: {failed}" if failed else ""))

    import cv2
    import torch
    from tqdm.auto import tqdm

    # Kaggle gives a GPU session four vCPUs, shared by two ranks, the ffmpeg prefetch thread
    # and every library's default thread pool. Left alone, torch and OpenCV each spawn one
    # worker per core and the run spends its time context-switching instead of working. Every
    # stage that matters here is on the GPU, so these are deliberately small.
    torch.set_num_threads(2)
    cv2.setNumThreads(1)

    sys.path.insert(0, str(config["engine_dir"]))
    from engines import OcrConfig, OcrPipeline  # noqa: E402

    stage_checkpoint(Path(config["ckpt_path"]), config["ckpt_sha256"])

    videos = [(video_id, manifest["videos"][video_id])
             for video_id in sorted(manifest["videos"]) if video_id in selected_video_ids]
    videos = videos[rank::nprocs]
    total = sum(len(entry["frames"]) for _, entry in videos)
    log(f"GPU {rank}: {len(videos)} videos, {total:,} keyframes, "
        f"{os.cpu_count()} CPUs visible to this process")

    def build(batch_size):
        return OcrPipeline(OcrConfig(
            det_use_cuda=config["det_use_cuda"], small_rescan=config["small_rescan"],
            det_model=config["det_model"], rec_device="cuda:0", rec_batch_size=batch_size,
        ))

    writer = OcrShardWriter(Path(config["parts_dir"]) / f"part{config['part_index']}" /
                            f"shard_r{rank}.jsonl")
    skipped = {}
    part_dir = Path(config["parts_dir"]) / f"part{config['part_index']}"
    part_dir.mkdir(parents=True, exist_ok=True)

    def extract_video(video_id, entry):
        target = Path(config["temp_dir"]) / f"gpu{rank}" / video_id
        shutil.rmtree(target, ignore_errors=True)
        video_path = resolve_video_path(Path(manifest["root_resolved"]), video_id, entry["rel"])
        written = extract_exact_frames(video_path, entry["frames"], target, entry["fps"])
        return target, written

    # Where each frame's time actually goes. Without this a slow run only reports "slow", and
    # on Kaggle's hardware the difference between a slow detector and a slow recogniser decides
    # what to fix -- the two want opposite remedies.
    stages = {"wait": 0.0, "read": 0.0, "detect": 0.0, "recognise": 0.0}
    # VietOCR's predict_batch raises on some ragged batches, and _recognise then silently
    # falls back to recognising one crop at a time -- several times slower, and invisible in
    # the stage totals. Counting the failures is what tells the two apart.
    counters = {"batches": 0, "failed": 0}

    def detect_and_recognise(active, image):
        """``run()`` is exactly ``_recognise(image, detect_boxes(image))``; splitting it here
        is what attributes each frame's time to the stage that spent it."""
        mark = time.monotonic()
        boxes = active.detect_boxes(image)
        split = time.monotonic()
        lines = active._recognise(image, boxes)
        done = time.monotonic()
        stages["detect"] += split - mark
        stages["recognise"] += done - split
        return lines

    def report(writer, tag):
        """Plain log(), never progress.write(): a tqdm bar redraws over its own writes when
        stdout is not a TTY, so on Kaggle the report came out as a blank line and the stage
        numbers were lost -- which is the whole reason the report exists.
        """
        counted = max(len(writer.done), 1)
        elapsed = max(time.monotonic() - started, 1e-6)
        log(f"GPU {rank}: {tag} {counted:,} frames | "
            + stage_summary(stages, counted)
            + f" | fps={counted / elapsed:.2f}"
            + f" | rec batches={counters['batches']}"
            + f" failed={counters['failed']}")

    pool = ThreadPoolExecutor(max_workers=1)
    progress = tqdm(total=total, desc=f"GPU {rank} OCR", unit="frame", position=rank,
                    dynamic_ncols=True)
    started = time.monotonic()
    last_report = time.monotonic()
    pipeline = None
    batch_size = config["rec_batch_size"]
    next_future = pool.submit(extract_video, *videos[0]) if videos else None
    try:
        for index, (video_id, entry) in enumerate(videos):
            waited = time.monotonic()
            try:
                target, written = next_future.result()
            except Exception as exc:
                skipped[video_id] = f"{type(exc).__name__}: {exc}"
                progress.write(f"GPU {rank}: SKIPPING {video_id}: {exc}")
                if index + 1 < len(videos):
                    next_future = pool.submit(extract_video, *videos[index + 1])
                continue
            stages["wait"] += time.monotonic() - waited
            if index + 1 < len(videos):
                next_future = pool.submit(extract_video, *videos[index + 1])
            if not written:
                skipped[video_id] = "no decodable frames"
                progress.write(f"GPU {rank}: SKIPPING {video_id}: no decodable frames")
                shutil.rmtree(target, ignore_errors=True)
                continue

            for frame_id in written:
                key = f"{video_group(video_id)}/{video_id}/{frame_id}"
                if key in writer.done:
                    progress.update(1)
                    continue
                mark = time.monotonic()
                image = cv2.imread(str(target / f"{frame_id:06d}.png"))
                if image is None:
                    continue
                stages["read"] += time.monotonic() - mark
                if pipeline is None:
                    # One-shot OOM probe on the very first real frame: build at the
                    # configured batch size, halve once and rebuild on OOM, keep whatever
                    # worked. VietOCR crops are small and Kaggle GPUs have at least the
                    # RTX 3060's memory the default was tuned on, so more than one halving
                    # round is not expected -- _recognise's own per-image fallback already
                    # survives an isolated OOM later in the run.
                    for attempt in range(2):
                        candidate = build(batch_size)
                        try:
                            lines = detect_and_recognise(candidate, image)
                            pipeline = candidate
                            break
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            batch_size = max(1, batch_size // 2)
                            progress.write(f"GPU {rank}: OOM at startup, "
                                          f"rec_batch_size={batch_size}")
                    else:
                        raise RuntimeError(f"GPU {rank}: OOM even at rec_batch_size=1")
                    if config["det_use_cuda"]:
                        # Ask the session, not RapidOCR. `infer.use_cuda` records the intent:
                        # it is derived from get_available_providers(), which names every
                        # provider the build was compiled with and stays True even when the
                        # CUDA provider .so failed to load and ORT quietly created a CPU
                        # session. The session's own list is the outcome.
                        landed = pipeline.detector.text_det.infer.session.get_providers()
                        if not landed or landed[0] != "CUDAExecutionProvider":
                            raise RuntimeError(
                                f"GPU {rank}: det_use_cuda=True but the detector session "
                                f"landed on {landed} -- the CUDA provider library did not "
                                f"load (see the provider check and ORT's own error above)")
                    recognizer = pipeline.recognizer
                    real_predict_batch = recognizer.predict_batch

                    def counting_predict_batch(batch, **kwargs):
                        counters["batches"] += 1
                        try:
                            return real_predict_batch(batch, **kwargs)
                        except Exception:
                            counters["failed"] += 1
                            raise

                    recognizer.predict_batch = counting_predict_batch
                else:
                    lines = detect_and_recognise(pipeline, image)
                writer.write(key, lines)
                progress.update(1)
                elapsed = max(time.monotonic() - started, 1e-6)
                progress.set_postfix(video=video_id, fps=f"{progress.n / elapsed:.2f}",
                                     bs=batch_size, ckpt=len(writer.done))
                # On a time cadence, not a video cadence: at the throughput this job has
                # actually been getting, a five-video report is hours away, and the whole
                # point of the breakdown is to be readable in the first minute.
                if time.monotonic() - last_report >= STAGE_REPORT_SECONDS:
                    last_report = time.monotonic()
                    report(writer, "progress")
            shutil.rmtree(target, ignore_errors=True)
            if (index + 1) % 5 == 0 or index + 1 == len(videos):
                progress.write(f"GPU {rank}: {index + 1}/{len(videos)} videos, "
                               f"{len(writer.done):,}/{total:,} keyframes")
                report(writer, "videos")
    finally:
        writer.close()
        progress.close()
        pool.shutdown(wait=False)

    (part_dir / f"skipped_r{rank}.json").write_text(json.dumps(skipped, indent=2, sort_keys=True))
    log(f"GPU {rank}: complete, {len(writer.done):,} keyframes, "
        f"{len(skipped)} video(s) skipped | {stage_summary(stages, len(writer.done))} | "
        f"rec batches={counters['batches']} failed={counters['failed']}")


# -- merge + verify -------------------------------------------------------------------------

def merge_and_verify(parts_dir, output_path, manifest, selected_video_ids, skipped):
    """Reassemble every rank/part's JSONL shard into the flat dict build_ocr_index.py reads.

    Fail-closed: partitioning is disjoint by whole video, so any key seen twice is a bug, and
    the total key count must equal exactly the selected, non-skipped keyframes.
    """
    merged = {}
    for shard_path in sorted(Path(parts_dir).glob("part*/shard_r*.jsonl")):
        with open(shard_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = record["key"]
                if key in merged:
                    raise RuntimeError(f"duplicate key across shards: {key} (in {shard_path})")
                merged[key] = {"lines": record["lines"]}

    expected_keys = set()
    for video_id in selected_video_ids:
        if video_id in skipped:
            continue
        entry = manifest["videos"][video_id]
        for frame_id in entry["frames"]:
            expected_keys.add(f"{video_group(video_id)}/{video_id}/{frame_id}")
    if set(merged) != expected_keys:
        missing = expected_keys - set(merged)
        extra = set(merged) - expected_keys
        raise RuntimeError(
            f"merge mismatch: {len(missing)} expected key(s) missing, "
            f"{len(extra)} unexpected key(s); e.g. missing={sorted(missing)[:3]} "
            f"extra={sorted(extra)[:3]}")
    for key, value in merged.items():
        for line in value["lines"]:
            box, conf = line.get("box"), line.get("conf")
            if not isinstance(box, list) or len(box) != 4:
                raise RuntimeError(f"{key}: malformed box {box!r}")
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                raise RuntimeError(f"{key}: malformed conf {conf!r}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged))
    log(f"verified {len(merged):,} keyframes -> {output_path}")
    return len(merged)


# -- self-test ------------------------------------------------------------------------------

def self_test():
    """Offline check of every pure/mockable piece -- no GPU, no Kaggle, no OCR deps."""
    import runpy
    import tempfile

    # mp.spawn re-imports this file as __mp_main__ in every child. If main() is not guarded,
    # each child re-runs the whole pipeline and recurses into another spawn.
    if os.environ.get("AIC_SELFTEST_REENTRY") != "1":
        os.environ["AIC_SELFTEST_REENTRY"] = "1"
        try:
            runpy.run_path(__file__, run_name="__mp_main__")
        except SystemExit as exc:
            raise AssertionError(
                f"importing as __mp_main__ ran main(); mp.spawn children would recurse ({exc})"
            ) from exc
        finally:
            os.environ.pop("AIC_SELFTEST_REENTRY", None)

    # -- partitioning: deterministic, disjoint, covers everything --
    fake = {"videos": {f"M0{g}_V{i:03d}": {"frames": list(range(0, 30 * (i + 1), 3))}
                       for g in range(1, 6) for i in range(1, 4)}}
    partitions = [split_videos(fake, index, 3)[0] for index in range(3)]
    assert set.union(*partitions) == set(fake["videos"])
    assert not any(left & right for i, left in enumerate(partitions)
                   for right in partitions[i + 1:])
    assert partitions == [split_videos(fake, index, 3)[0] for index in range(3)]

    # -- CUDA provisioning: the embedded provider-check snippet must be valid Python (a typo
    #    in it only ever surfaces on Kaggle, inside a subprocess, after a 250MB pip install),
    #    and the library search must resolve real directories. Neither needs a GPU. --
    compile(ORT_PROVIDER_CHECK, "ORT_PROVIDER_CHECK", "exec")
    cuda_dirs = cuda_library_dirs()
    assert isinstance(cuda_dirs, list), cuda_dirs
    assert all(isinstance(d, str) and Path(d).is_dir() for d in cuda_dirs), cuda_dirs
    # The snippet is run as a subprocess, so a NameError inside it reads as "CUDA not ready"
    # rather than as a typo. Compiling catches syntax, not a missing import; this does.
    for name in ("ctypes", "os", "site", "sys"):
        assert f"import {name}" in ORT_PROVIDER_CHECK or f", {name}" in ORT_PROVIDER_CHECK, name

    # -- cuda_provider_ready: the compile-time list is not enough. A build compiled with CUDA
    #    whose provider library cannot dlopen still names CUDAExecutionProvider, and that is
    #    exactly how a whole canary run ended up on CPU with every check reporting success. --
    assert cuda_provider_ready(["CUDAExecutionProvider", "CPUExecutionProvider"], True)
    assert not cuda_provider_ready(["CUDAExecutionProvider", "CPUExecutionProvider"], False)
    assert not cuda_provider_ready(["CPUExecutionProvider"], True)
    assert not cuda_provider_ready([], False)
    # The pin is load-bearing: 1.27.0 moved the PyPI wheels to CUDA 13, which cannot load on
    # a CUDA 12 image. Guard the constant itself, since a silent bump is what broke this.
    assert tuple(int(p) for p in ORT_GPU_VERSION.split(".")) < (1, 27), ORT_GPU_VERSION
    assert all(tuple(int(p) for p in v.split(".")) < (1, 27) for v in ORT_GPU_FALLBACKS), (
        ORT_GPU_FALLBACKS)

    # -- stage_summary: per-frame milliseconds, never a division by zero --
    assert stage_summary({"detect": 1.0, "recognise": 0.5}, 10) == (
        "ms/frame detect=100 recognise=50")
    assert stage_summary({"detect": 1.0}, 0) == "ms/frame detect=1000"

    # -- chunk_ranges: never past the ffmpeg select limit, never a lone tail frame --
    assert chunk_ranges(0) == []
    assert chunk_ranges(1) == [(0, 1)]
    assert chunk_ranges(100) == [(0, 100)]
    assert chunk_ranges(101) == [(0, 51), (51, 101)]
    for count in (1, 2, 99, 100, 101, 199, 301, 1000):
        ranges = chunk_ranges(count)
        assert ranges[0][0] == 0 and ranges[-1][1] == count
        assert all(b - a <= 100 for a, b in ranges), (count, ranges)
        if count > 1:
            assert ranges[-1][1] - ranges[-1][0] > 1, (count, ranges)

    # -- extract_chunk: a dropped frame never gets zipped against the wrong id --
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        damaged = 1080
        calls = []

        def fake_ffmpeg_extract(video, ids, fps, batch, threads):
            calls.append((ids[0], ids[-1]))
            shutil.rmtree(batch, ignore_errors=True)
            batch.mkdir(parents=True, exist_ok=True)
            if damaged in ids:
                position = ids.index(damaged)
                content = (ids[:position] + ids[position + 1:])[:len(ids) - 1]
            else:
                content = list(ids)
            for i, true_id in enumerate(content):
                (batch / f"f{i:06d}.png").write_text(str(true_id))
            return sorted(batch.glob("f*.png"))

        window_ids = list(range(0, 2040, 60))
        out = root / "out"
        out.mkdir()
        real = globals()["ffmpeg_extract"]
        globals()["ffmpeg_extract"] = fake_ffmpeg_extract
        try:
            written = extract_chunk(Path("torn.mp4"), window_ids, 30.0, root / ".batch", 2, out)
        finally:
            globals()["ffmpeg_extract"] = real
        assert written == [i for i in window_ids if i != damaged], written
        for frame_id in written:
            held = (out / f"{frame_id:06d}.png").read_text()
            assert held == str(frame_id), f"{frame_id} holds frame {held}: mislabelled"

    # -- resolve_video_path: direct hit, then suffix-match fallback --
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "Videos_M01" / "videos").mkdir(parents=True)
        (root / "Videos_M01" / "videos" / "M01_V001.mp4").write_bytes(b"")
        assert resolve_video_path(root, "M01_V001", "Videos_M01/videos/M01_V001.mp4") == (
            root / "Videos_M01" / "videos" / "M01_V001.mp4")
        # Wrong rel, but the file exists somewhere under root -- suffix match recovers it.
        assert resolve_video_path(root, "M01_V001", "wrong/nesting/M01_V001.mp4") == (
            root / "Videos_M01" / "videos" / "M01_V001.mp4")
        try:
            resolve_video_path(root, "M02_V001", "Videos_M02/videos/M02_V001.mp4")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing video must raise FileNotFoundError")

    # -- video_group: derived from the id, never trusted off a manifest entry --
    assert video_group("M01_V001") == "M01"
    assert video_group("S01_V012") == "S01"

    # -- filter_manifest_groups: keeps only the requested groups, recomputes the counts --
    wide_manifest = {"videos": {
        "M01_V001": {"frames": [1, 2, 3]},
        "M06_V001": {"frames": [4, 5]},
        "S01_V001": {"frames": [6]},
    }}
    narrowed = filter_manifest_groups(wide_manifest, ("M01",))
    assert set(narrowed["videos"]) == {"M01_V001"}
    assert narrowed["video_count"] == 1
    assert narrowed["keyframe_count"] == 3

    # -- find_video_root: a manifest whose rel depth doesn't match the real mount's nesting
    # (e.g. a flat "<group>/<video_id>.mp4" manifest attached against a dataset that actually
    # nests "Videos_<group>/videos/<video_id>.mp4" -- exactly what tommr211/aic-keyframes's
    # batch2_2026.json vs. the real yunsoo21 video dataset turned out to be) must still resolve
    # to the true top-level mount dir, not a too-deep guess that strands every group but the
    # sample's.
    with tempfile.TemporaryDirectory() as temp:
        kaggle_input = Path(temp) / "kaggle_input"
        mount = kaggle_input / "some-dataset-mount-name"
        for group in ("M01", "M02"):
            video_dir = mount / f"Videos_{group}" / "videos"
            video_dir.mkdir(parents=True)
            (video_dir / f"{group}_V001.mp4").write_bytes(b"")
        fake_manifest = {"videos": {
            "M01_V001": {"rel": "M01/M01_V001.mp4"},  # flat -- does not match the real nesting
            "M02_V001": {"rel": "M02/M02_V001.mp4"},
        }}
        root = find_video_root(fake_manifest, root_input=kaggle_input)
        assert root == mount, root
        # And every group -- not just the sample find_video_root probed -- resolves under it.
        for video_id, entry in fake_manifest["videos"].items():
            resolve_video_path(root, video_id, entry["rel"])

    # -- stage_engine_code: offline decode of the embedded bundle, deterministic, idempotent --
    import gzip

    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp) / "engine"
        result = stage_engine_code(base)
        assert result == base
        assert (base / "engines.py").read_bytes() == gzip.decompress(
            base64.b64decode(ENGINE_PY_GZ_B64))
        assert (base / "vietocr_configs" / "base.yml").exists()
        assert (base / "vietocr_configs" / "vgg-seq2seq.yml").exists()
        stage_engine_code(base)  # idempotent re-stage must not error

    # -- stage_checkpoint: sha256-verified copy into /tmp, mismatch raises, no network --
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "source.pth"
        source.write_bytes(b"fake checkpoint bytes")
        good_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        try:
            staged = stage_checkpoint(source, good_sha)
            assert staged == CKPT_TARGET
            assert CKPT_TARGET.read_bytes() == b"fake checkpoint bytes"
            try:
                stage_checkpoint(source, "0" * 64)
            except RuntimeError as exc:
                assert "sha256" in str(exc)
            else:
                raise AssertionError("checksum mismatch must raise")
        finally:
            CKPT_TARGET.unlink(missing_ok=True)

    # -- OcrShardWriter: resumable, idempotent, append-only --
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "shard.jsonl"
        writer = OcrShardWriter(path)
        writer.write("M01/M01_V001/3", [{"text": "a", "box": [[0, 0]] * 4, "conf": 0.9}])
        writer.write("M01/M01_V001/3", [{"text": "dup", "box": [[0, 0]] * 4, "conf": 0.1}])
        writer.close()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1, "duplicate write must be a no-op"

        reopened = OcrShardWriter(path)
        assert reopened.done == {"M01/M01_V001/3"}
        reopened.write("M01/M01_V001/11", [])
        reopened.close()
        assert len(path.read_text().strip().splitlines()) == 2

    # -- merge_and_verify: exact schema, duplicate-key and count-mismatch failure paths --
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manifest = {"videos": {
            "M01_V001": {"fps": 25.0, "frames": [3, 11]},  # no "group" field -- matches the
            "M02_V001": {"fps": 25.0, "frames": [5]},      # real manifest; group is derived
        }}
        line = {"text": "x", "box": [[0, 0], [1, 0], [1, 1], [0, 1]], "conf": 0.8}
        part0 = root / "parts" / "part0"
        part0.mkdir(parents=True)
        (part0 / "shard_r0.jsonl").write_text(
            json.dumps({"key": "M01/M01_V001/3", "lines": [line]}) + "\n" +
            json.dumps({"key": "M01/M01_V001/11", "lines": []}) + "\n")
        (part0 / "shard_r1.jsonl").write_text(
            json.dumps({"key": "M02/M02_V001/5", "lines": []}) + "\n")

        count = merge_and_verify(root / "parts", root / "out.json", manifest,
                                 {"M01_V001", "M02_V001"}, {})
        assert count == 3
        merged = json.loads((root / "out.json").read_text())
        assert merged["M01/M01_V001/3"] == {"lines": [line]}

        # Duplicate key across shards must raise.
        (part0 / "shard_r2.jsonl").write_text(
            json.dumps({"key": "M01/M01_V001/3", "lines": []}) + "\n")
        try:
            merge_and_verify(root / "parts", root / "out2.json", manifest,
                             {"M01_V001", "M02_V001"}, {})
        except RuntimeError as exc:
            assert "duplicate key" in str(exc)
        else:
            raise AssertionError("duplicate key across shards must raise")
        (part0 / "shard_r2.jsonl").unlink()

        # A skipped video contributes no shard rows and must not be expected in the merge.
        part1 = root / "parts_skipped" / "part0"
        part1.mkdir(parents=True)
        (part1 / "shard_r0.jsonl").write_text(
            json.dumps({"key": "M01/M01_V001/3", "lines": [line]}) + "\n" +
            json.dumps({"key": "M01/M01_V001/11", "lines": []}) + "\n")
        count = merge_and_verify(root / "parts_skipped", root / "out3.json", manifest,
                                 {"M01_V001", "M02_V001"}, {"M02_V001": "no decodable frames"})
        assert count == 2

    print("self-test OK")


# -- main -----------------------------------------------------------------------------------

def main():
    if os.environ.get("AIC_SELFTEST_REENTRY") == "1":
        raise AssertionError("main() ran while the file was re-imported as __mp_main__")

    if "--self-test" in sys.argv:
        self_test()
        return

    if not Path("/kaggle/input").exists():
        raise SystemExit("full runs are only supported inside Kaggle (or use --self-test)")

    # A run costs GPU-hours, so the cheap offline checks go first.
    self_test()

    part_index = int(os.environ.get("AIC_PART_INDEX", PART_INDEX))
    part_count = int(os.environ.get("AIC_PART_COUNT", PART_COUNT))
    canary = os.environ.get("AIC_CANARY", "1" if CANARY else "0") == "1"
    rec_batch_size = int(os.environ.get("AIC_REC_BATCH_SIZE", REC_BATCH_SIZE))

    ensure_ocr_deps()

    manifest = filter_manifest_groups(load_manifest(), GROUPS)
    log(f"filtered to {GROUPS}: {manifest['video_count']} videos, "
        f"{manifest['keyframe_count']:,} keyframes")
    video_root = find_video_root(manifest)
    engine_dir = stage_engine_code(WORK / ".engine")
    ckpt_path = download_checkpoint(WORK / ".ckpt")
    manifest["root_resolved"] = str(video_root)
    log(f"video root: {video_root}; engine: {engine_dir}; checkpoint: {ckpt_path}")

    selected_video_ids, partition_totals = split_videos(manifest, part_index, part_count)
    only = os.environ.get("AIC_ONLY_VIDEOS")
    only_ids = {v.strip() for v in only.split(",") if v.strip()} if only else set(ONLY_VIDEOS)
    if only_ids:
        unknown = sorted(only_ids - set(manifest["videos"]))
        if unknown:
            raise SystemExit(f"ONLY_VIDEOS are not in this manifest: {unknown}")
        selected_video_ids = only_ids
        log(f"ONLY_VIDEOS: OCR-ing just these {len(only_ids)} video(s)")
    if canary:
        selected_video_ids = set(sorted(selected_video_ids)[:2])
    expected = sum(len(manifest["videos"][video_id]["frames"])
                   for video_id in selected_video_ids)

    tag = "_".join(sorted(GROUPS))
    output = WORK / "ocr" / f"batch2_{tag}_part{part_index}.json"
    parts = WORK / ".ocr_parts" / tag
    temp = FRAME_DIR / tag
    for path in (parts, temp):
        shutil.rmtree(path, ignore_errors=True)

    # mp.spawn uses the "spawn" start method, so each rank's environment is fixed at exec
    # time: the loader path has to be set here, not inside run_rank where it would already be
    # too late. run_rank preloads the same directories into its own process too; this covers
    # whatever it does not.
    cuda_dirs = cuda_library_dirs()
    if cuda_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            cuda_dirs + [p for p in [os.environ.get("LD_LIBRARY_PATH", "")] if p])
        log(f"LD_LIBRARY_PATH seeded with {len(cuda_dirs)} CUDA lib dir(s)")

    import torch

    nprocs = max(1, min(2, torch.cuda.device_count()))
    log(f"part {part_index}/{part_count}; selected_videos={len(selected_video_ids)}; "
        f"selected_frames={expected:,}; partition_frames={partition_totals}; "
        f"gpus={nprocs}; canary={canary}")
    config = {
        "part_index": part_index, "parts_dir": str(parts), "temp_dir": str(temp),
        "engine_dir": str(engine_dir), "ckpt_path": str(ckpt_path), "ckpt_sha256": CKPT_SHA256,
        "det_use_cuda": DET_USE_CUDA, "small_rescan": SMALL_RESCAN, "det_model": DET_MODEL,
        "rec_batch_size": rec_batch_size,
    }
    spawnable = getattr(sys.modules.get("__main__"), "__file__", None) is not None
    if nprocs == 1 or not spawnable:
        if nprocs > 1:
            log(f"{nprocs} GPUs but no importable __main__; running ranks sequentially")
        for rank in range(nprocs):
            run_rank(rank, nprocs, config, manifest, selected_video_ids)
    else:
        import torch.multiprocessing as mp
        mp.spawn(run_rank, args=(nprocs, config, manifest, selected_video_ids),
                 nprocs=nprocs, join=True)

    skipped = {}
    for path in sorted(Path(parts).glob("part*/skipped_r*.json")):
        skipped.update(json.loads(path.read_text()))
    if skipped:
        log(f"WARNING: {len(skipped)} of {len(selected_video_ids)} video(s) skipped:")
        for video_id, reason in sorted(skipped.items()):
            log(f"  {video_id}: {reason}")

    merge_and_verify(parts, output, manifest, selected_video_ids, skipped)
    if skipped:
        output.with_suffix(".skipped.json").write_text(
            json.dumps(skipped, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(parts, ignore_errors=True)
    shutil.rmtree(temp, ignore_errors=True)
    log(f"DONE: {expected:,} keyframes selected over "
        f"{len(selected_video_ids) - len(skipped)} videos -> {output}")


if __name__ == "__main__":
    main()
