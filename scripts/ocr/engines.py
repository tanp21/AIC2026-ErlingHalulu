"""The OCR pipeline: detect, rescan small text, rectify, recognise.

Structure and defaults come from measurements on the three acceptance frames in
``benchmarks/ocr/ocr_ground_truth.json``; ``docs/ocr-search.md`` records the numbers.
The short version of what the bake-off established:

* **Detection** — PP-OCRv4's DBNet (via ``rapidocr-onnxruntime``) is excellent.
  On the pillar frame it finds all 17 lines with near-exact boxes.
* **Recognition** — PP-OCRv4's bundled recogniser is the *Chinese* model and
  strips every Vietnamese diacritic (``HỎA HỒNG NHẬT TẢO`` came back as
  ``HOA HONG NHAT TAO``), scoring 2/14 on the pillar. VietOCR's ``vgg_seq2seq``
  scores 13/14 and is five times faster than its ``vgg_transformer`` sibling.
* **Rectification is not optional.** Feeding axis-aligned bounding-box crops of
  tilted text to VietOCR produces pure noise (``markercán``, ``CONTRACTION``).
  Warping the detection *polygon* to a horizontal rectangle first is what makes
  the perspective-tilted recipe card readable at all.
* **Small isolated text needs a second look.** A 11x15px flood-gauge digit is
  invisible to whole-frame detection at any affordable resolution — recovering
  it needs roughly 8x magnification, which costs 11s/frame applied to the whole
  image. Applied only to the *neighbourhood of small detections already found*,
  it costs a fraction of that and recovers the digit. See :meth:`_rescan_small`.
* **A VLM tier was evaluated and rejected.** Qwen2.5-VL-3B scored 0/7 on the
  gauge (it never resolves the digits), 10/14 on the pillar, and hallucinated
  plausible-but-wrong Vietnamese on the recipe card, at 5-16s per frame. Worse
  than this pipeline on every acceptance frame and orders of magnitude slower.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class OcrConfig:
    """Every knob the pipeline exposes. Serialised into the resume manifest.

    Changing any value changes :meth:`fingerprint`, which invalidates previously
    extracted videos so a re-run does not silently mix outputs from two configs.
    """

    # --- detection -------------------------------------------------------
    det_limit_side_len: int = 736      # limit_type is "min": the SHORT side floor
    det_box_thresh: float = 0.2
    det_thresh: float = 0.15
    det_unclip_ratio: float = 1.8
    det_use_cuda: bool = False
    # "v4" uses the bundled rapidocr-onnxruntime DBNet; "v5-mobile"/"v5-server"
    # pull PP-OCRv5 detectors through the newer `rapidocr` package.
    #
    # v4 wins on measurement despite v5 being newer. v5-mobile does resolve the
    # flood-gauge digit unaided (making the rescan redundant), but it emits extra
    # spurious boxes outside the text columns, and those break the reading-order
    # cut on the two-column pillar — phrase recall drops 0.667 -> 0.500. Correct
    # reading order is worth more than a cheaper gauge.
    det_model: str = "v4"

    # --- small-text rescan ----------------------------------------------
    small_rescan: bool = True
    small_max_width: int = 60          # "small isolated" = narrow AND short;
    small_max_height: int = 40         # a long thin news ticker must not qualify
    small_cluster_gap: float = 4.0     # multiples of median small height
    small_margin: float = 3.0
    small_target_height: int = 240     # ~8x for 30px text; 7/7 on the gauge frame
    small_max_rois: int = 6
    small_max_megapixels: float = 48.0  # hard ceiling on rescan work per frame

    # --- crop rectification ----------------------------------------------
    # Padding is small on purpose. More headroom sounds like it should protect
    # tone marks, but measured on the pillar frame it makes things worse
    # (12/14 at 0.02 vs 10/14 at 0.10): a wider crop rescales the glyphs down
    # inside a fixed-height recogniser input, and the marks lose more to
    # resampling than they gain in margin.
    rect_pad_x: float = 0.02
    rect_pad_y: float = 0.02
    rect_min_height: int = 48

    # --- recognition -------------------------------------------------------
    # vgg_seq2seq over vgg_transformer on evidence: more accurate on the
    # acceptance frames *and* roughly 3x faster. Ensembling the two by
    # confidence was tried and rejected — it tracks the weaker model.
    rec_model: str = "vgg_seq2seq"
    rec_device: str = "cuda:0"
    rec_batch_size: int = 32
    # CLAHE on the crop's lightness channel before recognition. Worth a measured
    # +1 on the pillar frame (11/14 -> 12/14): it recovers HỎA and TẢO, whose
    # tone marks are low-contrast gold-on-white. Ten preprocessing variants were
    # compared; contrast normalisation was the only one that helped, and
    # majority-voting across variants was worse than this alone. 0 disables.
    rec_clahe_clip: float = 2.0
    rec_clahe_grid: int = 8
    # Drops detector hallucinations on textured background (a doorframe scored
    # 0.32 and, being positioned outside the text columns, corrupted the
    # reading-order cut). Real text on these frames sits well above this.
    rec_min_conf: float = 0.45
    rec_min_length: int = 1
    # Second recogniser for lines the first one was unsure of. Off by default on
    # measurement, and the measurement is worth recording because the idea looks
    # obviously good.
    #
    # It does what it promises: the two models fail differently, so where
    # vgg_seq2seq collapses into a repetition loop on the reference signboard's
    # tagline ("thu thu thu thu...", 0.26 confidence) vgg_transformer returns a
    # legible 0.76, and that frame's character error rate falls from 0.62 to
    # 0.38. But rescuing a line by raising its confidence does not make the line
    # real, and the confidence floor is what keeps detector hallucinations out.
    # Enabling this readmitted a phantom "MEHIM" into the pillar frame, cost that
    # frame KHẤP, and cost the signboard "Mộc Châu" — a place name, which is
    # exactly what a query asks for. Required-string recall over the four
    # reference frames fell from 1.000 to 0.857.
    #
    # Set to "vgg_transformer" to trade precision for recall on marginal lines.
    rec_fallback_model: str = ""
    rec_fallback_below: float = 0.60
    # Make a word's tone marks agree with the majority spelling of the same word
    # elsewhere in the frame. Costs nothing (pure string work, no model) and is
    # worth +2 required strings on the recipe card. See
    # scripts.ocr.postprocess.harmonize_diacritics.
    harmonize: bool = True

    # --- optional vision-LLM post-correction --------------------------------
    # Off by default: it triples to octuples per-frame cost, which no
    # whole-corpus sweep can absorb. Worth enabling on a shortlist of
    # low-confidence frames, where it measurably repairs tone marks and whole
    # words. See scripts/ocr/postcorrect.py.
    postcorrect: bool = False
    postcorrect_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    postcorrect_4bit: bool = True

    # --- optional second reader ---------------------------------------------
    # An independent Vietnamese vision-LLM shown cropped text blocks, merged in
    # additively. It exists to recover lines the recogniser cannot read at all,
    # which no amount of correction can fix because the corrector is not allowed
    # to invent text. Worth 0.618 -> 0.197 CER on the reference signboard, whose
    # tagline collapses the recogniser into a repetition loop. Costs a few
    # seconds per frame; requires `postcorrect` for its merge target to be the
    # corrected text rather than the raw draft. See scripts/ocr/reader.py.
    reader: bool = False
    reader_model: str = "5CD-AI/Vintern-1B-v3_5"

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict) -> "OcrConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def is_degenerate(text: str, *, min_repeats: int = 4) -> bool:
    """Whether a transcription is a decoder repetition loop rather than a reading.

    A sequence-to-sequence recogniser fed an unreadable crop does not fail
    cleanly — it latches onto one token and emits it until the length limit.
    Measured on the reference signboard's tagline: ``'Viêm lới Hiện nhườ, ... thu
    thu thu thu thu thu thu thu'``. Confidence alone does not reliably catch
    these (they can score in the middle of the range), and admitting one is
    expensive: it survives post-correction, which has nothing to ground a repair
    on, and it inflated that frame's character error rate from 0.62 to 1.55.
    """
    words = text.split()
    if len(words) < min_repeats:
        return False
    run = 1
    for previous, current in zip(words, words[1:]):
        run = run + 1 if current == previous else 1
        if run >= min_repeats:
            return True
    return False


def polygon_bbox(box) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def order_corners(box) -> np.ndarray:
    """Order four polygon points as top-left, top-right, bottom-right, bottom-left.

    Uses the coordinate sum/difference trick, which is orientation-stable for the
    near-rectangular quads a text detector emits.
    """
    pts = np.asarray(box, dtype=np.float32)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(total)], pts[np.argmin(diff)],
         pts[np.argmax(total)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def rectify(image: np.ndarray, box, *, pad_x: float, pad_y: float, min_height: int):
    """Warp a detection polygon to an upright rectangle, padded and upscaled.

    Returns ``None`` for degenerate polygons. The padding is proportional rather
    than absolute so it scales with text size, and the vertical component is the
    larger of the two because Vietnamese tone marks sit *above* the glyph body —
    a tight polygon clips them and turns ``QUỶ`` into ``QUY``.
    """
    ordered = order_corners(box)
    tl, tr, br, bl = ordered
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 4 or height < 4:
        return None
    px, py = int(round(width * pad_x)), int(round(height * pad_y))
    out_w, out_h = width + 2 * px, height + 2 * py
    destination = np.array(
        [[px, py], [out_w - px, py], [out_w - px, out_h - py], [px, out_h - py]],
        dtype=np.float32,
    )
    warped = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(ordered, destination),
        (out_w, out_h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    if out_h < min_height:
        scale = min_height / out_h
        warped = cv2.resize(
            warped, (max(1, int(out_w * scale)), min_height), interpolation=cv2.INTER_CUBIC
        )
    return warped


def cluster_boxes(boxes: list[tuple[float, float, float, float]], gap: float):
    """Single-link clustering of bounding boxes that lie within ``gap`` of each other.

    Clustering matters more than it looks: taking one global union of every small
    detection produced a wide ROI (the gauge column *plus* an unrelated ticker
    logo), which diluted the magnification budget and failed to recover the
    digit. Clustering keeps each ROI tight.
    """
    clusters: list[list[tuple[float, float, float, float]]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        for cluster in clusters:
            cx0 = min(c[0] for c in cluster)
            cy0 = min(c[1] for c in cluster)
            cx1 = max(c[2] for c in cluster)
            cy1 = max(c[3] for c in cluster)
            if (box[0] < cx1 + gap and box[2] > cx0 - gap
                    and box[1] < cy1 + gap and box[3] > cy0 - gap):
                cluster.append(box)
                break
        else:
            clusters.append([box])
    return clusters


def _intersection(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    intersection = _intersection(a, b)
    if not intersection:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def containment(inner: tuple[float, float, float, float],
                outer: tuple[float, float, float, float]) -> float:
    """Fraction of ``inner`` that lies inside ``outer``.

    IoU is the wrong test for de-duplicating rescan output. Magnifying a region
    makes the detector split one long line into several short ones, and a short
    fragment of a long box has *low* IoU with it — a 60px word inside a 670px
    news ticker scores about 0.09 — so it survives an IoU filter and the ticker
    gets transcribed twice. Containment catches the fragment; IoU catches the
    near-identical re-detection. Both are needed.
    """
    intersection = _intersection(inner, outer)
    if not intersection:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return intersection / area if area > 0 else 0.0


VIETOCR_CONFIG_DIR = Path(__file__).resolve().parent / "vietocr_configs"


def _load_vietocr_config(name: str):
    """Build a VietOCR ``Cfg`` without touching the network when possible.

    ``vietocr.tool.config.Cfg.load_config_from_name`` fetches both the base
    config and the model-specific config from ``vocr.vn`` on *every* call, with
    no caching -- unlike the model weights, which VietOCR's own
    ``download_weights`` does cache (by filename, under ``tempfile.gettempdir()``,
    which is why placing an already-downloaded ``vgg_seq2seq.pth``/
    ``vgg_transformer.pth`` at ``/tmp/<name>.pth`` is enough to unblock that half
    on its own). As of 2026-09 that host's TLS certificate has expired, which
    fails closed for everyone, not just this environment -- verified with a raw
    ``ssl`` handshake, independent of this pipeline's own HTTP client.

    ``VIETOCR_CONFIG_DIR`` vendors the same three YAML files VietOCR would have
    fetched (pulled from the project's own GitHub source, not vocr.vn), so this
    reproduces ``load_config_from_name``'s merge (base, then the model config on
    top) entirely from local, checked-in files. Falls back to the network path
    if the vendored files are ever missing, so this doesn't silently mask a
    real fix once upstream's cert is renewed.
    """
    import yaml
    from vietocr.tool.config import Cfg, url_config

    base_path = VIETOCR_CONFIG_DIR / "base.yml"
    model_path = VIETOCR_CONFIG_DIR / url_config[name]
    if not (base_path.exists() and model_path.exists()):
        return Cfg.load_config_from_name(name)
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config.update(yaml.safe_load(model_path.read_text(encoding="utf-8")))
    return Cfg(config)


class OcrPipeline:
    """Detector + optional small-text rescan + rectifier + Vietnamese recogniser.

    Both models are loaded lazily on first use so that importing this module (in
    tests, or to read :class:`OcrConfig`) costs nothing and needs no GPU.
    """

    def __init__(self, config: OcrConfig | None = None):
        self.config = config or OcrConfig()
        self._detector = None
        self._recognizer = None
        self._fallback = None
        self._corrector = None
        self._reader = None

    # -- lazy model loading -------------------------------------------------

    @property
    def detector(self):
        if self._detector is None:
            cfg = self.config
            if cfg.det_model == "v4":
                from rapidocr_onnxruntime import RapidOCR

                engine = RapidOCR(
                    det_use_cuda=cfg.det_use_cuda,
                    det_box_thresh=cfg.det_box_thresh,
                    det_thresh=cfg.det_thresh,
                    det_unclip_ratio=cfg.det_unclip_ratio,
                )
                engine.text_det.limit_side_len = cfg.det_limit_side_len
            else:
                from rapidocr import (
                    LangDet, LangRec, ModelType, OCRVersion, RapidOCR,
                )

                ocr_version, model_type = {
                    "v5-mobile": (OCRVersion.PPOCRV5, ModelType.MOBILE),
                    "v5-server": (OCRVersion.PPOCRV5, ModelType.SERVER),
                    "v6-medium": (OCRVersion.PPOCRV6, ModelType.MEDIUM),
                }[cfg.det_model]
                engine = RapidOCR(params={
                    "Global.log_level": "error",
                    "Det.ocr_version": ocr_version,
                    "Det.lang_type": LangDet.CH,
                    "Det.model_type": model_type,
                    "Det.box_thresh": cfg.det_box_thresh,
                    "Det.thresh": cfg.det_thresh,
                    "Det.unclip_ratio": cfg.det_unclip_ratio,
                    "Det.limit_side_len": cfg.det_limit_side_len,
                    # Recognition is VietOCR's job; this recogniser is never
                    # consulted, but the package requires one to be configured.
                    "Rec.ocr_version": OCRVersion.PPOCRV4,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.MOBILE,
                })
            self._detector = engine
        return self._detector

    def _load_recognizer(self, name: str):
        from vietocr.tool.predictor import Predictor

        cfg = _load_vietocr_config(name)
        cfg["device"] = self.config.rec_device
        # The CNN backbone weights arrive with the recogniser checkpoint;
        # fetching ImageNet weights on top of that is a pure waste and fails
        # outright on an offline machine.
        cfg["cnn"]["pretrained"] = False
        return Predictor(cfg)

    @property
    def recognizer(self):
        if self._recognizer is None:
            self._recognizer = self._load_recognizer(self.config.rec_model)
        return self._recognizer

    @property
    def fallback_recognizer(self):
        if self._fallback is None:
            self._fallback = self._load_recognizer(self.config.rec_fallback_model)
        return self._fallback

    # -- stages -------------------------------------------------------------

    def _detect(self, image: np.ndarray) -> list[list[list[float]]]:
        if self.config.det_model == "v4":
            result, _ = self.detector(image)
            return [box for box, _text, _conf in (result or [])]
        output = self.detector(image)
        boxes = getattr(output, "boxes", None)
        if boxes is None:
            return []
        return [[[float(x), float(y)] for x, y in box] for box in boxes]

    def _enhance(self, crop: np.ndarray) -> np.ndarray:
        """Normalise local contrast on the lightness channel only.

        Applied in LAB rather than on RGB so hue is untouched — the gold-on-white
        signage this was tuned against loses its tone marks to low luminance
        contrast, not to colour.
        """
        cfg = self.config
        if cfg.rec_clahe_clip <= 0:
            return crop
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        equalised = cv2.createCLAHE(
            clipLimit=cfg.rec_clahe_clip,
            tileGridSize=(cfg.rec_clahe_grid, cfg.rec_clahe_grid),
        ).apply(lightness)
        return cv2.cvtColor(cv2.merge((equalised, a, b)), cv2.COLOR_LAB2BGR)

    def _rescan_small(self, image: np.ndarray, boxes) -> list:
        """Re-detect around clusters of small isolated text at high magnification.

        Whole-frame detection cannot resolve very small glyphs, and raising the
        resolution of the *entire* frame enough to fix that is unaffordable. But
        small text rarely appears alone: where the detector already found some
        tiny boxes, more tiny text is likely nearby and just below threshold.
        Magnifying only those neighbourhoods costs a fraction of a full-frame
        high-resolution pass.
        """
        cfg = self.config
        height, width = image.shape[:2]
        small = [
            polygon_bbox(b) for b in boxes
            if polygon_bbox(b)[2] - polygon_bbox(b)[0] < cfg.small_max_width
            and polygon_bbox(b)[3] - polygon_bbox(b)[1] < cfg.small_max_height
        ]
        if not small:
            return []
        median_height = float(np.median([b[3] - b[1] for b in small])) or 1.0
        clusters = cluster_boxes(small, cfg.small_cluster_gap * median_height)
        clusters.sort(key=len, reverse=True)

        existing = [polygon_bbox(b) for b in boxes]
        scale = cfg.small_target_height / median_height
        margin = cfg.small_margin * median_height
        budget = cfg.small_max_megapixels * 1e6
        found = []
        for cluster in clusters[: cfg.small_max_rois]:
            x0 = int(max(0, min(c[0] for c in cluster) - margin))
            y0 = int(max(0, min(c[1] for c in cluster) - margin))
            x1 = int(min(width, max(c[2] for c in cluster) + margin))
            y1 = int(min(height, max(c[3] for c in cluster) + margin))
            roi = image[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            cost = roi.shape[0] * roi.shape[1] * scale * scale
            if cost > budget:
                continue
            budget -= cost
            enlarged = cv2.resize(
                roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            for box in self._detect(enlarged):
                mapped = [[p[0] / scale + x0, p[1] / scale + y0] for p in box]
                bounds = polygon_bbox(mapped)
                # The rescan exists to find text too small to detect at frame
                # scale, so anything it returns that is not itself small is a
                # re-detection of something already known. Accepting those is how
                # a news ticker ends up transcribed twice.
                width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
                if (width > cfg.small_max_width * 1.5
                        or height > cfg.small_max_height * 1.5):
                    continue
                if any(iou(bounds, other) >= 0.3
                       or containment(bounds, other) >= 0.6
                       for other in existing):
                    continue
                existing.append(bounds)
                found.append(mapped)
        return found

    def _retry_unsure(self, crops, texts: list[str], confidences: list[float]) -> None:
        """Re-read the crops the primary recogniser was unsure of, in place.

        Only the lower-confidence tail is retried, so the slower model runs on a
        small minority of lines. The more confident reading wins; where the
        primary collapsed into a repetition loop it is replaced outright, since
        such an output is not a reading at all whatever confidence it carries.
        """
        cfg = self.config
        unsure = [
            index for index, conf in enumerate(confidences)
            if conf < cfg.rec_fallback_below or is_degenerate(texts[index])
        ]
        if not unsure:
            return
        for start in range(0, len(unsure), cfg.rec_batch_size):
            chunk = unsure[start:start + cfg.rec_batch_size]
            batch = [crops[index] for index in chunk]
            try:
                retry_texts, retry_probs = self.fallback_recognizer.predict_batch(
                    batch, return_prob=True
                )
            except Exception:
                retry_texts, retry_probs = [], []
                for crop in batch:
                    text, prob = self.fallback_recognizer.predict(crop, return_prob=True)
                    retry_texts.append(text)
                    retry_probs.append(prob)
            for index, text, prob in zip(chunk, retry_texts, retry_probs):
                prob = float(prob)
                if is_degenerate(text):
                    continue
                if prob > confidences[index] or is_degenerate(texts[index]):
                    texts[index], confidences[index] = text, prob

    def _recognise(self, image: np.ndarray, boxes) -> list[dict]:
        from PIL import Image

        cfg = self.config
        crops, kept = [], []
        for box in boxes:
            warped = rectify(
                image, box,
                pad_x=cfg.rect_pad_x, pad_y=cfg.rect_pad_y, min_height=cfg.rect_min_height,
            )
            if warped is None:
                continue
            warped = self._enhance(warped)
            crops.append(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))
            kept.append(box)
        if not crops:
            return []

        texts: list[str] = []
        confidences: list[float] = []
        for start in range(0, len(crops), cfg.rec_batch_size):
            batch = crops[start:start + cfg.rec_batch_size]
            # predict_batch requires uniform-ish inputs and raises on some
            # ragged batches; per-image prediction is the safe fallback.
            try:
                predicted = self.recognizer.predict_batch(batch, return_prob=True)
                batch_texts, batch_probs = predicted
            except Exception:
                batch_texts, batch_probs = [], []
                for crop in batch:
                    text, prob = self.recognizer.predict(crop, return_prob=True)
                    batch_texts.append(text)
                    batch_probs.append(prob)
            texts.extend(batch_texts)
            confidences.extend(float(p) for p in batch_probs)

        if cfg.rec_fallback_model:
            self._retry_unsure(crops, texts, confidences)

        lines = []
        for box, text, conf in zip(kept, texts, confidences):
            text = (text or "").strip()
            if len(text) < cfg.rec_min_length or conf < cfg.rec_min_conf:
                continue
            if is_degenerate(text):
                continue
            lines.append({
                "text": text,
                "box": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in box],
                "conf": round(conf, 4),
            })
        return lines

    # -- public API ---------------------------------------------------------

    def detect_boxes(self, image: np.ndarray):
        """Return text polygons, including the configured small-text rescan."""
        boxes = self._detect(image)
        if self.config.small_rescan:
            boxes = boxes + self._rescan_small(image, boxes)
        return boxes

    @property
    def corrector(self):
        if self._corrector is None:
            from scripts.ocr.postcorrect import VlmPostCorrector

            self._corrector = VlmPostCorrector(
                self.config.postcorrect_model,
                device=self.config.rec_device,
                load_in_4bit=self.config.postcorrect_4bit,
            )
        return self._corrector

    @property
    def block_reader(self):
        if self._reader is None:
            from scripts.ocr.reader import VinternBlockReader

            self._reader = VinternBlockReader(
                self.config.reader_model, device=self.config.rec_device
            )
        return self._reader

    def run(self, image: np.ndarray) -> list[dict]:
        """OCR one BGR frame into the raw line records written by ``run_ocr``."""
        return self._recognise(image, self.detect_boxes(image))

    def read(self, image: np.ndarray, *, ignore_regions=None) -> str:
        """OCR one frame all the way to a single reading-order string.

        This is the whole pipeline including the optional post-correction stage,
        which needs the finished, ordered text rather than the per-line records
        :meth:`run` returns — the vision-LLM corrects far better with the whole
        frame's text in front of it than line by line.

        ``ignore_regions`` masks out burned-in station furniture as normalised
        ``(x0, y0, x1, y1)`` boxes. Masking happens before ordering rather than
        after, because a logo does not merely add junk text: sitting outside the
        content columns, it distorts the projection profiles the XY-cut reads and
        can change the order of everything else.
        """
        from scripts.ocr.postprocess import (
            Line, drop_regions, harmonize_diacritics, reading_order, render_text,
            to_lines,
        )

        height, width = image.shape[:2]
        boxes = self.detect_boxes(image)
        lines = drop_regions(
            to_lines(self._recognise(image, boxes)),
            ignore_regions, width=width, height=height,
        )
        # For the second reader, keep the *detection* boxes rather than the
        # recognised lines. The regions worth handing it are precisely the ones
        # the recogniser failed on, and those have already been dropped by the
        # confidence floor by this point — feeding it the survivors would ask it
        # to re-read only what already worked.
        detected = drop_regions(
            [Line(text="?", box=tuple((float(x), float(y)) for x, y in box))
             for box in boxes],
            ignore_regions, width=width, height=height,
        )
        masked_boxes = [[list(point) for point in line.box] for line in detected]
        ordered = reading_order(lines)
        if self.config.harmonize:
            # After masking, so a logo the recogniser reads inconsistently cannot
            # cast a vote on the spelling of real content.
            ordered = harmonize_diacritics(ordered)
        text = render_text(ordered)
        if not text:
            return text
        if self.config.postcorrect:
            text = self._postcorrect(image, text, ordered, lines)
        if self.config.reader:
            from scripts.ocr.postcorrect import absorb_reader

            # Additive merge, and the masked boxes rather than the raw ones: a
            # block built around the station crawl would have the reader
            # transcribe exactly the text the mask exists to remove.
            text = absorb_reader(text, self.block_reader.read(image, masked_boxes))
        return text

    def _postcorrect(self, image: np.ndarray, text: str, ordered, lines) -> str:
        from PIL import Image

        from scripts.ocr.postprocess import deskew_angle

        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Carry each word's recognition confidence through, so the corrector can
        # defer to the recogniser on words it was already sure of.
        words = [
            (word, line.conf)
            for line in ordered
            for word in line.text.strip().split()
        ]
        # The corrector reads a tilted page badly; the same tilt the reading-order
        # pass already measures tells it when to look again at a straightened copy.
        return self.corrector.correct(
            rgb, text, words, deskew=deskew_angle(lines)
        )

    def run_path(self, path) -> list[dict]:
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"could not read image: {path}")
        return self.run(image)
