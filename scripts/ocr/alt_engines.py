"""Alternative OCR/VLM engines for the 2026-09 shot-audit bake-off.

These are comparison points against the tuned pipeline in
``scripts.ocr.engines``, not replacements for it. Each function here matches
``OcrPipeline.read()``'s contract -- ``read(image, *, ignore_regions=None) ->
str`` -- so ``scripts.benchmarks.bench_ocr`` can score them with the same
``scripts.ocr.scoring.score_frame`` used for the tuned pipeline, without
routing them through ``OcrConfig``/``OcrPipeline`` at all.

``ignore_regions`` are masked by painting them black before the image reaches
the engine, rather than by dropping recognised lines after the fact (the
tuned pipeline's approach in ``OcrPipeline.read``). That's simpler and works
uniformly across engines that don't all expose per-line boxes the way
``OcrPipeline`` does.

None of these are installed by default:

* ``read_gemini`` needs only ``GEMINI_API_KEY`` -- ``google-genai`` is already
  a root dependency (see ``scripts/captioning/caption_shots_gemini.py``). It is an
  online, paid call and therefore not eligible for the offline production
  container; it exists here purely as a bake-off reference point.
* ``read_paddleocr`` and ``read_easyocr`` need ``paddleocr``/``paddlepaddle``
  and ``easyocr`` respectively, installed in an isolated venv (their torch/
  onnxruntime pins are prone to colliding with the existing ``.venv-ocr``).
* ``read_tesseract`` needs the system ``tesseract-ocr`` binary plus the
  ``tesseract-ocr-vie`` language pack, and the lightweight ``pytesseract``
  binding.

The PaddleOCR/EasyOCR adapters below follow each package's documented API but
have not been exercised against an installed engine yet -- validate the
result shape (box/text/conf unpacking) once the venv exists, before trusting
bake-off numbers built on them.
"""
from __future__ import annotations

import numpy as np


def _mask_ignore_regions(image: np.ndarray, ignore_regions) -> np.ndarray:
    """Paint normalised (x0,y0,x1,y1) boxes black before the engine sees them."""
    if not ignore_regions:
        return image
    image = image.copy()
    height, width = image.shape[:2]
    for x0, y0, x1, y1 in ignore_regions:
        image[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)] = 0
    return image


# --- Gemini as a VLM reader --------------------------------------------------

READ_PROMPT_GEMINI = (
    "Đọc chính xác toàn bộ chữ nhìn thấy được trong ảnh, theo đúng thứ tự đọc "
    "tự nhiên (trái sang phải, trên xuống dưới, cột nào xong cột đó). Giữ "
    "nguyên dấu tiếng Việt nếu chắc chắn, nếu không chắc thì viết không dấu. "
    "Bỏ qua logo kênh truyền hình, đồng hồ góc màn hình, banner tin chạy phía "
    "dưới, và chữ 'REPLAY'. Nếu ảnh không có chữ, in ra chuỗi rỗng. Chỉ in ra "
    "chữ đọc được, không giải thích, không thêm chú thích."
)

def _collapse_repeated_lines(text: str) -> str:
    """Dedupe immediate-repeat lines.

    Gemini hits the same repetition-loop failure mode VietOCR does (see
    ``scripts.ocr.engines.is_degenerate``): shown a frame with a sponsor logo
    stencilled many times along a barrier, it reads the rest of the frame
    correctly, then gets stuck re-emitting one line ('TON DONG A' on
    finish-arch-partial/clear-1) until it hits max_output_tokens. Collapsing
    consecutive duplicate lines recovers the correct leading content instead
    of throwing the whole response away.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    collapsed: list[str] = []
    for line in lines:
        if not collapsed or collapsed[-1] != line:
            collapsed.append(line)
    return " ".join(collapsed)


_gemini_client = None


def _gemini() -> "genai.Client":  # noqa: F821 - type-checked only when imported
    global _gemini_client
    if _gemini_client is None:
        import os

        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def read_gemini(
    image: np.ndarray, *, ignore_regions=None, model_id: str = "gemini-3.1-flash-lite",
) -> str:
    """Read one frame with Gemini as a VLM reader.

    Calls the API directly rather than reusing
    ``scripts.captioning.caption_shots_gemini.generate`` -- that helper hard-fails on a
    MAX_TOKENS response, which is right for captioning (a truncated caption
    is just broken) but wrong here: a repetition-loop truncation still has
    the correct reading in its untruncated prefix, recoverable by
    :func:`_collapse_repeated_lines`.
    """
    import time

    import cv2
    from google.genai import types

    from scripts.audio.gemini_clean import _is_retryable, _retry_delay_seconds

    masked = _mask_ignore_regions(image, ignore_regions)
    ok, jpeg = cv2.imencode(".jpg", masked, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("cannot encode frame for Gemini")
    contents = [
        types.Part.from_bytes(data=jpeg.tobytes(), mime_type="image/jpeg"),
        READ_PROMPT_GEMINI,
    ]
    client = _gemini()
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config={
                    "temperature": 0.0,
                    "thinking_config": {"thinking_level": "minimal"},
                    "media_resolution": "MEDIA_RESOLUTION_HIGH",
                    "max_output_tokens": 2048,
                },
            )
            break
        except Exception as error:
            if attempt == 4 or not _is_retryable(error):
                raise
            delay = _retry_delay_seconds(error)
            time.sleep(delay if delay is not None else 2 ** attempt)
    return _collapse_repeated_lines(response.text or "")


# --- PaddleOCR (multilingual, not routed through VietOCR) -------------------

_paddle_engine = None


def _paddle():
    global _paddle_engine
    if _paddle_engine is None:
        from paddleocr import PaddleOCR

        _paddle_engine = PaddleOCR(lang="vi", use_angle_cls=True, show_log=False)
    return _paddle_engine


def read_paddleocr(image: np.ndarray, *, ignore_regions=None) -> str:
    from scripts.ocr.postprocess import Line, reading_order, render_text

    masked = _mask_ignore_regions(image, ignore_regions)
    result = _paddle().ocr(masked, cls=True)
    lines = []
    for page in result or []:
        for box, (text, _conf) in page or []:
            text = (text or "").strip()
            if text:
                lines.append(Line(text=text, box=tuple((float(x), float(y)) for x, y in box)))
    return render_text(reading_order(lines))


# --- EasyOCR ------------------------------------------------------------

_easyocr_reader = None


def _easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        _easyocr_reader = easyocr.Reader(["vi"], gpu=True)
    return _easyocr_reader


def read_easyocr(image: np.ndarray, *, ignore_regions=None) -> str:
    from scripts.ocr.postprocess import Line, reading_order, render_text

    masked = _mask_ignore_regions(image, ignore_regions)
    results = _easyocr().readtext(masked)
    lines = [
        Line(text=text.strip(), box=tuple((float(x), float(y)) for x, y in box))
        for box, text, _conf in results
        if text and text.strip()
    ]
    return render_text(reading_order(lines))


# --- Tesseract ----------------------------------------------------------

def read_tesseract(image: np.ndarray, *, ignore_regions=None) -> str:
    """CPU baseline. Uses Tesseract's own page-segmentation, not our reading_order."""
    import cv2
    import pytesseract

    masked = _mask_ignore_regions(image, ignore_regions)
    rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb, lang="vie")
    return " ".join(text.split())


#: Name -> adapter, for scripts.benchmarks.bench_ocr's grid.
ENGINES = {
    "gemini": read_gemini,
    "paddleocr": read_paddleocr,
    "easyocr": read_easyocr,
    "tesseract": read_tesseract,
}
