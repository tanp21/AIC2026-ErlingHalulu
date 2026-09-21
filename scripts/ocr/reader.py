"""A second, independent reader: a Vietnamese vision-LLM shown cropped text blocks.

This exists because the main pipeline and this model fail in complementary ways,
which the bake-off measured directly.

The main pipeline is a line recogniser plus a corrector that is forbidden to
invent text. That is the right design — it is what keeps hallucinated Vietnamese
out of the index — but it means a line the recogniser cannot read is a line that
is simply lost. On the reference signboard the tagline under the banner is
detected, then collapses in the recogniser into a repetition loop scoring 0.26,
below the confidence floor. Nothing downstream can recover it: the corrector has
no draft line to repair, and admitting the collapsed output instead pushed that
frame's character error rate from 0.62 to 1.55.

``5CD-AI/Vintern-1B-v3_5`` reads it. The model is InternVL2.5-1B fine-tuned for
Vietnamese OCR and handwriting, and it is used here in the only role it is good
at. Two measurements shaped that choice:

* **As a corrector it does nothing.** Given a draft to repair it returns the
  draft verbatim — four generation settings produced byte-identical output on
  the recipe card — and with beam search it echoes the instructions into its
  answer. A 1B model handed a long instruction with text embedded in it copies
  rather than reasons.
* **As a reader of a whole frame it stops early.** On the busy news frames it
  reads one region, usually the crawl, and quits: 0/7 on the flood gauge, 0/2 on
  the signboard.

Given *one cropped block at a time* it is accurate and quick: the gauge column in
1.3 s at CER 0.000, and the signboard banner including the tagline at 0.237 where
the main pipeline scores 0.618. Its output is merged by
:func:`scripts.ocr.postcorrect.absorb_reader`, which can add coverage but never
overwrite the main pipeline — necessary, because on that same banner it misreads
the place name ``Mộc Châu`` as ``Mai Chiu``.
"""
from __future__ import annotations

import cv2
import numpy as np

from scripts.ocr.engines import cluster_boxes, polygon_bbox

# Re-exported: the tiling rule now lives in `agent/vlm_models.py`, which is the
# copy the production image carries (scripts/ocr is not in it). The OCR tests
# import it from here, and keeping one definition avoids the two drifting apart.
from agent.vlm_models import tile_image  # noqa: F401  (re-export)

#: Short, in Vietnamese, and purely an instruction — no embedded text. Long
#: prompts are what this model echoes back instead of following.
READ_PROMPT = (
    "<image>\nĐọc chính xác toàn bộ chữ trong ảnh, giữ nguyên dấu tiếng Việt. "
    "Chỉ in ra chữ đọc được, không giải thích."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def text_blocks(
    image: np.ndarray,
    boxes,
    *,
    gap_ratio: float = 1.5,
    margin_ratio: float = 0.35,
    target_height: int = 64,
    max_side: int = 1600,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Group detections into blocks and return each as ``(bbox, cropped image)``.

    Blocks rather than lines: a single line stripped of its surroundings gives the
    model nothing to reason with, and magnifying one word made it *worse* in
    testing — 4x lost the tone marks on the reference signboard, 8x returned
    ``Mac Clean``. A block keeps neighbouring lines for context while excluding
    the rest of the frame, which is what stops the model reading the news crawl
    and stopping there.

    Each block is scaled so its own median line is about ``target_height`` pixels,
    so a 10px document and a 50px banner both arrive at a size the model handles,
    subject to ``max_side`` as a cost ceiling.
    """
    bounds = [polygon_bbox(box) for box in boxes]
    if not bounds:
        return []
    heights = sorted(b[3] - b[1] for b in bounds)
    unit = heights[len(heights) // 2] or 1.0

    blocks = []
    for cluster in cluster_boxes(bounds, gap_ratio * unit):
        margin = margin_ratio * unit
        x0 = int(max(0, min(c[0] for c in cluster) - margin))
        y0 = int(max(0, min(c[1] for c in cluster) - margin))
        x1 = int(min(image.shape[1], max(c[2] for c in cluster) + margin))
        y1 = int(min(image.shape[0], max(c[3] for c in cluster) + margin))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        local = sorted(c[3] - c[1] for c in cluster)
        scale = min(
            target_height / (local[len(local) // 2] or 1.0),
            max_side / max(crop.shape[0], crop.shape[1]),
        )
        if scale > 1.05:
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        blocks.append(((x0, y0, x1, y1), crop))
    return blocks


class VinternBlockReader:
    """Reads cropped text blocks with a Vietnamese vision-LLM. Loaded on first use."""

    def __init__(
        self,
        model_name: str = "5CD-AI/Vintern-1B-v3_5",
        *,
        device: str = "cuda:0",
        max_tiles: int = 6,
        max_new_tokens: int = 512,
        max_blocks: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.max_tiles = max_tiles
        self.max_new_tokens = max_new_tokens
        self.max_blocks = max_blocks
        self._model = None
        self._tokenizer = None
        self._transform = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # The checkpoint ships remote code written against transformers 4.x, which
        # does not define `all_tied_weights_keys`; 5.x reads it during loading.
        # Resolving the class first and supplying the default is cheaper than
        # pinning transformers back, since the corrector needs 5.x.
        model_cls = get_class_from_dynamic_module(
            "modeling_internvl_chat.InternVLChatModel", self.model_name
        )
        if not hasattr(model_cls, "all_tied_weights_keys"):
            model_cls.all_tied_weights_keys = {}
        self._model = model_cls.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval().to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        self._transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def read_crop(self, crop: np.ndarray) -> str:
        """Transcribe one BGR crop."""
        import torch
        from PIL import Image

        self._load()
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixels = torch.stack(
            [self._transform(tile) for tile in tile_image(pil, self.max_tiles)]
        ).to(torch.bfloat16).to(self.device)
        # Greedy, no repetition penalty: beam search is what makes this model
        # echo its own instructions back, and the penalty distorts the repeated
        # tokens that legitimately occur in ingredient lists.
        config = dict(
            max_new_tokens=self.max_new_tokens, do_sample=False,
            num_beams=1, repetition_penalty=1.0,
        )
        answer, _ = self._model.chat(
            self._tokenizer, pixels, READ_PROMPT, config,
            history=None, return_history=True,
        )
        return " ".join(answer.split())

    def read(self, image: np.ndarray, boxes, **block_kwargs) -> str:
        """Transcribe every text block of a frame, in reading order."""
        from scripts.ocr.postprocess import Line, reading_order, render_text

        blocks = text_blocks(image, boxes, **block_kwargs)
        if not blocks:
            return ""
        # Largest blocks first, so a per-frame budget spends itself on the text
        # that matters rather than on whatever happened to be detected first.
        blocks.sort(key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]), reverse=True)
        lines = []
        for (x0, y0, x1, y1), crop in blocks[: self.max_blocks]:
            text = self.read_crop(crop)
            if text:
                lines.append(Line(
                    text=text,
                    box=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                ))
        return render_text(reading_order(lines))
