"""Loading and running the local vision-language models.

Three families, two loading paths. ``InternVL`` checkpoints ship their own remote
code and need ``trust_remote_code``; the Qwen ones are native to transformers 5.x
and go through ``AutoProcessor``. Everything heavy is imported inside
:func:`load_model` so this module stays cheap to import — ``agent/vlm.py`` is
pulled in by the agent, which runs in the same process as the server's own
models.

Nothing here decides *which* model is used. That comes from
``benchmarks/agent/vision_config.json``, written by
``scripts.benchmarks.bench_agent_vision``.
"""
from __future__ import annotations

from typing import Protocol

#: Pixel side the Qwen vision tower works at. 28 px patches with a 2x2 merge, so
#: an image costs roughly ``side**2 / 784`` visual tokens.
QWEN_PATCH = 28

#: Cap on visual tokens per image. The full-resolution frame the `answer` job
#: sends is 1920x1080; uncapped it would cost ~2650 visual tokens on its own.
DEFAULT_MAX_PIXELS = 1280 * (QWEN_PATCH ** 2)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tile_image(image, max_tiles: int = 6, size: int = 448, thumbnail: bool = True):
    """Split a PIL image into the aspect-matched 448px tile grid InternVL expects.

    Moved here from ``scripts/ocr/reader.py``, which now re-exports it: the OCR
    scripts are not in the production image but this module is.
    """
    width, height = image.size
    ratio = width / height
    grids = sorted(
        {(cols, rows)
         for total in range(1, max_tiles + 1)
         for cols in range(1, total + 1)
         for rows in range(1, total + 1)
         if cols * rows <= max_tiles},
        key=lambda g: g[0] * g[1],
    )
    area = width * height
    best, best_diff = (1, 1), float("inf")
    for grid in grids:
        diff = abs(ratio - grid[0] / grid[1])
        if diff < best_diff:
            best, best_diff = grid, diff
        elif diff == best_diff and grid[0] * grid[1] > best[0] * best[1]:
            # On a tie prefer more tiles only when there are pixels to justify
            # them. Without this a 448x448 crop is split into four tiles plus a
            # thumbnail — five times the compute for no extra detail.
            if area > 0.5 * size * size * max_tiles:
                best = grid
    cols, rows = best
    resized = image.resize((size * cols, size * rows))
    tiles = [
        resized.crop((
            (index % cols) * size, (index // cols) * size,
            (index % cols + 1) * size, (index // cols + 1) * size,
        ))
        for index in range(cols * rows)
    ]
    if thumbnail and len(tiles) != 1:
        tiles.append(image.resize((size, size)))
    return tiles


def dtype_kwargs(dtype: str) -> dict:
    """``dtype=`` on transformers 5.x, ``torch_dtype=`` before it.

    The checkpoint's own remote code is written against 4.x while the server runs
    5.x, so the version straddles both. Same branch as
    ``notebooks/kaggle_qwen3_caption_embed.py``.
    """
    import torch

    import transformers

    value = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype]
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": value} if major >= 5 else {"torch_dtype": value}


class VlmModel(Protocol):
    """What :class:`agent.vlm.LocalVLM` needs from a loaded model."""

    def generate(self, images: list, prompt: str, max_new_tokens: int) -> str: ...


def to_pil(data: bytes):
    """The Inspector carries JPEG bytes; every processor wants a PIL image."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def resize(image, side: int):
    """Downscale so the long edge is ``side``. Never upscales."""
    if max(image.size) <= side:
        return image
    scale = side / max(image.size)
    return image.resize((max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))))


class InternVLModel:
    """``InternVL`` checkpoints, including ``5CD-AI/Vintern-1B-v3_5``.

    The checkpoint's remote code predates transformers 5.x, which reads
    ``all_tied_weights_keys`` during loading; the class is resolved first so the
    attribute can be supplied. Without it loading raises, and pinning
    transformers back is not an option because the rest of the repo needs 5.x.
    """

    def __init__(self, model_id, *, device="cuda", dtype="bfloat16", max_tiles=6,
                 tile_side=448):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_tiles = max_tiles
        self.tile_side = tile_side
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

        model_cls = get_class_from_dynamic_module(
            "modeling_internvl_chat.InternVLChatModel", self.model_id
        )
        if not hasattr(model_cls, "all_tied_weights_keys"):
            model_cls.all_tied_weights_keys = {}
        self._model = model_cls.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,  # not installed, and the flag raises if it is missing
            **dtype_kwargs(self.dtype),
        ).eval().to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )
        self._transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((self.tile_side, self.tile_side), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def generate(self, images: list, prompt: str, max_new_tokens: int) -> str:
        """``images`` is ``[(label, PIL image)]``, label first — as Gemini gets it."""
        import torch

        self._load()
        pixel_values, patches, question = [], [], []
        for label, image in images:
            tiles = tile_image(image, self.max_tiles, self.tile_side)
            pixel_values.extend(self._transform(tile) for tile in tiles)
            patches.append(len(tiles))
            # `<image>` then the label, matching the order the prompt was written
            # against for Gemini.
            question.append("<image>\n" + label)
        question.append(prompt)
        config = dict(
            max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, repetition_penalty=1.0,
        )
        with torch.inference_mode():
            answer, _ = self._model.chat(
                self._tokenizer,
                torch.stack(pixel_values).to(torch.bfloat16).to(self.device),
                "\n".join(question),
                config,
                num_patches_list=patches,
                history=None,
                return_history=True,
            )
        return answer or ""


class QwenVLModel:
    """``Qwen3-VL`` and ``Qwen2.5-VL``, both native to transformers 5.x.

    One path for both: they share the message format and the processor API. No
    ``qwen_vl_utils`` — that helper only wraps ``apply_chat_template`` and would
    be a new production dependency for nothing.
    """

    def __init__(self, model_id, *, device="cuda", dtype="bfloat16",
                 max_pixels=DEFAULT_MAX_PIXELS, class_name=None):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_pixels = max_pixels
        self.class_name = class_name
        self._model = None
        self._processor = None

    def _model_class(self):
        import transformers

        name = self.class_name or "AutoModelForImageTextToText"
        return getattr(transformers, name)

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, max_pixels=self.max_pixels, trust_remote_code=False,
        )
        self._model = self._model_class().from_pretrained(
            self.model_id, **dtype_kwargs(self.dtype)
        ).eval().to(self.device)

    def generate(self, images: list, prompt: str, max_new_tokens: int) -> str:
        import torch

        self._load()
        content, pictures = [], []
        for label, image in images:
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": label})
            pictures.append(image)
        content.append({"type": "text", "text": prompt})
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=pictures, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        # Drop the prompt tokens; the reply is what follows them.
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def load_model(spec, *, device="cuda", max_tiles=6, tile_side=448,
               max_pixels=DEFAULT_MAX_PIXELS, path=None) -> VlmModel:
    """Build the right wrapper for ``spec``; weights load on first generate.

    A snapshot already on disk is used directly. Calling `snapshot_download` first
    would be a hub round trip on every load — and the container is offline, where
    there is nothing for it to ask.
    """
    from .vlm import snapshot

    if path is None:
        local = spec.snapshot_dir
        path = str(local) if local.is_dir() else snapshot(spec)
    model_id = str(path)
    if spec.family == "internvl":
        return InternVLModel(model_id, device=device, dtype=spec.dtype,
                             max_tiles=max_tiles, tile_side=tile_side)
    if spec.family == "qwen3vl":
        return QwenVLModel(model_id, device=device, dtype=spec.dtype,
                           max_pixels=max_pixels, class_name="Qwen3VLForConditionalGeneration")
    if spec.family == "qwen2_5vl":
        return QwenVLModel(model_id, device=device, dtype=spec.dtype,
                           max_pixels=max_pixels,
                           class_name="Qwen2_5_VLForConditionalGeneration")
    raise ValueError(f"no loader for family {spec.family!r}")


def free() -> None:
    """Release whatever the last model held. Called between benchmark models."""
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def peak_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None
