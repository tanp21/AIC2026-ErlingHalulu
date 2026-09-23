"""Local vision-language backends: the models, and how they are loaded.

The agent used to send every keyframe it wanted judged to a remote API as inline
JPEG, which is the heaviest thing it puts on the wire. At competition time the
venue's uplink is the bottleneck, so this module exists to let the five jobs in
:mod:`agent.vision` run against a model on the same machine instead.

Three models are supported and none is chosen by this module — the choice comes
from ``scripts.benchmarks.bench_agent_vision`` measured on the agent's own ground
truth, because the evidence that exists for any of them comes from a different
task. What there is, and what it is worth:

* ``scripts/ocr/reader.py`` measured ``Vintern-1B-v3_5`` reading *one region* of a
  whole frame and quitting (0/7 on the flood gauge, 0/2 on the signboard). Same
  input shape as a verify call, different question — that was transcribing text,
  not judging a scene — so it is a warning, not a verdict. The same page records
  the model doing 0.000 CER on that gauge *when handed a crop*.
* Its instruction-following is thin: given a long prompt with text embedded in it
  it copies rather than reasons, which is why the local path prompts it differently
  (see ``COMPACT_PROMPTS``).
* Its language backbone is ``Qwen2-0.5B``, which bounds how well it can hold a
  0-10 scale and a reason together.

None of that is a measurement of this job, and this module does not pretend
otherwise. Two things are deliberately *not* here: the models' weights (they must
be downloaded ahead of time, see :func:`snapshot`) and any decision about which one
wins.

The snapshots are materialised as real files rather than left in the Hugging Face
blob cache. The default cache stores each file as a symlink into ``blobs/``, and
``InternVL``'s remote code resolves its own modules relative to ``__file__``,
which follows that symlink somewhere the module is not. The container also runs
read-only with ``HF_HUB_OFFLINE=1``, so the download has to happen on the dev
machine and be mounted in — same arrangement as ``scripts/audio/capu.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading
import time

from .llm import LLMError, RETRYABLE_STATUS, parse_json

def vlm_dir() -> Path:
    """Where the snapshots are materialised; mounted read-only in the container.

    Read per call rather than at import: the agent's configuration comes from a
    merged environment, so an `AIC_VLM_DIR` set only in `.env` never reaches a
    module-level constant, and `backend/agent_runner.py` forwards the variable into
    the child environment for exactly this to pick up.
    """
    return Path(os.environ.get("AIC_VLM_DIR", Path.home() / ".cache" / "aic" / "vlm"))


@dataclass(frozen=True)
class VlmSpec:
    """One candidate model, with the knobs its own processor needs.

    ``family`` selects the loading and prompting path — the three families do not
    share a chat template, a preprocessing helper, or a way of emitting JSON.
    """

    name: str
    family: str
    #: bfloat16 where the checkpoint was trained in it, float16 otherwise.
    dtype: str
    note: str = ""

    @property
    def slug(self) -> str:
        return self.name.replace("/", "--")

    @property
    def snapshot_dir(self) -> Path:
        return vlm_dir() / self.slug


#: The candidates the benchmark picks between. Order is presentation only.
CANDIDATES: tuple[VlmSpec, ...] = (
    VlmSpec(
        "5CD-AI/Vintern-1B-v3_5",
        family="internvl",
        dtype="bfloat16",
        note="Vietnamese-tuned InternVL2.5-1B; Qwen2-0.5B backbone; remote code",
    ),
    VlmSpec(
        "Qwen/Qwen3-VL-2B-Instruct",
        family="qwen3vl",
        dtype="bfloat16",
        note="newest Qwen VLM at 2B",
    ),
    VlmSpec(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        family="qwen2_5vl",
        dtype="float16",
        note="3B, strongest general scene understanding of the three",
    ),
)

#: Full-file alternates that every checkpoint ships alongside its safetensors and
#: that nothing here loads. Skipped so the mount stays small.
_SKIP = ("*.pth", "*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.gguf")


def spec(name: str) -> VlmSpec:
    """Look a candidate up by hub id or by short slug."""
    for candidate in CANDIDATES:
        if name in (candidate.name, candidate.slug):
            return candidate
    raise KeyError(f"unknown VLM {name!r}; known: {[c.name for c in CANDIDATES]}")


def snapshot(candidate: VlmSpec | str, *, revision: str | None = None) -> str:
    """Download ``candidate`` as real files and return the directory.

    Run this with the network up. It is a no-op once the files are there.
    """
    from huggingface_hub import snapshot_download

    if isinstance(candidate, str):
        candidate = spec(candidate)
    target = candidate.snapshot_dir
    target.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        candidate.name,
        revision=revision,
        local_dir=str(target),
        ignore_patterns=list(_SKIP),
    )


def missing() -> list[VlmSpec]:
    """Candidates whose snapshot is absent or does not look like a model."""
    absent = []
    for candidate in CANDIDATES:
        directory = candidate.snapshot_dir
        has_config = (directory / "config.json").is_file()
        has_weights = any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))
        if not (has_config and has_weights):
            absent.append(candidate)
    return absent


def pick_device() -> str:
    """Where to put the model. Deliberately its own variable.

    ``AUDIO_EMBED_DEVICE`` exists to move the *audio* embedder to CPU under
    memory pressure; this is a separate model and wants a separate switch.
    """
    return os.environ.get("AIC_VLM_DEVICE", "cuda" if _cuda() else "cpu")


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# --- output formatting -------------------------------------------------------
#
# The prompts in `agent.prompts` never state the output shape. They rely on
# Gemini's `response_json_schema` to carry it, so a local model handed the same
# text has nothing telling it to answer `{"candidates": [...]}` at all. The
# schema has to be rendered back into words for every backend except Gemini.


def format_instruction(schema: dict) -> str:
    """Render a response schema as a short imperative output-format block.

    Small models copy an example far more reliably than they honour a spec, so
    the shape is shown literally and every required key is named — including the
    ones *inside* an array, which is where the agent's schemas keep them.
    """
    lines = ["Reply with JSON only. No prose, no markdown fence."]
    example, _ = _shape(schema)
    lines.append(f"Shape: {example}")
    for path, keys in _required_paths(schema):
        where = f'each "{path}"' if path else "the object"
        lines.append(f"Give every part its keys — {where} must carry: {', '.join(keys)}.")
    return "\n".join(lines)


def _required_paths(schema, prefix: str = "") -> list[tuple[str, list[str]]]:
    """Every level that declares required keys, as ``(dotted path, keys)``."""
    found: list[tuple[str, list[str]]] = []
    if not isinstance(schema, dict):
        return found
    properties = schema.get("properties")
    if properties:
        required = [name for name in schema.get("required", []) if name in properties]
        if required:
            found.append((prefix, required))
        for name, sub in properties.items():
            found.extend(_required_paths(sub, f"{prefix}.{name}".strip(".")))
    items = schema.get("items")
    if isinstance(items, dict):
        found.extend(_required_paths(items, prefix))
    return found


def _shape(schema: dict, depth: int = 0) -> tuple[str, list[str]]:
    """A one-line example of ``schema``, plus the required keys at its top level."""
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = [name for name in schema.get("required", []) if name in properties]
        parts = []
        for name, sub in properties.items():
            rendered, _ = _shape(sub, depth + 1)
            parts.append(f'"{name}": {rendered}')
        body = ", ".join(parts) or "…"
        return "{" + body + "}", required
    if kind == "array":
        item, _ = _shape(schema.get("items", {}), depth + 1)
        return f"[{item}, …]", []
    if kind == "integer" or kind == "number":
        low, high = schema.get("minimum"), schema.get("maximum")
        if low is not None and high is not None:
            return f"<number {low}-{high}>", []
        return "<number>", []
    if kind == "string":
        allowed = schema.get("enum")
        if allowed:
            return "<" + "|".join(str(option) for option in allowed) + ">", []
        return "<text>", []
    if kind == "boolean":
        return "<true|false>", []
    return "<value>", []


#: Jobs that score every image independently. `verify` and `sequence` ask the same
#: question of each frame; the others reason *across* frames (which one is first,
#: what contradicts the query, what does the answer say) and cannot be split.
PER_IMAGE_JOBS = frozenset({"verify", "sequence"})

#: Short prompts for the local backends, one image per call, per language.
#:
#: The prompts in `agent.prompts` are long English instructions because Gemini
#: follows them. `scripts/ocr/reader.py` measured what that does to a 1B model:
#: handed a long instruction with text embedded in it, it *copies instead of
#: reasoning* — asked to score five frames it invents the ones it cannot see and
#: runs past its token budget. A short, purely instructional prompt is what it can
#: actually follow.
#:
#: Vietanamese is the default because the query is Vietnamese and `5CD-AI/Vintern-1B-v3_5`
#: is tuned for it — with an English prompt it answers in English, with a
#: Vietnamese one it answers in Vietnamese, so it takes its cue from the prompt.
#: Whether that also helps the Qwen models is a measurement, not an assumption:
#: `AIC_VLM_PROMPT=en` switches this, and `bench_agent_vision` scores either way.
#:
#: Neither asks for a free-text `reason`. It is only ever logged —
#: `CandidatePool.ordered` ranks on `score` and `scene` alone — and a free-text
#: explanation is where a small model spends its whole token budget, gets truncated
#: mid-JSON, and turns a good score into an unparsable reply.
#:
#: `scene` and `answer` *are* asked for, but only when something reads them:
#: `pool.py` adds its scene bonus only when `verdict.scene is not None` and only
#: across more than one scene, and `_confident` needs `verdict.answer` to stop a Q&A
#: run early. Asking for them unconditionally would cost tokens on every call to
#: serve the minority that need them.
COMPACT_PROMPTS = {
    "vi": {
        "verify": (
            "Đánh giá khung hình này theo mô tả sau, cho điểm 0-10 "
            "(10 = khớp chính xác cảnh được mô tả, 0 = không liên quan).\n"
            "{scenes}Mô tả: {query}\n"
            'Chỉ trả lời JSON, không giải thích: {{"id": "{cid}", "score": <0-10>{extra}}}'
        ),
        "sequence": (
            "Khung hình này có phải là lúc BẮT ĐẦU của sự kiện sau không? "
            "Cho điểm 0-10 (10 = đúng ngay lúc bắt đầu, 5 = sự kiện đang diễn ra, 0 = cảnh khác).\n"
            "Sự kiện: {query}\n"
            'Chỉ trả lời JSON, không giải thích: {{"id": "{cid}", "score": <0-10>}}'
        ),
    },
    "en": {
        "verify": (
            "Rate this frame against the description, 0-10 "
            "(10 = exactly the scene described, 0 = unrelated).\n"
            "{scenes}Description: {query}\n"
            'Reply with JSON only, no explanation: {{"id": "{cid}", "score": <0-10>{extra}}}'
        ),
        "sequence": (
            "Is this frame the START of the event below? Rate 0-10 "
            "(10 = exactly the first moment, 5 = already under way, 0 = a different scene).\n"
            "Event: {query}\n"
            'Reply with JSON only, no explanation: {{"id": "{cid}", "score": <0-10>}}'
        ),
    },
}
DEFAULT_PROMPT_LANGUAGE = "vi"

#: Which language each family is prompted in.
#:
#: The queries are Vietnamese and `5CD-AI/Vintern-1B-v3_5` is Vietnamese-tuned — give
#: it an English prompt and it answers in English, so it takes its cue from the
#: prompt. The Qwen VLMs follow English instructions more crisply. This is a
#: starting point to re-measure, not a settled fact: `AIC_VLM_PROMPT` overrides it
#: for an ablation and `bench_agent_vision` scores either side.
FAMILY_LANGUAGE = {"internvl": "vi", "qwen3vl": "en", "qwen2_5vl": "en"}

#: The extra output fields, per language, when `scenes` or `question` make them
#: worth asking for. `{index}` is the scene number to report.
COMPACT_EXTRAS = {
    "vi": {
        "scene": ', "scene": <số thứ tự cảnh khớp nhất>',
        "answer": ', "answer": "<câu trả lời ngắn đọc được trong ảnh, hoặc rỗng>"',
    },
    "en": {
        "scene": ', "scene": <the matching scene number>',
        "answer": ', "answer": "<the short answer readable in the frame, or empty>"',
    },
}


def compact_prompt(job: str, query: str, cid: str, *, language=None, scenes="",
                   question="", limit: int = 400) -> str | None:
    """A one-image prompt for `job`, or None to use the caller's long one.

    `scenes` and `question` are the caller's own renderings. They decide whether
    `scene` and `answer` are asked for, and they must be the *per-frame* inputs:
    `sequence` scores each frame against its own event, so handing it the joined
    multi-event query would ask whether one frame starts all of them at once.
    """
    language = language or DEFAULT_PROMPT_LANGUAGE
    table = COMPACT_PROMPTS.get(language, COMPACT_PROMPTS[DEFAULT_PROMPT_LANGUAGE])
    extras = COMPACT_EXTRAS.get(language, COMPACT_EXTRAS[DEFAULT_PROMPT_LANGUAGE])
    template = table.get(job)
    if not template or not query:
        return None
    extra = ""
    if scenes:
        extra += extras["scene"]
    if question:
        extra += extras["answer"]
    return template.format(query=query[:limit], cid=cid, scenes=scenes, extra=extra)


def single_from(parsed, field: str, wanted: str) -> dict | None:
    """The one item a single-image question asked for, or None if it did not answer it.

    Handles the two shapes a small model actually produces when asked about one
    image: the bare object, and the array it was told not to emit — in which case
    the element naming `wanted` is the answer and anything else is not.
    """
    if not isinstance(parsed, dict):
        return None
    if field not in parsed:
        return parsed
    rows = parsed.get(field)
    if not isinstance(rows, list) or not rows:
        return None
    if len(rows) == 1:
        return rows[0] if isinstance(rows[0], dict) else None
    for row in rows:
        if isinstance(row, dict) and str(row.get("id", "")).strip("[] ") == wanted:
            return row
    return None


def array_field(schema: dict) -> tuple[str, dict] | None:
    """The single "list of objects carrying an id" a schema asks for, if it has one."""
    for name, sub in (schema.get("properties") or {}).items():
        items = sub.get("items") if isinstance(sub, dict) else None
        if isinstance(items, dict) and "id" in (items.get("properties") or {}):
            return name, items
    return None


class VLMError(LLMError):
    """Deliberately an ``LLMError``.

    ``Inspector.pick_frame`` catches ``LLMError`` specifically and ``_lookup``
    catches broadly; a local backend that raised something else would turn a
    degraded job into a crash.
    """


def label_ids(labels) -> list[str]:
    """The bracketed id at the start of each label: ``"[C1] L21_V024 t=…"`` -> ``"C1"``."""
    return [cid for cid, _ in labelled_ids(labels)]


def labelled_ids(labels) -> list[tuple[str, str]]:
    """``(id, label)`` for each label that carries one, keeping the pairing intact."""
    found = []
    for label, _ in labels:
        match = re.match(r"\s*\[([^\]\s]+)\]", label or "")
        if match:
            found.append((match.group(1), label or ""))
    return found


def align_ids(reply: dict, schema: dict, labels) -> dict:
    """Map the ids a small model echoed loosely back onto the ids it was sent.

    Asked to answer about ``[C1]``, a 1-3B model commonly replies ``"C1"`` at best
    and copies the *rest* of the label — ``"L21_V024 t=810.6s"`` — at worst. The
    callers match strictly (`Inspector.verify` drops any id it does not know), so
    without this a model that is otherwise answering correctly scores nothing at
    all, and does it silently.

    Also drops duplicates and unknown ids, which the same models produce freely.
    """
    pairs = labelled_ids(labels)
    if not pairs or not isinstance(reply, dict):
        return reply
    for field, sub in (schema.get("properties") or {}).items():
        items = sub.get("items") if isinstance(sub, dict) else None
        if not isinstance(items, dict) or "id" not in (items.get("properties") or {}):
            continue
        rows = reply.get(field)
        if not isinstance(rows, list):
            continue
        seen, cleaned = set(), []
        for row in rows:
            if not isinstance(row, dict):
                continue
            resolved = _resolve_id(str(row.get("id", "")), pairs)
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            cleaned.append({**row, "id": resolved})
        reply[field] = cleaned
    return reply


def _resolve_id(raw: str, pairs: list[tuple[str, str]]) -> str | None:
    """Recover the intended id, preferring the least clever reading that works."""
    token = raw.strip().strip("[]").strip()
    known = [cid for cid, _ in pairs]
    if token in known:
        return token
    # The model numbered its own list instead of using ours.
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(known):
            return known[index]
    lowered = token.lower()
    # It echoed the label body. Prefer the label it is actually a substring of, and
    # only accept an unambiguous match — guessing between two candidates is worse
    # than leaving the candidate unverified, which every caller already handles.
    hits = [cid for cid, label in pairs if lowered and lowered in label.lower()]
    if len(hits) == 1:
        return hits[0]
    for cid, label in pairs:
        if f"[{cid.lower()}]" in f" {lowered} ":
            return cid
    return None


# --- backends ----------------------------------------------------------------


class LocalVLM:
    """Runs the model in this process.

    One generation at a time, behind a lock: ``Inspector.verify`` submits up to
    four chunks from a thread pool, and four concurrent forwards on one 12 GB
    card is how you meet the OOM playbook. Serialising costs little — there is no
    network round trip to hide, which is the opposite of the trade-off
    ``AgentOptions`` was tuned for against Gemini.
    """

    def __init__(self, candidate=None, *, backend=None, device=None,
                 image_side=448, answer_side=1280, max_new_tokens=128,
                 max_tiles=6, usage=None, prompt_language=None):
        self.spec = candidate if candidate is not None else default_spec()
        self._backend = backend
        self.device = device or pick_device()
        self.image_side = image_side
        self.answer_side = answer_side
        self.prompt_language = (
            prompt_language
            or os.environ.get("AIC_VLM_PROMPT")
            or FAMILY_LANGUAGE.get(self.spec.family, DEFAULT_PROMPT_LANGUAGE)
        )
        # A one-object reply is ~40 tokens. The cap is what keeps a model that
        # starts rambling from spending the whole time budget, and a reply cut off
        # mid-JSON parses as nothing at all.
        self.max_new_tokens = max_new_tokens
        self.max_tiles = max_tiles
        self.usage = usage
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._disabled = False

    def _model(self):
        if self._backend is None:
            with self._load_lock:
                if self._backend is None:
                    self._backend = loaded_model(
                        self.spec, device=self.device, max_tiles=self.max_tiles
                    )
        return self._backend

    def judge(self, images, prompt, schema, *, max_output_tokens=8192, job=None,
              query=None, scenes="", question=None, salvage=False):
        """Same contract as ``GeminiClient.judge``: labelled images + prompt -> JSON.

        `salvage` is accepted and ignored: a reply cut off by a token limit is a
        DeepSeek endpoint quirk (see `DeepSeekClient.judge`), not something this
        backend can produce.

        For the per-image jobs this fans out into one call per frame and reassembles
        the array the caller expects. A 1-3B model handed five frames and a
        schema saying "a list of five objects" invents the ones it cannot see and
        runs past its token budget doing it; asked about one frame at a time it
        answers, and the ids stay attached to the right images because there is
        only ever one.

        `query` is the original Vietnamese question, and for `sequence` a *list* of
        them, one per frame — each frame is scored against its own event, so a
        joined multi-event string would ask whether one frame starts them all. The
        long English prompt the caller built is for Gemini; the local path rebuilds
        a short one from these.
        """
        if self._disabled:
            raise VLMError("local vision backend disabled after a failure")
        if not images:
            raise VLMError("no images to judge")
        from . import vlm_models

        pictures = []
        for label, data in images:
            image = vlm_models.to_pil(data)
            side = self.answer_side if job == "answer" else self.image_side
            pictures.append((label, vlm_models.resize(image, side)))
        split = array_field(schema)
        if split is not None and job in PER_IMAGE_JOBS:
            name, item = split
            rows: list[dict] = []
            failures = []
            for index, (label, image) in enumerate(pictures):
                cid = next(iter(label_ids([(label, image)])), "")
                one_query = query[index] if isinstance(query, (list, tuple)) else query
                short = compact_prompt(job, one_query or "", cid, scenes=scenes,
                                       question=question or "",
                                       language=self.prompt_language)
                try:
                    answered = self._decide(
                        [(label, image)], short or prompt, item, name, single=bool(short)
                    )
                except VLMError as exc:
                    # One frame the model could not answer costs that candidate its
                    # verdict and nothing else. A candidate with no verdict is already
                    # a state the pool handles; a whole chunk lost to one bad frame is
                    # the failure mode worth avoiding.
                    failures.append(exc)
                    continue
                for row in answered:
                    # The id is stamped, not read: this call was about exactly one
                    # image and we already know which. A small model asked to echo
                    # "[C1]" answers "L21_V024 t=810.6s" often enough that trusting
                    # it would silently unscore the candidate.
                    rows.append({**row, "id": cid})
            if not rows and failures:
                raise failures[0]
            return {name: rows}
        return self._decide(pictures, prompt, schema, None)

    def _decide(self, pictures, prompt, schema, field, *, single=False,
                max_output_tokens=8192):
        """One generation, one parse, one repair retry. Raises `VLMError` on failure."""
        ask = prompt
        if field is not None and not single:
            # A single-object question: name the one id and drop the array wrapper,
            # which is the part small models get wrong.
            cid = next(iter(label_ids(pictures)), "?")
            ask = (
                f"{prompt}\n\nReply with JSON only, exactly one object — not a list — "
                f"describing the single image labelled [{cid}].\n"
                + format_instruction(schema)
            )
        elif field is None:
            ask = prompt + "\n\n" + format_instruction(schema)
        tokens = min(max_output_tokens, self.max_new_tokens)
        wanted = next(iter(label_ids(pictures)), "") if field is not None else ""
        reply = self._generate(pictures, ask, tokens)
        parsed = self._parse(reply)
        if parsed is not None:
            if field is None:
                return align_ids(parsed, schema, pictures)
            chosen = single_from(parsed, field, wanted)
            if chosen is not None:
                return [chosen]
        repair = ask + "\n\nYour previous reply was not valid JSON. Reply with the JSON object only."
        reply = self._generate(pictures, repair, tokens)
        parsed = self._parse(reply)
        if parsed is None:
            raise VLMError(f"local VLM returned unparsable JSON: {(reply or '')[:200]}")
        if field is None:
            return align_ids(parsed, schema, pictures)
        chosen = single_from(parsed, field, wanted)
        if chosen is None:
            raise VLMError(f"local VLM answered about the wrong image: {(reply or '')[:200]}")
        return [chosen]

    @staticmethod
    def _parse(reply: str):
        try:
            return parse_json(reply)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _generate(self, pictures, question, tokens) -> str:
        import torch

        try:
            model = self._model()
        except Exception as exc:
            # A missing snapshot, an offline hub or a bad dependency all land here;
            # callers only know VLMError, and `pick_frame` catches LLMError
            # specifically, so anything else would turn a degraded job into a crash.
            raise VLMError(f"local VLM could not load: {exc}") from exc
        with self._lock:
            try:
                answer = model.generate(pictures, question, tokens)
            except torch.cuda.OutOfMemoryError as first:
                torch.cuda.empty_cache()
                smaller = pictures[: max(1, len(pictures) // 2)]
                if len(smaller) == len(pictures):
                    self._disabled = True
                    raise VLMError("local VLM ran out of GPU memory") from first
                try:
                    answer = model.generate(smaller, question, tokens)
                except torch.cuda.OutOfMemoryError as second:
                    # Halving once is a degradation; failing twice is a backend that
                    # will keep competing with the server's own models for the card.
                    self._disabled = True
                    torch.cuda.empty_cache()
                    raise VLMError("local VLM ran out of GPU memory") from second
            except Exception as exc:
                raise VLMError(f"local VLM failed: {exc}") from exc
        if self.usage is not None:
            self.usage.add(vlm_calls=1)
        return answer


class HttpVLM:
    """Forwards every judge call to a VLM served elsewhere (``AIC_VLM_URL``).

    Same shape as ``backend/search_core``: unset the URL and the model runs in
    this process; set it and the work goes to a host that has the GPU budget.
    """

    def __init__(self, base_url, *, model=None, timeout=60.0, retries=2, usage=None,
                 http=None, sleep=time.sleep, max_body_mb=3,
                 trusted_user=None, user=None, password=None):
        """`max_body_mb` sits under the server's `MAX_CORE_BYTES` (4 MiB, see
        `backend/access_control.py`), so an oversized request is refused here rather
        than travelling to the ingress to be rejected.

        The identity and credentials mirror `SearchClient`: `/core/*` is behind the
        same middleware as every other route, so a forwarded judge call without
        `X-AIC-User` is answered 401 and every vision job degrades to unverified.
        """
        import httpx

        self.model = model
        self.usage = usage
        self._retries = retries
        self._sleep = sleep
        self._max_body = max_body_mb * 1024 * 1024
        headers = {"X-AIC-User": trusted_user} if trusted_user else {}
        auth = (user, password) if user and password else None
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers, auth=auth,
        )

    def judge(self, images, prompt, schema, *, max_output_tokens=8192, job=None,
              query=None, scenes="", question=None, salvage=False):
        """Forwarded to the host running the model, which ignores `salvage` like the local
        backend does -- the flag only means something to a backend that can be cut off."""
        import base64

        payload = {
            "model": self.model, "job": job, "prompt": prompt, "schema": schema,
            "max_output_tokens": max_output_tokens, "scenes": scenes, "question": question,
            # A list for `sequence`, a string for the others; the remote end rebuilds
            # its prompt from whichever it gets.
            "query": list(query) if isinstance(query, (list, tuple)) else query,
            "images": [{"label": label, "b64": base64.b64encode(data).decode()}
                       for label, data in images],
        }
        body = json.dumps(payload)
        if len(body) > self._max_body:
            # The `answer` job carries a full-resolution frame; refuse rather
            # than pushing megabytes through the ingress.
            raise VLMError(f"vision request too large ({len(body)} bytes)")
        last = None
        for attempt in range(self._retries + 1):
            try:
                response = self._http.post("/core/vision/judge", content=body,
                                           headers={"content-type": "application/json"})
            except Exception as exc:
                last = exc
                if attempt == self._retries:
                    raise VLMError(f"vision service unreachable: {exc}") from exc
                self._sleep(2 ** attempt)
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self._retries:
                self._sleep(2 ** attempt)
                continue
            if response.status_code >= 400:
                raise VLMError(f"vision service: HTTP {response.status_code} {response.text[:200]}")
            try:
                data = response.json()
            except ValueError as exc:
                # A 200 with an error page is a real failure mode with HAProxy in
                # front, and JSONDecodeError is not a VLMError.
                raise VLMError(
                    f"vision service returned non-JSON: {response.text[:200]}") from exc
            if not isinstance(data, dict):
                raise VLMError(f"vision service returned {type(data).__name__}, not an object")
            if self.usage is not None:
                self.usage.add(vlm_calls=1)
            return data.get("reply") or {}
        raise VLMError(f"vision service: {last}")


def default_spec() -> VlmSpec:
    """The model to load when the deployment does not name one.

    `AIC_VLM_MODEL` is the deployment's choice, made from
    `benchmarks/visual/vlm_verify_2026.json` and the table in `docs/agent.md`. The
    fallback is the smallest candidate — a first run on a card that may already be
    holding PE-Core and Qwen should not start by trying to fit the largest one.
    """
    chosen = os.environ.get("AIC_VLM_MODEL")
    return spec(chosen) if chosen else CANDIDATES[0]


_MODELS: dict = {}
_MODELS_LOCK = threading.Lock()


def loaded_model(candidate: VlmSpec, *, device: str, max_tiles: int):
    """The checkpoint wrapper for `candidate`, built once and reused.

    A server-hosted run is one query, so building the model per run would re-read
    the snapshot every query and, with two runs allowed at once, hold two copies on
    one card. Only the *model* is shared: each `LocalVLM` keeps its own `usage` and
    its own out-of-memory `_disabled` flag. Weights still load on the first
    `generate`, so constructing this does no GPU work.
    """
    key = (candidate.name, device, max_tiles)
    with _MODELS_LOCK:
        if key not in _MODELS:
            from . import vlm_models

            _MODELS[key] = vlm_models.load_model(candidate, device=device, max_tiles=max_tiles)
        return _MODELS[key]


def build_vision(settings, *, usage=None, gemini=None):
    """Pick the backend named by ``AIC_AGENT_VISION``.

    ``gemini`` is the already-built client the caller tuned for this mode (fast
    mode shortens timeouts and hedges slow calls). It has to be reused rather than
    rebuilt here, or the Inspector would judge with an untuned client while the web
    lookup kept the tuned one.

    ``gemini`` stays the default until the benchmark says otherwise: it is the
    measured baseline, and a local model that loses to it should not be the
    thing the team relies on on competition day.
    """
    choice = (getattr(settings, "vision", None) or "gemini").lower()
    if choice in ("gemini", "deepseek"):
        if choice == "deepseek":
            # `deepseek-flash` reads images; `deepseek-v4-pro` does not, and answers a
            # vision call by emitting schema-shaped JSON with every score 0 — which
            # looks like a working judge returning poor results rather than a model
            # that cannot see. Measured: flash reads a solid-colour test image
            # correctly, v4-pro returns an empty `content`. The default is therefore
            # the one that works, whatever else is configured.
            from .llm import DeepSeekClient

            return DeepSeekClient(
                settings.deepseek_api_key,
                os.environ.get("AIC_VLM_MODEL") or "deepseek-flash",
                settings.deepseek_base_url, thinking=False, usage=usage,
            )
        if gemini is not None:
            return gemini
        if not getattr(settings, "gemini_api_key", None):
            raise ValueError("AIC_AGENT_VISION=gemini needs GEMINI_API_KEY")
        from .llm import GeminiClient

        return GeminiClient(
            settings.gemini_api_key, (getattr(settings, "gemini_model", None) or "gemini-3.8-flash"),
            thinking_level=getattr(settings, "gemini_thinking_level", "low"), usage=usage,
        )
    if choice != "local":
        raise ValueError(
            f"AIC_AGENT_VISION must be 'local', 'gemini' or 'deepseek', not {choice!r}")
    url = getattr(settings, "vlm_url", None)
    if url:
        return HttpVLM(url, model=getattr(settings, "vlm_model", None), usage=usage,
                       trusted_user=getattr(settings, "trusted_user", None),
                       user=getattr(settings, "server_user", None),
                       password=getattr(settings, "server_password", None))
    chosen = getattr(settings, "vlm_model", None)
    return LocalVLM(
        spec(chosen) if chosen else None,
        device=getattr(settings, "vlm_device", None),
        image_side=getattr(settings, "vlm_image_side", 448),
        answer_side=getattr(settings, "vlm_answer_side", 1280),
        max_new_tokens=getattr(settings, "vlm_max_new_tokens", 128),
        max_tiles=getattr(settings, "vlm_max_tiles", 6),
        usage=usage,
    )


