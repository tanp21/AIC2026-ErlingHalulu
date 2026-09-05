"""Optional vision-LLM post-correction of an OCR draft.

The pipeline in :mod:`scripts.ocr.engines` gets Vietnamese diacritics right most
of the time and reading order right nearly always, but it still loses individual
tone marks on stylised or very small text. A vision-LLM shown *both* the image
and the draft can repair those, because it has linguistic context the
line-by-line recogniser does not.

Two measured facts shape this module:

**Grounding is everything.** Asked to read the acceptance frames from scratch,
Qwen2.5-VL scored 0/7 on the flood gauge (it never resolves the digits) and 0/14
on the recipe card, and invented fluent Vietnamese that was not in the image
("kẹo hai rau củ", "Pha khỏi bếp"). Given the same images *plus* the draft, it
fixed real errors instead — ``HIVS`` to ``HTV9``, ``vàng từng phòng`` to ``vàng
tặc trong rừng phòng`` — taking the gauge frame from CER 0.123 to 0.025.

**It still sometimes rewrites rather than corrects.** On the densest frame the
first attempt dropped a third of the content (word recall 0.756 to 0.513). So
every correction passes :func:`accept_correction` before being used, and a
rejected correction leaves the draft untouched. The guard is what makes this
safe to enable by default on a subset.

Cost note: this roughly triples to octuples per-frame time, so it is *not*
intended for a whole-corpus sweep. Use it on a shortlist — frames whose mean
recognition confidence is low, which is where it pays.

**A Vietnamese-specialised model was evaluated and lost.** ``5CD-AI/Vintern-1B-v3_5``
(InternVL2.5-1B fine-tuned for Vietnamese OCR and handwriting) is eight times
smaller and correspondingly cheaper, which would matter a great deal over 290k
frames. As a *reader* it is genuinely good for its size — 12/14 on the pillar
frame at CER 0.020 with no draft at all, and it resolves ``Quỷ``. As a
*corrector* it does not work: given the draft it returns it verbatim. On the
recipe card all four generation settings tried (beams 1 and 3, repetition
penalty 1.0 and 2.5, 9 and 16 tiles) produced byte-identical output, which is
the draft. With beam search it echoes the instructions into its answer, taking
the signboard frame to CER 6.25. A 1B model given a long instruction with text
embedded in it copies rather than reasons over the image. Qwen2.5-VL-7B beats it
on every reference frame: 14/14 against 13/14 on the recipe card and the pillar,
2/2 against 1/2 on the signboard. It earns its place in a different role
entirely — see :mod:`scripts.ocr.reader`.

**A Vietnamese LLaVA was evaluated and cannot read at all.** ``Vi-VLM/Vistral-V-7B``
(SigLIP-base + Vistral-7B-Chat) scored **zero** required strings across all four
reference frames, under three prompts, three generation settings, and both
whole-frame and cropped-block modes. Character error rates ran from 0.72 to
115.7 — it emits far more text than the frames contain. What it emits is fluent
invented signage: ``Chúc mừng sinh nhật``, ``Trung tâm mua sắm``, ``Bánh mì kẹp
thịt ngon nhất thế giới``, and it twice insists the Vietnamese pillar is Chinese.
Its scene captions are accurate ("a woman holding a sheet of paper", "people
standing in the rain"), so this is a genuine capability gap rather than a broken
integration. Two causes, both structural: its vision tower is
``siglip-base-patch16-256``, which squashes any frame to 256x256 and gives the
language model 256 visual tokens against Qwen's ~1400, so 10px body text arrives
at about 2px; and its own model card states it was never trained on OCR. Cropping
to a single banner — enough pixels by any measure — did not help, which is the
part that settles it.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from scripts.ocr.scoring import fold, normalize


@dataclass(frozen=True)
class CorrectionGuard:
    """Thresholds deciding whether a proposed correction is a correction at all.

    A genuine tone-mark repair keeps the same words in the same order and changes
    only a few characters. A hallucinated rewrite changes the word count, drops
    content, or replaces words wholesale. These three limits separate the two
    without needing to know which characters are "allowed" to change.
    """

    #: Largest tolerated change in word count, as a fraction of the draft's.
    max_word_delta: float = 0.25
    #: Word-count change always tolerated regardless of draft length.
    #:
    #: A purely relative limit is unusable on short drafts, and short drafts are
    #: the common case — the median keyframe carries about four lines. Splitting
    #: a run-together detection into its real words is one of the most valuable
    #: corrections available (``GIAMMECCHING`` into ``GIA MỘC CHÂU``), and on a
    #: five-word draft that lone fix is a 40% word-count change, so the relative
    #: guard rejected it outright. Two words is enough headroom for a split while
    #: staying far too small to smuggle in an invented clause.
    word_delta_allowance: int = 2
    #: Least fraction of the draft's words that must survive, ignoring diacritics.
    #: Diacritic-blind on purpose: changing a tone mark is the whole point, so
    #: counting survival by exact match would reject every useful correction.
    min_word_retention: float = 0.75
    #: Largest tolerated character-level divergence, as a fraction of draft length.
    max_char_drift: float = 0.35


def _folded_words(text: str) -> list[str]:
    return fold(text).split()


def word_retention(draft: str, candidate: str) -> float:
    """Fraction of the draft's words still present in the candidate, ignoring tone marks."""
    draft_words = _folded_words(draft)
    if not draft_words:
        return 1.0
    remaining: dict[str, int] = {}
    for word in _folded_words(candidate):
        remaining[word] = remaining.get(word, 0) + 1
    kept = 0
    for word in draft_words:
        if remaining.get(word, 0) > 0:
            remaining[word] -= 1
            kept += 1
    return kept / len(draft_words)


def char_drift(draft: str, candidate: str) -> float:
    """Character-level distance between draft and candidate, ignoring tone marks.

    Folded, so that correcting ten tone marks reads as near-zero drift while
    rewriting a sentence does not.
    """
    from scripts.ocr.scoring import levenshtein

    reference = fold(draft)
    if not reference:
        return 0.0
    return levenshtein(reference, fold(candidate)) / len(reference)


def accept_correction(
    draft: str, candidate: str, guard: CorrectionGuard | None = None
) -> bool:
    """Whether ``candidate`` is a safe replacement for ``draft``."""
    guard = guard or CorrectionGuard()
    draft, candidate = (draft or "").strip(), (candidate or "").strip()
    if not candidate:
        return False
    if not draft:
        # With no draft there is nothing to measure divergence against, so every
        # check below passes vacuously and the model's output would be accepted
        # unconditionally — the exact ungrounded case that invents text.
        return False
    if normalize(candidate) == normalize(draft):
        return False           # nothing changed; nothing to accept
    draft_words = _folded_words(draft)
    if draft_words:
        delta = abs(len(_folded_words(candidate)) - len(draft_words))
        allowed = max(guard.word_delta_allowance, guard.max_word_delta * len(draft_words))
        if delta > allowed:
            return False
    if word_retention(draft, candidate) < guard.min_word_retention:
        return False
    return char_drift(draft, candidate) <= guard.max_char_drift


def _opcodes(draft: str, candidate: str):
    """Word-level diff on *folded* words, so a tone repair reads as a match."""
    import difflib

    return difflib.SequenceMatcher(
        a=_folded_words(draft), b=_folded_words(candidate), autojunk=False
    ).get_opcodes()


def lost_spans(draft: str, candidate: str, *, min_run: int = 3) -> list[tuple[int, int]]:
    """Draft word spans of ``min_run`` or more that the candidate does not carry.

    Net loss, not just the ``delete`` opcode. Content loss usually arrives
    disguised as a shrinking ``replace``: on the reference recipe card the
    corrector turned thirteen draft words into two, and because the first of them
    partially matched, the diff merged the whole span into one replacement rather
    than a replacement plus a deletion. Counting only outright deletions misses
    exactly the case worth catching.
    """
    spans = []
    for tag, i1, i2, j1, j2 in _opcodes(draft, candidate):
        if tag in ("delete", "replace") and (i2 - i1) - (j2 - j1) >= min_run:
            spans.append((i1, i2))
    return spans


def restore_deletions(draft: str, candidate: str, *, min_run: int = 3) -> str:
    """Put back whole phrases the corrector dropped, keeping its other fixes.

    :func:`accept_correction`'s guards are all aggregate — total word delta,
    total retention, total drift — and a localised deletion hides inside them.
    Measured on the reference recipe card: the corrector silently dropped the
    card's closing line, twelve consecutive words, while every threshold stayed
    comfortably clear at eight percent of a 149-word draft. That line is real
    text, and losing it makes it unsearchable.

    Rejecting the whole correction would fix the loss and throw away its genuine
    repairs with it — on that same frame it is what turns ``dâm`` into the
    correct ``dăm``. Splicing the dropped words back in keeps both, and follows
    from what this stage is for: it corrects characters, it does not get to edit.

    Runs shorter than ``min_run`` are left alone, since a one- or two-word
    deletion is more likely the corrector removing something the recogniser
    genuinely emitted twice.

    Within a flagged span the corrector's words are kept and only the draft words
    it failed to carry are appended after them. Discarding the whole span instead
    was tried and is worse: when the corrector rewrites a long stretch the diff
    merges it into one large replacement, so throwing the span away reverts every
    genuine fix inside it — on the reference card that cost the ``dâm`` to
    ``dăm`` repair along with several others. Appending is imperfect on word
    order, and that is the right trade: a line in a slightly odd position is
    still searchable, a deleted one is not.
    """
    draft_words = draft.split()
    candidate_words = candidate.split()
    if not draft_words or not candidate_words:
        return candidate
    lost = {span[0] for span in lost_spans(draft, candidate, min_run=min_run)}
    out: list[str] = []
    for tag, i1, i2, j1, j2 in _opcodes(draft, candidate):
        out.extend(candidate_words[j1:j2])
        if i1 not in lost or tag not in ("delete", "replace"):
            continue
        carried: dict[str, int] = {}
        for word in _folded_words(" ".join(candidate_words[j1:j2])):
            carried[word] = carried.get(word, 0) + 1
        for word in draft_words[i1:i2]:
            key = fold(word)
            if carried.get(key, 0) > 0:
                carried[key] -= 1
            else:
                out.append(word)
    return " ".join(out)


def reconcile_views(draft: str, primary: str, secondary: str) -> str:
    """Combine two corrections of the same draft, letting the recogniser arbitrate.

    Showing the corrector a deskewed copy of a tilted page more than halves its
    character error rate on the reference recipe card, because the model no
    longer has to read text running at nineteen degrees. But each view also
    introduces errors the other gets right: the original reads ``chảo nóng già``
    correctly where the deskewed one says ``giã``, and the deskewed one reads
    ``đổ ra chén`` where the original says ``chảo``.

    What breaks the tie is that the recogniser — which read every line
    individually, already rectified — agrees with whichever view is right in both
    cases: it returned ``già`` and it returned ``chến``. So where the two views
    disagree on a word, the draft's own reading decides, and only where the draft
    supports neither does ``secondary`` win by default, it being the stronger
    view overall.

    Alignment is on folded words, so the comparison is between spellings of the
    same word rather than between unrelated text.
    """
    import difflib

    primary_words, secondary_words = primary.split(), secondary.split()
    if not primary_words:
        return secondary
    if not secondary_words:
        return primary
    draft_forms = {fold(word) for word in draft.split()} | set(_folded_words(draft))
    draft_surfaces = set(draft.split())

    out: list[str] = []
    matcher = difflib.SequenceMatcher(
        a=_folded_words(primary), b=_folded_words(secondary), autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                first, second = primary_words[i1 + offset], secondary_words[j1 + offset]
                if first == second:
                    out.append(first)
                elif first in draft_surfaces and second not in draft_surfaces:
                    out.append(first)
                else:
                    out.append(second)
        elif tag == "replace":
            first = primary_words[i1:i2]
            second = secondary_words[j1:j2]
            first_support = sum(1 for w in _folded_words(" ".join(first))
                                if w in draft_forms)
            second_support = sum(1 for w in _folded_words(" ".join(second))
                                 if w in draft_forms)
            out.extend(first if first_support > second_support else second)
        elif tag == "delete":
            # Present in one view only: keep it rather than lose content.
            out.extend(primary_words[i1:i2])
        else:
            out.extend(secondary_words[j1:j2])
    return " ".join(out)


def absorb_reader(spine: str, reader: str, *, min_gain: int = 3) -> str:
    """Add text an independent reader saw and ``spine`` missed, without altering ``spine``.

    A second model reading the frame from scratch has different coverage rather
    than merely a different opinion. Measured on the reference signboard: the
    main pipeline returns ``KHU DU LỊCH QUỐC GIA MỘC CHÂU`` and never sees the
    tagline beneath it at all, because the recogniser collapses on that line and
    the corrector is forbidden to invent text. A Vietnamese-specialised reader
    given the cropped banner returns the tagline — and gets the place name wrong,
    calling it ``Mai Chiu``.

    So the two contributions must be treated differently, and position is what
    separates them. Aligned on folded words, a reading that competes for the same
    slot as the spine is an opinion, and the spine keeps it: it came from the
    corrector, which had the recogniser's grounded draft to work from. A reading
    that occupies no slot at all is coverage, and it is appended. The merge is
    additive by construction, so a weaker reader can widen recall but can never
    corrupt a word the stronger pipeline already got right.

    New coverage rarely arrives as a clean ``insert``. On that signboard the diff
    pairs the spine's two-word ``MỘC CHÂU`` against eleven reader words in a
    single replacement, because the competing name sits immediately before the
    new text; honouring only ``insert`` opcodes absorbs nothing at all. Where a
    replacement grows by ``min_gain`` words or more, the excess is therefore read
    off the *end* of the reader's span — the reader worked through the same
    region in the same order, so what it has beyond the spine's slot is its tail.
    That recovers the tagline without also dragging in ``Mai Chiu``.
    """
    import difflib

    spine_words, reader_words = spine.split(), reader.split()
    if not reader_words:
        return spine
    if not spine_words:
        return reader
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=_folded_words(spine), b=_folded_words(reader), autojunk=False
    ).get_opcodes():
        out.extend(spine_words[i1:i2])
        if tag == "insert":
            out.extend(reader_words[j1:j2])
        elif tag == "replace":
            gain = (j2 - j1) - (i2 - i1)
            if gain >= min_gain:
                out.extend(reader_words[j2 - gain:j2])
    return " ".join(out)


def merge_by_confidence(
    words: list[tuple[str, float]], corrected: str, *, max_conf: float = 0.80
) -> str | None:
    """Take the model's word only where the recogniser was unsure of its own.

    Whole-frame correction is all-or-nothing, and on the pillar frame that was a
    bad trade: the model repaired ``QUỲ`` to ``QUỶ`` but simultaneously spoiled
    ``KHẤP`` into ``KHẨP``. The recogniser's own confidence separates the two
    cleanly — it read ``QUỶ`` at 0.69 and ``KHẤP`` at 0.86 — so a word-level merge
    gated on confidence keeps both.

    ``words`` pairs each draft word with the confidence of the line it came from.
    Returns ``None`` when the corrected text does not align word-for-word with the
    draft, which is the caller's signal to keep the draft untouched: without
    alignment there is no way to tell a substitution from an insertion, and
    guessing would corrupt text the recogniser got right.
    """
    corrected_words = corrected.split()
    if len(corrected_words) != len(words):
        return None
    merged = []
    for (draft_word, conf), candidate in zip(words, corrected_words):
        merged.append(candidate if conf < max_conf else draft_word)
    return " ".join(merged)


#: Naming the *kinds* of error is what makes this work, and one earlier line in
#: it was actively harmful. The prompt used to demand "the EXACT same order and
#: the same number of words", which forbids the single most valuable correction
#: available: splitting a run-together detection back into real words. On the
#: reference banner that instruction is the whole reason ``GIAMMECCHING`` came
#: back as ``MỘC CHUẨN`` instead of ``MỘC CHÂU`` — told it could not change the
#: word count, the model produced a same-length guess. Dropping the clause and
#: naming signage and place names explicitly fixes it. Insertions are bounded by
#: :func:`accept_correction` instead, which is the right place for a limit that
#: has to hold whatever the model was told.
WHOLE_FRAME_PROMPT = (
    "This image is a frame from a Vietnamese video. The text below was extracted "
    "from it by an OCR system, already in reading order, and contains errors.\n\n"
    "OCR TEXT:\n{draft}\n\n"
    "Correct it, using the image as the source of truth. Typical errors:\n"
    "- wrong Vietnamese tone marks (Ả Ạ Ã Á À confused with each other, "
    "or Ă confused with Â)\n"
    "- two adjacent words run together into one, or one word split in two\n"
    "- stylised, handwritten or embossed text on signs and banners misread; "
    "these are frequently Vietnamese place names\n\n"
    "Rules:\n"
    "- Keep the same reading order.\n"
    "- Do not add any text that is not visible in the image.\n"
    "- Do not add commentary or headings.\n"
    "Output only the corrected text."
)


class VlmPostCorrector:
    """Lazily-loaded vision-LLM wrapper that repairs an OCR draft.

    Loading is deferred to first use so importing this module costs nothing and
    requires no GPU, keeping :func:`accept_correction` unit-testable on its own.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        min_pixels: int = 128 * 28 * 28,
        deskew_threshold: float = 5.0,
        max_pixels: int = 2560 * 28 * 28,
        guard: CorrectionGuard | None = None,
        max_conf: float = 0.80,
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.min_pixels = min_pixels
        self.deskew_threshold = deskew_threshold
        self.max_pixels = max_pixels
        self.guard = guard or CorrectionGuard()
        self.max_conf = max_conf
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,
        )

        quantization = None
        if self.load_in_4bit:
            # 4-bit NF4 puts the 7B model in about 6GB, which fits a 12GB card
            # alongside the detector and recogniser.
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization,
            dtype=torch.float16,
            device_map=self.device,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def _generate(self, image, draft: str) -> str:
        import torch

        self._load()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": WHOLE_FRAME_PROMPT.format(draft=draft)},
        ]}]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt], images=[image], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
        decoded = self._processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        return " ".join(decoded.split())

    def correct(self, image, draft: str, words: list[tuple[str, float]] | None = None,
                *, deskew: float = 0.0) -> str:
        """Return a corrected draft, or the original when the guard rejects the result.

        ``image`` is a PIL image of the whole frame. ``words`` optionally pairs
        each draft word with its recogniser confidence; when supplied, the
        correction is merged word-by-word under :func:`merge_by_confidence`
        instead of being taken wholesale.

        ``deskew`` is the page's measured tilt in degrees. When it is large the
        frame is corrected a second time from a straightened copy and the two
        readings are reconciled by :func:`reconcile_views` — worth roughly half
        the character errors on a tilted page, at the cost of a second pass.
        """
        if not draft.strip():
            return draft

        candidate = self._generate(image, draft)
        if abs(deskew) >= self.deskew_threshold:
            straightened = image.rotate(
                deskew, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
            )
            candidate = reconcile_views(
                draft, candidate, self._generate(straightened, draft)
            )
        # Repair before judging: a correction that is sound apart from having
        # dropped a line should be salvaged, not discarded.
        candidate = restore_deletions(draft, candidate)
        if not accept_correction(draft, candidate, self.guard):
            return draft
        if words:
            merged = merge_by_confidence(words, candidate, max_conf=self.max_conf)
            if merged is not None:
                return merged
        return candidate
