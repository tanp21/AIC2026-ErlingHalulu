"""Compose the transcript-cleaning transforms in one agreed order.

Both the benchmark and ``build_clean_transcripts`` go through here, so the
pipeline the sweep measures is by construction the pipeline that ships. The
transforms themselves live in :mod:`scripts.audio.postprocess` and stay pure;
this module owns only the ordering and the parquet reading.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
from pathlib import Path

from scripts.audio.postprocess import (
    PHRASE_WORDS,
    Word,
    boilerplate_document_frequency,
    drop_low_confidence,
    merge_tokens_to_words,
    normalize_numbers,
    phrases_above,
    strip_boilerplate,
)


@dataclass(frozen=True)
class CleanConfig:
    """Which cleaning steps run, and how hard.

    Defaults are deliberately inert — an all-default config reproduces the raw
    transcripts — so the benchmark's baseline and its candidates go down exactly
    the same code path.
    """

    #: Drop words whose NLL exceeds this. ``inf`` keeps everything.
    max_nll: float = math.inf
    #: ``None`` disables boilerplate removal.
    boilerplate_min_videos: int | None = None
    boilerplate_phrase_words: int = PHRASE_WORDS
    #: Rewrite spelled-out numbers as digits.
    numbers: bool = False
    #: Hugging Face id of a capitalisation/punctuation restorer, or ``None``.
    capu_model: str | None = None
    #: Gemini model id for restoration via the API, or ``None``. Mutually
    #: exclusive with ``capu_model`` — both do the same job in the pipeline
    #: (the last, character-adding step), so at most one restorer runs.
    gemini_model: str | None = None

    def __post_init__(self) -> None:
        if self.capu_model and self.gemini_model:
            raise ValueError(
                "CleanConfig: capu_model and gemini_model are mutually exclusive"
            )

    @property
    def label(self) -> str:
        parts = []
        parts.append("raw" if math.isinf(self.max_nll) else f"nll<={self.max_nll:g}")
        if self.boilerplate_min_videos is not None:
            parts.append(
                f"boiler>={self.boilerplate_min_videos}v"
                f"/{self.boilerplate_phrase_words}w"
            )
        if self.numbers:
            parts.append("numbers")
        if self.capu_model:
            parts.append("capu:" + self.capu_model.split("/")[-1].replace("-capu", ""))
        if self.gemini_model:
            parts.append("gemini:" + self.gemini_model)
        return " ".join(parts)

    def to_json(self) -> dict:
        return {
            "max_nll": None if math.isinf(self.max_nll) else self.max_nll,
            "boilerplate_min_videos": self.boilerplate_min_videos,
            "boilerplate_phrase_words": self.boilerplate_phrase_words,
            "numbers": self.numbers,
            "capu_model": self.capu_model,
            "gemini_model": self.gemini_model,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CleanConfig":
        max_nll = data.get("max_nll")
        return cls(
            max_nll=math.inf if max_nll is None else float(max_nll),
            boilerplate_min_videos=data.get("boilerplate_min_videos"),
            boilerplate_phrase_words=data.get("boilerplate_phrase_words", PHRASE_WORDS),
            numbers=data.get("numbers", False),
            capu_model=data.get("capu_model"),
            gemini_model=data.get("gemini_model"),
        )


RAW = CleanConfig()


def load_words(step_data: Path) -> dict[str, list[Word]]:
    """Read ``step_data.parquet`` and merge its BPE pieces into words per video."""
    import pyarrow.parquet as pq

    columns = pq.read_table(
        step_data, columns=["audio_key", "offset", "token", "log_prob"]
    ).to_pydict()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for key, offset, token, nll in zip(
        columns["audio_key"], columns["offset"], columns["token"], columns["log_prob"]
    ):
        grouped[key].append({"offset": offset, "token": token, "log_prob": nll})
    return {key: merge_tokens_to_words(rows) for key, rows in sorted(grouped.items())}


def clean_corpus(
    words_by_video: dict[str, list[Word]],
    config: CleanConfig,
    *,
    document_frequency=None,
    capu=None,
    gemini=None,
) -> dict[str, list[Word]]:
    """Apply ``config`` to a whole corpus.

    Corpus-wide, not per-video, because boilerplate is defined by recurrence
    *across* videos and cannot be recognised from one video alone.

    Order matters, and it is forced by what each step reads:

    1. **Confidence filtering** first, so a garbage word inside a sponsor read
       does not stop that read being recognised.
    2. **Boilerplate removal** while words are still bare. Phrase matching folds
       diacritics but keeps punctuation, so a comma appended by restoration
       would stop ``ty,`` matching ``ty``.
    3. **Number normalisation** before restoration, for the same reason in
       reverse: it matches on whole words, and ``tư.`` is not ``tư``. Feeding
       the restorer digits is harmless — it was trained on ordinary text, which
       has them.
    4. **Capitalisation and punctuation** last, since it is the only step that
       adds characters other steps would then have to look past.

    ``document_frequency`` lets a sweep reuse the expensive phrase count across
    candidates. It must have been computed from *this* word stream at
    ``config.boilerplate_phrase_words``; pass ``None`` and it is computed here.
    ``capu`` supplies an already-loaded restorer; pass ``None`` and one is
    loaded from ``config.capu_model``. ``gemini`` is the same idea for
    ``config.gemini_model`` — at most one of the two configs is set, so at
    most one of the two loads.
    """
    cleaned = words_by_video
    if math.isfinite(config.max_nll):
        cleaned = {
            key: drop_low_confidence(words, max_nll=config.max_nll)
            for key, words in cleaned.items()
        }
    if config.boilerplate_min_videos is not None:
        if document_frequency is None:
            document_frequency = boilerplate_document_frequency(
                cleaned, phrase_words=config.boilerplate_phrase_words
            )
        phrases = phrases_above(document_frequency, config.boilerplate_min_videos)
        cleaned = {
            key: strip_boilerplate(
                words, phrases, phrase_words=config.boilerplate_phrase_words
            )
            for key, words in cleaned.items()
        }
    if config.numbers:
        cleaned = {key: normalize_numbers(words) for key, words in cleaned.items()}
    if config.capu_model:
        from scripts.audio.capu import CapuModel, restore_words

        if capu is None:
            capu = CapuModel(config.capu_model).load()
        total = len(cleaned)
        restored = {}
        for index, (key, words) in enumerate(sorted(cleaned.items()), 1):
            restored[key] = restore_words(words, capu)
            if index % 50 == 0:
                print(f"[clean] restored {index}/{total} videos", flush=True)
        cleaned = restored
    elif config.gemini_model:
        from scripts.audio.gemini_clean import GeminiCleaner, restore_words_gemini

        if gemini is None:
            gemini = GeminiCleaner(config.gemini_model).load()
        total = len(cleaned)
        restored = {}
        for index, (key, words) in enumerate(sorted(cleaned.items()), 1):
            restored[key] = restore_words_gemini(words, gemini)
            if index % 20 == 0:
                print(f"[clean] gemini-restored {index}/{total} videos", flush=True)
        cleaned = restored
    return cleaned


def sweep_configs() -> list[CleanConfig]:
    """Candidate configurations for ``bench_audio_search --sweep``.

    One axis at a time against the raw baseline first, so the report attributes
    a recall change to the transform that caused it, then the axes combined.
    """
    candidates = [RAW]
    # The '<' placeholder averages 2.89 NLL against 0.11 for real speech, and
    # the corpus 75th percentile is 0.109 — so these bracket "obvious garbage
    # only" through "confident words only".
    candidates += [replace(RAW, max_nll=value) for value in (3.0, 1.0, 0.3)]
    candidates += [
        replace(RAW, boilerplate_min_videos=value) for value in (40, 20, 10, 5)
    ]
    candidates.append(replace(RAW, numbers=True))
    return candidates
