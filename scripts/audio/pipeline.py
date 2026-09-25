"""Compose the transcript-cleaning step in one place.

``build_clean_transcripts`` goes through here so the config it reads and the
restoration it runs are the same object. Restoration (capitalisation,
punctuation, and the occasional ASR-mistranscription fix) runs through Gemini
(:mod:`scripts.audio.gemini_clean`) — the only cleaning transform left in this
module.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.audio.postprocess import Word, merge_tokens_to_words


@dataclass(frozen=True)
class CleanConfig:
    """Which cleaning step runs.

    The default is deliberately inert — an all-default config reproduces the
    raw transcripts.
    """

    #: Gemini model id for restoration via the API, or ``None`` to skip
    #: restoration entirely.
    gemini_model: str | None = None

    @property
    def label(self) -> str:
        return f"gemini:{self.gemini_model}" if self.gemini_model else "raw (no transform enabled)"

    def to_json(self) -> dict:
        return {"gemini_model": self.gemini_model}

    @classmethod
    def from_json(cls, data: dict) -> "CleanConfig":
        return cls(gemini_model=data.get("gemini_model"))


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
    gemini=None,
) -> dict[str, list[Word]]:
    """Apply ``config`` to a whole corpus.

    ``gemini`` supplies an already-loaded restorer; pass ``None`` and one is
    loaded from ``config.gemini_model``.
    """
    cleaned = words_by_video
    if config.gemini_model:
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
