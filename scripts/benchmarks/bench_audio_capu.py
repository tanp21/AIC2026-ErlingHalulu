#!/usr/bin/env python3
"""Bake-off between Vietnamese capitalisation/punctuation restoration models.

There is no punctuated version of the transcripts to score against — that is the
whole problem. So the evaluation is built by *removing* formatting from
Vietnamese that already has it: the competition query files and
``RULES_GUIDELINES.md``, about 8,500 words. Lowercase them, strip the
punctuation, ask the model to put it back, and compare against the original.

Three numbers decide it:

**Word-count preservation** — the safety gate. ``apply_capu_output`` rejects any
chunk where the model returns a different number of words, so a model that
splits or merges words is not merely inaccurate, it is inert. This is reported
first because a model can score well on the other two and still be useless.

**Punctuation F1** and **capitalisation accuracy** — quality, per mark, so a
model that only ever predicts commas is visible as such.

The domain is admittedly not broadcast narration: competition queries are
written prose. This bench picks between candidates and catches a broken
integration; whether restoration helps *retrieval* is a separate question, and
only ``bench_audio_search --dense`` answers it. BM25 lowercases and drops
punctuation, so it cannot see this work at all.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "benchmarks" / "audio" / "audio_config.json"
QUERIES = ROOT / "queries"
RULES = ROOT / "RULES_GUIDELINES.md"

#: The label set the published models restore. Marks outside it are folded onto
#: the nearest supported one so the reference cannot ask for the impossible.
PUNCTUATION = (".", ",", ":", "?")
_FOLD_PUNCTUATION = {"!": ".", ";": ",", "…": ".", "–": ",", "—": ","}

_WORDS_PER_SAMPLE = 40


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"^\s*[#>*\-|]+", " ", text, flags=re.MULTILINE)
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)


def load_reference_text(queries_dir: Path = QUERIES, rules: Path = RULES) -> list[str]:
    """Every scrap of correctly punctuated Vietnamese in the repo."""
    chunks = []
    for path in sorted(queries_dir.glob("query-p*-groupA/*.txt")):
        chunks.append(path.read_text(encoding="utf-8"))
    exam = queries_dir / "SOTUYEN1-bo-de-thi.zip"
    if exam.exists():
        with zipfile.ZipFile(exam) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith(".txt"):
                    chunks.append(archive.read(name).decode("utf-8"))
    if rules.exists():
        chunks.append(_strip_markdown(rules.read_text(encoding="utf-8")))
    return chunks


def _fold_marks(word: str) -> str:
    return "".join(_FOLD_PUNCTUATION.get(ch, ch) for ch in word)


@dataclass(frozen=True)
class Token:
    """One reference word: its letters, whether it was capitalised, what followed."""

    word: str          # letters only, lowercased
    capitalised: bool
    punctuation: str   # one of PUNCTUATION, or "" for none


def tokenize_reference(text: str) -> list[Token]:
    tokens = []
    for raw in _fold_marks(text).split():
        trailing = ""
        stripped = raw
        while stripped and stripped[-1] in PUNCTUATION:
            if not trailing:
                trailing = stripped[-1]
            stripped = stripped[:-1]
        letters = "".join(
            ch for ch in stripped
            if ch.isalnum() or unicodedata.category(ch).startswith("M")
        )
        if not letters:
            continue
        tokens.append(Token(letters.lower(), letters[0].isupper(), trailing))
    return tokens


def build_samples(chunks, words_per_sample: int = _WORDS_PER_SAMPLE) -> list[list[Token]]:
    """Cut the reference into model-sized pieces, never across a source boundary."""
    samples = []
    for chunk in chunks:
        tokens = tokenize_reference(chunk)
        for start in range(0, len(tokens), words_per_sample):
            piece = tokens[start:start + words_per_sample]
            if len(piece) >= 5:
                samples.append(piece)
    return samples


def to_input(tokens) -> str:
    """What the ASR would have produced: lowercase, unpunctuated."""
    return " ".join(token.word for token in tokens)


@dataclass
class CapuReport:
    model_id: str
    samples: int = 0
    aligned: int = 0
    words: int = 0
    seconds: float = 0.0
    capital_correct: int = 0
    capital_reference: int = 0
    capital_predicted: int = 0
    capital_true_positive: int = 0
    punctuation: Counter = field(default_factory=Counter)

    @property
    def preservation(self) -> float:
        return self.aligned / self.samples if self.samples else 0.0

    @property
    def capital_accuracy(self) -> float:
        return self.capital_correct / self.words if self.words else 0.0

    @property
    def capital_f1(self) -> float:
        return _f1(self.capital_true_positive, self.capital_predicted, self.capital_reference)

    def punctuation_f1(self, mark: str) -> float:
        return _f1(
            self.punctuation[f"tp:{mark}"],
            self.punctuation[f"pred:{mark}"],
            self.punctuation[f"ref:{mark}"],
        )

    @property
    def macro_f1(self) -> float:
        scores = [self.punctuation_f1(mark) for mark in PUNCTUATION]
        return sum(scores) / len(scores)

    @property
    def words_per_second(self) -> float:
        return self.words / self.seconds if self.seconds else 0.0


def _f1(true_positive: int, predicted: int, reference: int) -> float:
    if not predicted or not reference:
        return 0.0
    precision = true_positive / predicted
    recall = true_positive / reference
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_sample(report: CapuReport, reference, restored: str) -> None:
    """Fold one restored sample into ``report``, or count it as unaligned."""
    hypothesis = tokenize_reference(restored)
    report.samples += 1
    if len(hypothesis) != len(reference):
        # Rejected by apply_capu_output in production, so it earns no credit.
        return
    report.aligned += 1
    for want, got in zip(reference, hypothesis):
        report.words += 1
        if want.capitalised == got.capitalised:
            report.capital_correct += 1
        report.capital_reference += want.capitalised
        report.capital_predicted += got.capitalised
        report.capital_true_positive += want.capitalised and got.capitalised
        if want.punctuation:
            report.punctuation[f"ref:{want.punctuation}"] += 1
        if got.punctuation:
            report.punctuation[f"pred:{got.punctuation}"] += 1
        if want.punctuation and want.punctuation == got.punctuation:
            report.punctuation[f"tp:{want.punctuation}"] += 1


def evaluate(model_id: str, samples, *, device: str | None, limit: int | None) -> CapuReport:
    from scripts.audio.capu import CapuModel

    model = CapuModel(model_id, device=device).load()
    report = CapuReport(model_id)
    chosen = samples[:limit] if limit else samples
    for index, reference in enumerate(chosen, 1):
        started = time.perf_counter()
        try:
            restored = model.restore(to_input(reference))
        except Exception as error:  # a candidate that cannot run has lost
            print(f"[bench-capu] {model_id} failed on sample {index}: {error}")
            report.samples += 1
            continue
        report.seconds += time.perf_counter() - started
        score_sample(report, reference, restored)
        if index % 25 == 0:
            print(f"[bench-capu] {model_id}: {index}/{len(chosen)} samples", flush=True)
    return report


def print_reports(reports) -> None:
    print(f"\n{'model':<34} {'align':>6} {'cap.F1':>7} {'cap.acc':>8} "
          + " ".join(f"{mark:>6}" for mark in PUNCTUATION)
          + f" {'macro':>7} {'w/s':>7}")
    print("-" * 104)
    for report in reports:
        marks = " ".join(f"{report.punctuation_f1(m):6.3f}" for m in PUNCTUATION)
        print(f"{report.model_id:<34} {report.preservation:6.1%} "
              f"{report.capital_f1:7.3f} {report.capital_accuracy:8.1%} {marks} "
              f"{report.macro_f1:7.3f} {report.words_per_second:7.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=None,
                        help="model ids to compare (default: scripts.audio.capu.CANDIDATES)")
    parser.add_argument("--device", default=None, help="override AUDIO_EMBED_DEVICE")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N samples")
    parser.add_argument("--write-config", action="store_true",
                        help="record the winner in benchmarks/audio/audio_config.json")
    args = parser.parse_args()

    from scripts.audio.capu import CANDIDATES

    samples = build_samples(load_reference_text())
    total_words = sum(len(s) for s in samples)
    print(f"[bench-capu] {len(samples)} samples, {total_words:,} reference words")

    reports = [
        evaluate(model_id, samples, device=args.device, limit=args.limit)
        for model_id in (args.models or CANDIDATES)
    ]
    print_reports(reports)

    # A model that cannot hold the word count is unusable however good its F1,
    # so preservation sorts before quality.
    winner = max(reports, key=lambda r: (r.preservation, r.macro_f1, r.capital_f1))
    print(f"\n[bench-capu] winner: {winner.model_id}")

    if args.write_config:
        payload = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
        payload["capu"] = {
            "model_id": winner.model_id,
            "word_count_preserved": round(winner.preservation, 4),
            "punctuation_macro_f1": round(winner.macro_f1, 4),
            "capitalisation_f1": round(winner.capital_f1, 4),
            "words_per_second": round(winner.words_per_second, 1),
            "eval": "strip-and-restore over queries/ and RULES_GUIDELINES.md",
        }
        CONFIG.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(f"[bench-capu] wrote {CONFIG}")


if __name__ == "__main__":
    main()
