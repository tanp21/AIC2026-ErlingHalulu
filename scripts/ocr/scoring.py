"""Scoring primitives for the OCR engine bake-off.

Pure functions only: no I/O, no engine imports, no GPU. Everything here runs in
the unit test suite so the metrics themselves stay trustworthy even when no OCR
engine is installed.

Two normalisation levels are used throughout:

``normalize``
    NFC, casefolded, whitespace-collapsed. Diacritics are preserved, because
    telling ``TAO`` from ``TẢO`` is the entire point of a Vietnamese-first
    engine.
``fold``
    ``normalize`` plus diacritic stripping. Only ever used to report a
    *diagnostic* second number: when folded recall is high but normalized
    recall is low, the detector found the text and the recogniser mangled the
    tone marks, which is a completely different problem from missing the text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC-normalise, casefold, and collapse whitespace. Diacritics survive."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def fold(text: str) -> str:
    """``normalize`` plus diacritic removal, for diagnostics only.

    Vietnamese ``đ``/``Đ`` has no combining-mark decomposition, so it is mapped
    explicitly; without that, ``ĐỊA`` and ``DIA`` would still differ after
    stripping combining marks.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return normalize(stripped.replace("đ", "d").replace("Đ", "D"))


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings, O(len(a) * len(b)) time, O(len(b)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, 1):
        current = [i]
        for j, ch_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (ch_a != ch_b),  # substitution
            ))
        previous = current
    return previous[-1]


def strip_spans(text: str, spans=()) -> str:
    """Normalise ``text`` and delete each of ``spans`` from it.

    Used to take provably unreadable text out of scoring. It is applied to the
    *hypothesis as well as the reference*, which is the point: deleting it from
    the reference alone would turn a lucky read into a penalty, since the words
    would then look like an insertion. Excluded text must cost nothing either
    way, or the metric quietly discourages reading it.
    """
    out = normalize(text)
    for span in spans or ():
        span = normalize(span)
        if span:
            out = out.replace(span, " ")
    return _WHITESPACE.sub(" ", out).strip()


def character_error_rate(reference: str, hypothesis: str, *, exclude=()) -> float:
    """Levenshtein distance normalised by reference length.

    Returns 0.0 for an empty reference (nothing to get wrong). Values above 1.0
    are possible and meaningful: the engine hallucinated more than it read.

    ``exclude`` drops spans from both sides before measuring; see
    :func:`strip_spans`.
    """
    ref, hyp = strip_spans(reference, exclude), strip_spans(hypothesis, exclude)
    if not ref:
        return 0.0
    return levenshtein(ref, hyp) / len(ref)


def word_recall(reference: str, hypothesis: str, *, exclude=()) -> float:
    """Fraction of reference word tokens present in the hypothesis, with multiplicity.

    Complements CER: a page whose blocks come out in a different order scores
    terribly on CER but near 1.0 here, which isolates reading-order problems
    from recognition problems.
    """
    ref_words = strip_spans(reference, exclude).split()
    if not ref_words:
        return 1.0
    remaining: dict[str, int] = {}
    for word in strip_spans(hypothesis, exclude).split():
        remaining[word] = remaining.get(word, 0) + 1
    found = 0
    for word in ref_words:
        if remaining.get(word, 0) > 0:
            remaining[word] -= 1
            found += 1
    return found / len(ref_words)


def contains(haystack: str, needle: str, *, diacritic_blind: bool = False) -> bool:
    """Substring test under the chosen normalisation level."""
    transform = fold if diacritic_blind else normalize
    return transform(needle) in transform(haystack)


@dataclass(frozen=True)
class FrameScore:
    """Per-frame result of scoring one engine configuration."""

    frame_id: str
    acceptance: bool
    required_total: int
    required_found: int
    required_missing: tuple[str, ...]
    required_found_folded: int
    phrase_total: int
    phrase_found: int
    phrase_missing: tuple[str, ...]
    phrase_found_folded: int
    cer: float
    word_recall: float
    seconds: float
    tags: tuple[str, ...] = ()

    @property
    def required_recall(self) -> float:
        return 1.0 if not self.required_total else self.required_found / self.required_total

    @property
    def folded_recall(self) -> float:
        """Required-string recall ignoring diacritics.

        The practically important companion to :attr:`required_recall`. The
        Elasticsearch mapping indexes an ``asciifolding`` sub-field, so a hit
        whose only defect is a wrong tone mark is still retrievable. The gap
        between the two numbers is precisely the share of failures that survive
        into search, versus those the index recovers.
        """
        if not self.required_total:
            return 1.0
        return self.required_found_folded / self.required_total

    @property
    def phrase_recall(self) -> float:
        return 1.0 if not self.phrase_total else self.phrase_found / self.phrase_total

    @property
    def folded_phrase_recall(self) -> float:
        """Phrase recall ignoring diacritics — the reading-order signal on its own.

        Strict phrase recall conflates two unrelated failures: emitting the words
        in the wrong order, and getting one tone mark wrong somewhere in a
        thirty-character phrase. Only the first is a layout bug. Comparing this
        against :attr:`phrase_recall` separates them.
        """
        if not self.phrase_total:
            return 1.0
        return self.phrase_found_folded / self.phrase_total

    @property
    def passed(self) -> bool:
        """A frame passes only when nothing required is missing."""
        return not self.required_missing and not self.phrase_missing


def score_frame(entry: dict, text: str, seconds: float = 0.0) -> FrameScore:
    """Score one engine's output for one ground-truth entry.

    ``text`` is the engine's reading-order-joined output for the whole frame.
    Required substrings are searched in it directly, so an engine that detects
    every word but emits them in the wrong order still fails ``required_phrases``
    while passing ``required`` — which is exactly the distinction the two-column
    pillar frame exists to expose.
    """
    required = tuple(entry.get("required", ()))
    phrases = tuple(entry.get("required_phrases", ()))
    # Text on the frame that has been *demonstrated* unreadable, not merely text
    # the current engine misses. Kept in 'full_text' so the reference stays an
    # honest transcription, and excluded here so CER measures the achievable.
    unreadable = tuple(entry.get("unreadable", ()))
    missing = tuple(s for s in required if not contains(text, s))
    missing_phrases = tuple(p for p in phrases if not contains(text, p))
    return FrameScore(
        frame_id=entry["id"],
        acceptance=bool(entry.get("acceptance")),
        required_total=len(required),
        required_found=len(required) - len(missing),
        required_missing=missing,
        required_found_folded=sum(
            1 for s in required if contains(text, s, diacritic_blind=True)
        ),
        phrase_total=len(phrases),
        phrase_found=len(phrases) - len(missing_phrases),
        phrase_missing=missing_phrases,
        phrase_found_folded=sum(
            1 for p in phrases if contains(text, p, diacritic_blind=True)
        ),
        cer=character_error_rate(entry.get("full_text", ""), text, exclude=unreadable),
        word_recall=word_recall(entry.get("full_text", ""), text, exclude=unreadable),
        seconds=seconds,
        tags=tuple(entry.get("tags", ())),
    )


@dataclass
class ConfigReport:
    """Aggregate of every :class:`FrameScore` for one engine configuration."""

    config_id: str
    scores: list[FrameScore] = field(default_factory=list)

    def _mean(self, attribute: str, scores: list[FrameScore]) -> float:
        if not scores:
            return 0.0
        return sum(getattr(s, attribute) for s in scores) / len(scores)

    @property
    def acceptance_scores(self) -> list[FrameScore]:
        return [s for s in self.scores if s.acceptance]

    #: Default acceptance thresholds.
    #:
    #: Exact recall is deliberately below 1.0. One required string on the pillar
    #: frame — the hook-above in ``QUỶ`` — is not reachable by any recogniser
    #: measured: VietOCR reads it as ``QUỲ`` under all ten crop-preprocessing
    #: variants tried, and Qwen2.5-VL reads it as ``QUÝ``. Demanding 1.0 would
    #: make the gate permanently unsatisfiable and therefore meaningless.
    #:
    #: The two folded thresholds stay at 1.0 and carry the real weight: together
    #: they assert that every required string was *found and placed correctly*,
    #: so any failure is a tone mark rather than missed text or scrambled reading
    #: order. Those are the failures Elasticsearch's asciifolding sub-field
    #: recovers at query time; missed text is not recoverable at all.
    MIN_REQUIRED_RECALL = 0.90
    MIN_FOLDED_RECALL = 1.0
    MIN_FOLDED_PHRASE_RECALL = 1.0

    def gate(
        self,
        *,
        min_required: float | None = None,
        min_folded: float | None = None,
        min_folded_phrase: float | None = None,
    ) -> bool:
        """Whether this configuration clears the acceptance thresholds."""
        acceptance = self.acceptance_scores
        if not acceptance:
            return False
        min_required = self.MIN_REQUIRED_RECALL if min_required is None else min_required
        min_folded = self.MIN_FOLDED_RECALL if min_folded is None else min_folded
        if min_folded_phrase is None:
            min_folded_phrase = self.MIN_FOLDED_PHRASE_RECALL
        return (
            self._mean("required_recall", acceptance) >= min_required
            and self._mean("folded_recall", acceptance) >= min_folded
            and self._mean("folded_phrase_recall", acceptance) >= min_folded_phrase
        )

    @property
    def acceptance_passed(self) -> bool:
        return self.gate()

    @property
    def frames_per_second(self) -> float:
        total = sum(s.seconds for s in self.scores)
        return len(self.scores) / total if total > 0 else 0.0

    def by_tag(self, tag: str) -> list[FrameScore]:
        return [s for s in self.scores if tag in s.tags]

    def summary(self) -> dict:
        return {
            "config_id": self.config_id,
            "frames": len(self.scores),
            "acceptance_passed": self.acceptance_passed,
            "acceptance_required_recall": round(
                self._mean("required_recall", self.acceptance_scores), 4
            ),
            "acceptance_phrase_recall": round(
                self._mean("phrase_recall", self.acceptance_scores), 4
            ),
            "acceptance_folded_recall": round(
                self._mean("folded_recall", self.acceptance_scores), 4
            ),
            "required_recall": round(self._mean("required_recall", self.scores), 4),
            "folded_recall": round(self._mean("folded_recall", self.scores), 4),
            "phrase_recall": round(self._mean("phrase_recall", self.scores), 4),
            "folded_phrase_recall": round(
                self._mean("folded_phrase_recall", self.scores), 4
            ),
            "cer": round(self._mean("cer", self.scores), 4),
            "word_recall": round(self._mean("word_recall", self.scores), 4),
            "frames_per_second": round(self.frames_per_second, 2),
            "failures": {
                s.frame_id: {
                    "required": list(s.required_missing),
                    "phrases": list(s.phrase_missing),
                }
                for s in self.scores
                if not s.passed
            },
        }


def rank_configs(reports: list[ConfigReport], min_fps: float = 5.0) -> list[ConfigReport]:
    """Order configurations best-first under the Phase 1 selection rule.

    Gate first, then accuracy, then speed: a configuration that clears the
    acceptance frames always outranks one that does not, however fast it is.
    Throughput only breaks ties between configurations that are already correct,
    and ``min_fps`` demotes anything too slow to survive an 848k-frame sweep.
    """
    return sorted(
        reports,
        key=lambda r: (
            not r.acceptance_passed,
            r.frames_per_second < min_fps,
            -r.summary()["required_recall"],
            -r.summary()["phrase_recall"],
            r.summary()["cer"],
            -r.frames_per_second,
        ),
    )
