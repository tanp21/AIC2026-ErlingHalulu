#!/usr/bin/env python3
"""Gold-video recall for the audio branch, measured before and after cleaning.

The transcripts in ``data/2026/audio`` are raw ASR output: no punctuation, no
capitalisation, numbers spelled out as words, and a sponsor read that repeats
across 606 of the 873 videos. Every cleaning transform in :mod:`scripts.audio`
has to justify itself here — a transform that does not move recall does not
ship, however sensible it sounds.

The metric is **gold-video recall**: for each labelled query, the rank at which
the correct video first appears. Passage-level ranks are collapsed to their
video, which is what ``backend.audio_search.merge_temporal_hits`` does in production.

Three modes, in increasing cost and fidelity:

``--lexical`` (default)
    BM25 in pure Python over the same passages the indexer builds. No
    Elasticsearch, no GPU. It mirrors ``backend.audio_search.build_bm25_request``: the
    diacritic-preserving field is boosted 2x against a diacritic-folded field,
    combined with ``max`` rather than a sum because the production query is
    ``best_fields``.

``--dense``
    Qwen passage embeddings and cosine ranking, in numpy. Slower, needs a GPU
    and the Hugging Face cache, but it is the **only** offline way to see
    punctuation and capitalisation: BM25 lowercases and discards punctuation, in
    this bench and in Elasticsearch alike, so restoring them cannot change a
    lexical score by construction.

``--elasticsearch``
    The real thing — ``backend.audio_search.search_audio``, BM25 and kNN fused by RRF.
    Requires Elasticsearch up and the index built.

The offline modes are proxies for ranking *changes*, not estimates of production
recall. Use them to compare transcript variants against each other.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from backend import audio_search
from backend.audio_search import SemanticChunkConfig, TopicChunkConfig, make_passages
from scripts.indexing.build_audio_index import CUE_MAX_TOKENS
from scripts.audio.pipeline import load_words
from scripts.audio.postprocess import words_to_tokens
from scripts.ocr.scoring import fold, normalize

ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ROOT / "benchmarks" / "audio" / "audio_ground_truth.json"
RAW_STEP_DATA = ROOT / "data" / "2026" / "audio" / "step_data.parquet"
CLEAN_STEP_DATA = ROOT / "data" / "2026" / "audio" / "step_data_clean.parquet"
#: Passage embeddings are expensive and derived, so they live beside their
#: source under the gitignored data mount rather than in the repo.
EMBEDDING_CACHE = ROOT / "data" / "2026" / "audio" / ".bench_embeddings"

#: Reported cut-offs. 1 and 5 are what a competitor actually looks at; 100 is
#: the pool the fusion in ``backend.app.fuse_search_results`` gets to work with.
RECALL_AT = (1, 5, 10, 50, 100)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, *, diacritics: bool) -> list[str]:
    """Split into terms the way the two Elasticsearch analysers do.

    ``text`` uses the standard analyser plus lowercase, which keeps tone marks;
    ``text.folded`` adds ``asciifolding``. Reusing :func:`scripts.ocr.scoring.fold`
    keeps this bench, the OCR bench, and the postprocessors agreeing on what
    "the same word" means.
    """
    return _WORD.findall(normalize(text) if diacritics else fold(text))


@dataclass(frozen=True)
class QueryResult:
    """Where the gold video landed for one query."""

    query_id: str
    kind: str
    video_id: str
    rank: int | None  # None when the gold video is nowhere in the ranking


@dataclass(frozen=True)
class Report:
    label: str
    results: list[QueryResult]

    def recall_at(self, k: int) -> float:
        hits = sum(1 for r in self.results if r.rank is not None and r.rank <= k)
        return hits / len(self.results) if self.results else 0.0

    @property
    def median_rank(self) -> float:
        ranks = sorted(r.rank if r.rank is not None else math.inf for r in self.results)
        if not ranks:
            return math.inf
        middle = len(ranks) // 2
        if len(ranks) % 2:
            return ranks[middle]
        return (ranks[middle - 1] + ranks[middle]) / 2

    def ranks_by_query(self) -> dict[str, int | None]:
        return {r.query_id: r.rank for r in self.results}


class LexicalIndex:
    """Okapi BM25 over passages, scored on two analyser variants.

    Held in memory as plain dicts. At ~31k passages for the 2026 corpus this
    builds in a couple of seconds, which is the point: it makes "did that
    transform help?" a question you can answer between edits.
    """

    def __init__(self, passages, k1: float = 1.2, b: float = 0.75):
        self.video_ids = [p.video_id for p in passages]
        self.k1 = k1
        self.b = b
        # (best_fields_boost, phrase_boost, field) — the phrase_boost ratio
        # (4.0/2.0 = 2x the best_fields ratio 2.0/1.0) mirrors the ^4/^2 vs
        # ^2/^1 weights build_bm25_request gives its two multi_match clauses.
        self._fields = [
            (2.0, 4.0, self._build([p.text for p in passages], diacritics=True)),
            (1.0, 2.0, self._build([p.text for p in passages], diacritics=False)),
        ]

    def _build(self, texts, *, diacritics: bool):
        term_frequencies = []
        positions_by_passage = []
        for t in texts:
            tokens = tokenize(t, diacritics=diacritics)
            term_frequencies.append(Counter(tokens))
            positions: dict[str, list[int]] = defaultdict(list)
            for position, token in enumerate(tokens):
                positions[token].append(position)
            positions_by_passage.append(positions)
        lengths = [sum(tf.values()) for tf in term_frequencies]
        average_length = sum(lengths) / len(lengths) if lengths else 0.0
        postings = defaultdict(list)
        for index, tf in enumerate(term_frequencies):
            for term, count in tf.items():
                postings[term].append((index, count))
        total = len(term_frequencies)
        idf = {
            term: math.log(1 + (total - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }
        return postings, idf, lengths, average_length, positions_by_passage

    def _score_field(self, field, terms) -> dict[int, float]:
        postings, idf, lengths, average_length, _ = field
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            plist = postings.get(term)
            if not plist:
                continue
            weight = idf[term]
            for index, count in plist:
                norm = 1 - self.b + self.b * lengths[index] / average_length
                scores[index] += weight * count * (self.k1 + 1) / (count + self.k1 * norm)
        return scores

    @staticmethod
    def _phrase_positions_match(positions: dict, terms: list[str], slop: int) -> bool:
        """True if ``terms`` occur in order, each consecutive pair at most
        ``slop + 1`` tokens apart. A deliberately simplified proxy for
        Elasticsearch's phrase-query slop — good enough to distinguish "this
        exact phrase is present" from "it isn't", which is all the bonus
        needs; not a reimplementation of ES's internal phrase scoring.
        """
        for start in positions.get(terms[0], []):
            cursor = start
            ok = True
            for term in terms[1:]:
                candidates = [p for p in positions.get(term, []) if cursor < p <= cursor + slop + 1]
                if not candidates:
                    ok = False
                    break
                cursor = min(candidates)
            if ok:
                return True
        return False

    def _phrase_bonus(self, field, terms: list[str]) -> dict[int, float]:
        """Additive bonus for passages containing ``terms`` as a near-phrase.

        Mirrors ``build_bm25_request``'s second `multi_match` clause: a
        `bool`/`should` in Elasticsearch *sums* matching-clause scores, so an
        exact-phrase hit should score higher than the best_fields match
        alone, not just tiebreak it.
        """
        _, idf, _, _, positions_by_passage = field
        if len(terms) < 2:
            return {}
        weight = sum(idf.get(term, 0.0) for term in terms)
        bonus: dict[int, float] = {}
        for index, positions in enumerate(positions_by_passage):
            if self._phrase_positions_match(positions, terms, slop=2):
                bonus[index] = weight
        return bonus

    def rank_videos(self, query: str) -> list[str]:
        """Rank video ids by their single best-scoring passage."""
        combined: dict[int, float] = defaultdict(float)
        phrase_total: dict[int, float] = defaultdict(float)
        for best_boost, phrase_boost, field in self._fields:
            diacritics = field is self._fields[0][2]
            terms = tokenize(query, diacritics=diacritics)
            for index, score in self._score_field(field, terms).items():
                # best_fields takes the maximum across fields, not the sum.
                combined[index] = max(combined[index], best_boost * score)
            for index, bonus in self._phrase_bonus(field, terms).items():
                phrase_total[index] = max(phrase_total[index], phrase_boost * bonus)
        total: dict[int, float] = defaultdict(float)
        for index in set(combined) | set(phrase_total):
            # bool/should sums its matching clauses' scores.
            total[index] = combined.get(index, 0.0) + phrase_total.get(index, 0.0)
        best_per_video: dict[str, float] = {}
        for index, score in total.items():
            video_id = self.video_ids[index]
            if score > best_per_video.get(video_id, float("-inf")):
                best_per_video[video_id] = score
        return sorted(best_per_video, key=best_per_video.get, reverse=True)


def model_kwargs_for(quantize: str | None, device_map: str | None) -> dict:
    """``model_kwargs`` for ``SentenceTransformer``, covering quantization and
    multi-GPU sharding for models too large to fit fp16 on one GPU.

    bitsandbytes quantization requires ``device_map`` at load time (it cannot
    ``.to(device)`` a quantized model afterwards), so ``--quantize`` implies
    ``device_map="auto"`` unless ``--device-map`` overrides it. ``device_map``
    (quantized or not) means the caller must not also pass ``device=`` to
    ``SentenceTransformer`` — accelerate already placed the weights.
    """
    kwargs: dict = {}
    if quantize:
        import torch
        from transformers import BitsAndBytesConfig

        if quantize == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantize == "int4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        kwargs["device_map"] = device_map or "auto"
    elif device_map:
        kwargs["device_map"] = device_map
    return kwargs


class DenseIndex:
    """Qwen passage embeddings and cosine ranking, without Elasticsearch.

    This mode exists because BM25 **cannot see** punctuation or capitalisation:
    both the Elasticsearch ``standard`` analyser and this bench's tokenizer
    lowercase and drop punctuation, so restoring them is provably a no-op for
    the lexical half of the system. Their entire value is in the embedding half,
    and this is the cheapest honest way to measure it — the same model and the
    same query prompt ``build_audio_index`` and ``audio_search`` use, scored in
    numpy instead of over the network.
    """

    def __init__(self, passages, *, device: str | None = None, batch_size: int = 32,
                 cache_dir: Path | None = None, quantize: str | None = None,
                 device_map: str | None = None):
        import numpy as np

        self.video_ids = [p.video_id for p in passages]
        self.np = np
        self._device = device
        self._quantize = quantize
        self._device_map = device_map
        self._encoder = None
        self.vectors = self._embed([p.text for p in passages], device, batch_size, cache_dir)

    @staticmethod
    def _fingerprint(texts) -> str:
        import hashlib

        digest = hashlib.sha256()
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    def _embed(self, texts, device, batch_size, cache_dir):
        import numpy as np

        from backend import audio_search

        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Keyed on model + quantization too, not just passage text: two
            # variants of the same model can chunk to identical passage text
            # (embed_cues uses the same model), which would otherwise silently
            # replay one variant's vectors for the other.
            model_tag = audio_search.MODEL_NAME.replace("/", "_")
            variant = f"{model_tag}-{self._quantize or 'fp16'}-{self._device_map or 'single'}"
            cache_path = cache_dir / f"passages-{variant}-{self._fingerprint(texts)}.npy"
            if cache_path.exists():
                print(f"[bench-audio] reusing embeddings from {cache_path.name}")
                return np.load(cache_path)

        from backend import audio_search

        print(f"[bench-audio] embedding {len(texts):,} passages "
              f"(model {audio_search.MODEL_NAME})")
        vectors = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        if cache_path is not None:
            np.save(cache_path, vectors)
        return vectors

    @property
    def encoder(self):
        """Loaded on demand, so a cached-embedding run never touches the GPU."""
        if self._encoder is None:
            import os

            from sentence_transformers import SentenceTransformer

            from backend import audio_search

            kwargs = model_kwargs_for(self._quantize, self._device_map)
            self._encoder = SentenceTransformer(
                audio_search.MODEL_NAME,
                device=None if kwargs.get("device_map") else
                       (self._device or os.environ.get("AUDIO_EMBED_DEVICE", "cuda")),
                model_kwargs=kwargs or None,
            )
            # Same guard as passages_from_words' embed_cues and
            # build_audio_index.py's own indexer: a passage that couldn't be
            # split further (a single pathologically long, unpunctuated
            # sentence -- see the embed_cues guard above) still gets handed
            # to *this* separate encoder instance for its passage-level
            # vector, which had no cap of its own and OOM'd a 12 GB card the
            # same way.
            self._encoder.max_seq_length = CUE_MAX_TOKENS
        return self._encoder

    def rank_videos(self, query: str) -> list[str]:
        from backend import audio_search

        vector = self.encoder.encode(
            query,
            prompt=f"Instruct: {audio_search.QUERY_INSTRUCTION}\nQuery:",
            normalize_embeddings=True,
        ).astype(self.np.float32)
        scores = self.vectors @ vector
        best: dict[str, float] = {}
        for index in self.np.argsort(-scores):
            video_id = self.video_ids[index]
            if video_id not in best:
                best[video_id] = float(scores[index])
        return list(best)


def load_ground_truth(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def passages_from_words(words_by_video, window_seconds: int = 60, stride_seconds: int = 15,
                        *, strategy: str = "sliding", similarity_threshold: float = 0.5,
                        device: str | None = None, batch_size: int = 32,
                        chunk_config: "audio_search.SemanticChunkConfig | None" = None,
                        topic_config: "audio_search.TopicChunkConfig | None" = None,
                        llm_segment_fn=None, deepseek_concurrency: int = 8,
                        quantize: str | None = None, device_map: str | None = None):
    """Build passages exactly the way ``build_audio_index`` will.

    Going back through ``words_to_tokens`` rather than windowing the words
    directly is deliberate: the bench then exercises the same round-trip the
    indexer does, so a bug in it shows up here rather than in production.

    ``strategy in ("semantic", "semantic_duration", "topic")`` loads a Qwen
    encoder on first use (same model, same device-resolution order as
    ``DenseIndex.encoder``) to embed cues for boundary detection — a separate
    concern from ``DenseIndex``'s own passage-level embeddings, so it is not
    shared with that cache. ``topic`` only uses this encoder for its
    max-duration fallback splitter (``_split_oversized_topic_segment``), not
    for the LLM's own boundary decisions.

    ``topic`` is the one strategy that is network-bound rather than
    GPU-bound: each video makes several sequential DeepSeek calls inside
    ``make_topic_passages`` (one per sentence window --
    ``_stitch_topic_boundaries`` deliberately stays single-threaded, see its
    docstring), so a corpus-wide run is dispatched across videos on a bounded
    thread pool instead -- otherwise a ~1,200-video corpus would run one API
    call at a time. ``DeepSeekTopicSegmenter``'s cache reads/writes and
    ``_RateLimiter`` are shared safely across these threads (independent
    cache files per distinct sentence batch; the rate limiter's own lock is
    exactly what keeps the *combined* dispatch rate bounded across threads).
    """
    embed_cues = None
    if strategy in ("semantic", "semantic_duration", "topic"):
        import os

        from sentence_transformers import SentenceTransformer

        kwargs = model_kwargs_for(quantize, device_map)
        encoder = SentenceTransformer(
            audio_search.MODEL_NAME,
            device=None if kwargs.get("device_map") else
                   (device or os.environ.get("AUDIO_EMBED_DEVICE", "cuda")),
            model_kwargs=kwargs or None,
        )
        # Same guard as build_audio_index.py's own embed_cues: cleaning is
        # decided per chunk, so a video can come out part cleaned and part
        # raw, and a raw stretch with no punctuation for the sentence
        # splitter to find merges into one gigantic "sentence" (measured on
        # this exact corpus: S01-V001 produced a 52,589-character merged
        # sentence). Encoding that uncapped OOMs a 12 GB card via a
        # quadratic attention allocation, well before any --strategy or
        # boundary-detection logic runs.
        encoder.max_seq_length = CUE_MAX_TOKENS

        def embed_cues(texts):
            return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)

    if strategy == "topic":
        from concurrent.futures import ThreadPoolExecutor, as_completed

        items = sorted(words_by_video.items())

        def build_one(video_id, words):
            rows = words_to_tokens(video_id, words)
            return video_id, audio_search.make_topic_passages(
                video_id, rows, llm_segment_fn, embed_cues=embed_cues, config=topic_config
            )

        results: dict[str, list] = {}
        with ThreadPoolExecutor(max_workers=max(1, deepseek_concurrency)) as pool:
            futures = [pool.submit(build_one, video_id, words) for video_id, words in items]
            done = 0
            for future in as_completed(futures):
                video_id, video_passages = future.result()
                results[video_id] = video_passages
                done += 1
                if done % 20 == 0 or done == len(futures):
                    print(f"[bench-audio] topic strategy: {done}/{len(futures)} "
                          f"videos segmented", flush=True)
        passages = []
        for video_id, _words in items:
            passages.extend(results[video_id])
        return passages

    passages = []
    for video_id, words in sorted(words_by_video.items()):
        rows = words_to_tokens(video_id, words)
        if strategy == "semantic_duration":
            passages.extend(audio_search.make_bounded_semantic_passages(
                video_id, rows, embed_cues, config=chunk_config
            ))
        elif strategy == "semantic":
            passages.extend(audio_search.make_semantic_passages(
                video_id, rows, embed_cues, similarity_threshold=similarity_threshold
            ))
        else:
            passages.extend(make_passages(video_id, rows, window_seconds, stride_seconds))
    return passages


def rank_with_elasticsearch(query: str, limit: int) -> list[str]:
    from backend import audio_search

    ordered: list[str] = []
    for hit in audio_search.search_audio(query, limit):
        if hit.video_id not in ordered:
            ordered.append(hit.video_id)
    return ordered


def evaluate(label: str, queries: list[dict], rank_fn) -> Report:
    results = []
    for query in queries:
        ranking = rank_fn(query["text"])
        rank = ranking.index(query["video_id"]) + 1 if query["video_id"] in ranking else None
        results.append(
            QueryResult(query["id"], query["kind"], query["video_id"], rank)
        )
    return Report(label, results)


def print_report(report: Report, baseline: Report | None = None) -> None:
    print(f"\n=== {report.label} ({len(report.results)} queries) ===")
    cells = []
    for k in RECALL_AT:
        value = report.recall_at(k)
        cell = f"@{k}: {value:6.1%}"
        if baseline is not None:
            delta = value - baseline.recall_at(k)
            cell += f" ({delta:+.1%})"
        cells.append(cell)
    print("  recall  " + "   ".join(cells))
    print(f"  median rank: {report.median_rank}")
    if baseline is None:
        return
    before = baseline.ranks_by_query()
    moved = []
    for result in report.results:
        old, new = before.get(result.query_id), result.rank
        if old != new:
            moved.append((result.query_id, old, new))
    if not moved:
        print("  no query changed rank")
        return
    def delta(item):
        _, old, new = item
        return (new if new is not None else math.inf) - (old if old is not None else math.inf)
    moved.sort(key=delta)
    print("  rank changes (improved first):")
    for query_id, old, new in moved:
        print(f"    {query_id:<28} {str(old):>5} -> {str(new):>5}")


def print_summary(reports: list[Report], baseline: Report) -> None:
    """One row per candidate, so the whole sweep fits on a screen."""
    header = "  ".join(f"@{k:<5}" for k in RECALL_AT)
    print(f"\n{'config':<34}  {header}  median")
    print("-" * (34 + len(header) + 12))
    for report in reports:
        cells = []
        for k in RECALL_AT:
            delta = report.recall_at(k) - baseline.recall_at(k)
            marker = " " if report is baseline else ("+" if delta > 0 else ("-" if delta < 0 else "="))
            cells.append(f"{report.recall_at(k):5.1%}{marker}")
        print(f"{report.label:<34}  {'  '.join(cells)}  {report.median_rank}")


def print_duration_stats(passages, *, label: str) -> None:
    """Chunk count and duration distribution, alongside the recall tables.

    Recall on 48 queries hides small effects (see the module docstring's
    sibling report on transcript cleaning); chunk shape does not have that
    problem, so it is worth eyeballing directly rather than only inferring it
    from recall deltas.
    """
    if not passages:
        print(f"  {label:<34} no passages")
        return
    durations = sorted(p.end_sec - p.start_sec for p in passages)
    videos = len({p.video_id for p in passages})
    mean = sum(durations) / len(durations)
    median = durations[len(durations) // 2]
    p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))]
    print(f"  {label:<34} {len(passages):>7,} passages / {videos:>3} videos "
          f"({len(passages) / videos:5.1f}/video)  "
          f"duration mean={mean:5.1f}s median={median:5.1f}s p95={p95:5.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    parser.add_argument("--transcripts", type=Path, default=RAW_STEP_DATA,
                        help="step_data parquet to score")
    parser.add_argument("--baseline-transcripts", type=Path, default=None,
                        help="score this parquet too and report the delta. Use it to "
                             "compare an already-written cleaned parquet against the "
                             "raw one, which keeps the restoration model out of "
                             "memory while the embeddings are built")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--lexical", dest="mode", action="store_const", const="lexical",
                      help="pure-Python BM25; no Elasticsearch and no GPU (default)")
    mode.add_argument("--dense", dest="mode", action="store_const", const="dense",
                      help="rank by Qwen embeddings; the only offline way to see "
                           "punctuation and capitalisation, which BM25 discards")
    mode.add_argument("--elasticsearch", dest="mode", action="store_const",
                      const="elasticsearch",
                      help="use the real audio_search.search_audio path")
    parser.set_defaults(mode="lexical")
    parser.add_argument("--embedding-cache", type=Path, default=EMBEDDING_CACHE,
                        help="where to memoise passage embeddings between runs")
    parser.add_argument("--device", default=None, help="override AUDIO_EMBED_DEVICE")
    parser.add_argument("--model", default=None,
                        help="override audio_search.MODEL_NAME, e.g. to bench "
                             "Qwen3-Embedding-4B/8B instead of the shipped 0.6B")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="encode batch size for both boundary-detection cues "
                             "and DenseIndex passage embeddings; raise this on a "
                             "GPU with VRAM to spare (e.g. a rented card) to cut "
                             "wall-clock time, especially with a larger --model")
    parser.add_argument("--quantize", choices=["int8", "int4"], default=None,
                        help="load the embedding model with bitsandbytes quantization "
                             "(implies device_map='auto' unless --device-map is given)")
    parser.add_argument("--device-map", default=None,
                        help="accelerate device_map, e.g. 'balanced' to shard a model "
                             "too large for one GPU across all visible GPUs; overrides "
                             "--device when set")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--stride-seconds", type=int, default=15)
    parser.add_argument("--strategy",
                        choices=["sliding", "semantic", "semantic_duration", "topic"],
                        default="sliding",
                        help="passage-construction strategy (default: today's sliding window)")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                        help="cue-merge threshold for --strategy semantic")
    parser.add_argument("--chunk-sweep", action="store_true",
                        help="score similarity_threshold candidates (0.3-0.8) for "
                             "--strategy semantic against --transcripts as-is (no "
                             "cleaning transform), plus the sliding-window baseline")
    # --strategy semantic_duration only. Defaults match SemanticChunkConfig's.
    parser.add_argument("--context-window-sentences", type=int, default=3)
    parser.add_argument("--breakpoint-percentile", type=float, default=15.0)
    parser.add_argument("--min-scores-for-percentile", type=int, default=4)
    parser.add_argument("--min-duration-seconds", type=float, default=8.0)
    parser.add_argument("--target-duration-seconds", type=float, default=25.0)
    parser.add_argument("--max-duration-seconds", type=float, default=40.0)
    # None, not a literal default: semantic_duration's default (1) and
    # topic's (0, see TopicChunkConfig) genuinely differ, so this shared flag
    # is resolved per --strategy below.
    parser.add_argument("--overlap-sentences", type=int, default=None)
    # --strategy topic only.
    parser.add_argument("--sentences-per-batch", type=int, default=60)
    parser.add_argument("--batch-overlap-sentences", type=int, default=10)
    parser.add_argument("--fallback-context-window-sentences", type=int, default=3)
    parser.add_argument("--fallback-breakpoint-percentile", type=float, default=15.0)
    parser.add_argument("--deepseek-model", default=None)
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument("--deepseek-api-key", default=None,
                        help="falls back to the DEEPSEEK_API_KEY env var")
    parser.add_argument("--deepseek-rpm", type=float, default=None)
    parser.add_argument("--deepseek-cache-dir", type=Path, default=None,
                        help="persist this between sweep runs so repeated "
                             "--lexical/--dense passes never re-pay DeepSeek "
                             "for an unchanged video")
    parser.add_argument("--deepseek-concurrency", type=int, default=8,
                        help="videos segmented in parallel for --strategy topic "
                             "(each video itself makes several sequential "
                             "DeepSeek calls, one per sentence window)")
    # Phase 2, off by default -- not exercised by --duration-sweep below yet.
    pause = parser.add_mutually_exclusive_group()
    pause.add_argument("--pause-boost", dest="pause_boost", action="store_true", default=False)
    pause.add_argument("--no-pause-boost", dest="pause_boost", action="store_false")
    parser.add_argument("--pause-relative-percentile", type=float, default=75.0)
    parser.add_argument("--pause-borderline-semantic-percentile", type=float, default=30.0)
    parser.add_argument("--duration-sweep", action="store_true",
                        help="staged ablation for --strategy semantic_duration against "
                             "--transcripts as-is: context window, then breakpoint "
                             "percentile, then min/target/max duration, then overlap, "
                             "each stage keeping the previous stage's winner. Pause- "
                             "and shot-aware boosting are a later phase and are not "
                             "part of this sweep")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if args.model:
        audio_search.MODEL_NAME = args.model

    queries = load_ground_truth(args.ground_truth)
    chunk_config = SemanticChunkConfig(
        context_window_sentences=args.context_window_sentences,
        breakpoint_percentile=args.breakpoint_percentile,
        min_scores_for_percentile=args.min_scores_for_percentile,
        min_duration_seconds=args.min_duration_seconds,
        target_duration_seconds=args.target_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        overlap_sentences=(args.overlap_sentences if args.overlap_sentences is not None
                           else SemanticChunkConfig().overlap_sentences),
        pause_boost_enabled=args.pause_boost,
        pause_relative_percentile=args.pause_relative_percentile,
        pause_borderline_semantic_percentile=args.pause_borderline_semantic_percentile,
    )
    topic_config = TopicChunkConfig(
        sentences_per_batch=args.sentences_per_batch,
        batch_overlap_sentences=args.batch_overlap_sentences,
        min_duration_seconds=args.min_duration_seconds,
        target_duration_seconds=args.target_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        overlap_sentences=(args.overlap_sentences if args.overlap_sentences is not None
                           else TopicChunkConfig().overlap_sentences),
        fallback_context_window_sentences=args.fallback_context_window_sentences,
        fallback_breakpoint_percentile=args.fallback_breakpoint_percentile,
    )
    llm_segment_fn = None
    if args.strategy == "topic":
        # Lazily imported so every other strategy stays free of DeepSeek and
        # its API-key requirement.
        from scripts.audio.deepseek_topic_segment import build_segmenter

        llm_segment_fn = build_segmenter(
            model=args.deepseek_model, api_key=args.deepseek_api_key,
            base_url=args.deepseek_base_url, cache_dir=args.deepseek_cache_dir,
            requests_per_minute=args.deepseek_rpm,
        )

    if args.mode == "elasticsearch":
        report = evaluate(
            "elasticsearch", queries, lambda text: rank_with_elasticsearch(text, args.limit)
        )
        print_report(report)
        return

    def score_file(path: Path, *, strategy: str = "sliding",
                   similarity_threshold: float = 0.5,
                   duration_config: SemanticChunkConfig | None = None,
                   topic_chunk_config: "TopicChunkConfig | None" = None) -> Report:
        """Score a parquet as it stands, applying no transforms.

        Deliberately a separate path from ``score``: it holds one word stream at
        a time and never loads a restoration model, which is what makes a dense
        comparison fit in memory alongside Qwen.
        """
        words = load_words(path)
        passages = passages_from_words(words, args.window_seconds, args.stride_seconds,
                                       strategy=strategy, similarity_threshold=similarity_threshold,
                                       device=args.device, batch_size=args.batch_size,
                                       chunk_config=duration_config,
                                       topic_config=topic_chunk_config, llm_segment_fn=llm_segment_fn,
                                       deepseek_concurrency=args.deepseek_concurrency,
                                       quantize=args.quantize, device_map=args.device_map)
        if strategy == "semantic_duration":
            cfg = duration_config or SemanticChunkConfig()
            label = (f"{path.stem}:dur@w{cfg.context_window_sentences}"
                     f"p{cfg.breakpoint_percentile:g}"
                     f"[{cfg.min_duration_seconds:g}-{cfg.target_duration_seconds:g}"
                     f"-{cfg.max_duration_seconds:g}]ov{cfg.overlap_sentences}")
        elif strategy == "semantic":
            label = f"{path.stem}:semantic@{similarity_threshold:.1f}"
        elif strategy == "topic":
            cfg = topic_chunk_config or TopicChunkConfig()
            label = (f"{path.stem}:topic@b{cfg.sentences_per_batch}"
                     f"o{cfg.batch_overlap_sentences}"
                     f"[{cfg.min_duration_seconds:g}-{cfg.max_duration_seconds:g}]"
                     f"ov{cfg.overlap_sentences}")
        else:
            label = path.stem
        print(f"[bench-audio] {label:<34} "
              f"{sum(len(w) for w in words.values()):>9,} words  "
              f"{len(passages):>7,} passages")
        print_duration_stats(passages, label=label)
        del words
        index = (DenseIndex(passages, device=args.device, batch_size=args.batch_size,
                            cache_dir=args.embedding_cache,
                            quantize=args.quantize, device_map=args.device_map)
                 if args.mode == "dense" else LexicalIndex(passages))
        return evaluate(label, queries, index.rank_videos)

    if args.chunk_sweep:
        baseline = score_file(args.transcripts)
        candidates = [
            score_file(args.transcripts, strategy="semantic", similarity_threshold=t)
            for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
        ]
        print_summary([baseline] + candidates, baseline)
        return

    if args.duration_sweep:
        from dataclasses import replace

        baseline = score_file(args.transcripts)
        winner = SemanticChunkConfig()

        def stage(label: str, values, apply):
            nonlocal winner
            reports = [
                score_file(args.transcripts, strategy="semantic_duration",
                          duration_config=apply(winner, value))
                for value in values
            ]
            print(f"\n[bench-audio] duration-sweep stage: {label}")
            print_summary([baseline] + reports, baseline)
            best = max(range(len(reports)),
                      key=lambda i: (reports[i].recall_at(1), reports[i].recall_at(10)))
            winner = apply(winner, values[best])
            return winner

        # One axis at a time against the raw baseline, each stage fixing the
        # previous stage's winner before moving on. Pause- and shot-aware
        # boosting are a later phase, not swept here.
        stage("context_window_sentences", (1, 2, 3, 4),
             lambda cfg, v: replace(cfg, context_window_sentences=v))
        stage("breakpoint_percentile", (10.0, 15.0, 20.0, 25.0),
             lambda cfg, v: replace(cfg, breakpoint_percentile=v))
        stage("min/target/max duration", ((8, 20, 30), (8, 25, 40), (10, 25, 40), (8, 30, 45)),
             lambda cfg, v: replace(cfg, min_duration_seconds=v[0],
                                    target_duration_seconds=v[1], max_duration_seconds=v[2]))
        stage("overlap_sentences", (0, 1, 2),
             lambda cfg, v: replace(cfg, overlap_sentences=v))

        print(f"\n[bench-audio] duration-sweep winner: {winner.to_json()}")
        return

    if args.baseline_transcripts:
        baseline = score_file(args.baseline_transcripts, strategy=args.strategy,
                              similarity_threshold=args.similarity_threshold,
                              duration_config=chunk_config, topic_chunk_config=topic_config)
        candidate = score_file(args.transcripts, strategy=args.strategy,
                               similarity_threshold=args.similarity_threshold,
                               duration_config=chunk_config, topic_chunk_config=topic_config)
        print_summary([baseline, candidate], baseline)
        print_report(candidate, baseline)
        return

    # No sweep, no baseline comparison: a single report on --transcripts as it
    # stands. Comparing a cleaned parquet against the raw one is
    # --baseline-transcripts' job (above) — cleaning now happens out-of-process
    # via scripts.audio.build_clean_transcripts, not as an in-memory config
    # this bench applies itself.
    report = score_file(args.transcripts, strategy=args.strategy,
                        similarity_threshold=args.similarity_threshold,
                        duration_config=chunk_config, topic_chunk_config=topic_config)
    print_report(report)


if __name__ == "__main__":
    main()
