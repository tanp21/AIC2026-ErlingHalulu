import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import audio_search
from backend.audio_search import (
    AudioHit,
    SemanticChunkConfig,
    build_bm25_request,
    build_index_definition,
    build_knn_request,
    make_bounded_semantic_passages,
    make_passages,
    make_semantic_passages,
    merge_temporal_hits,
    parse_search_response,
    reciprocal_rank_fusion,
    search_audio,
)


class PassageTests(unittest.TestCase):
    def test_sixty_second_passage_keeps_spread_out_query_evidence(self):
        rows = [
            {"offset": 848.56, "token": " chuyến bay tư nhân"},
            {"offset": 857.16, "token": " phi hành đoàn bốn người"},
            {"offset": 889.52, "token": " ánh sáng cực quang ở vùng cực"},
        ]

        passages = make_passages("L21_V015", rows, window_seconds=60, stride_seconds=15)

        passage = next(p for p in passages if p.start_sec == 840)
        self.assertEqual(passage.end_sec, 900)
        self.assertEqual(passage.anchor_sec, 870)
        self.assertIn("chuyến bay tư nhân", passage.text)
        self.assertIn("ánh sáng cực quang", passage.text)

    def test_passages_cover_tokens_on_window_boundary(self):
        rows = [
            {"offset": 0.0, "token": " first"},
            {"offset": 59.999, "token": " inside"},
            {"offset": 60.0, "token": " boundary"},
        ]

        passages = make_passages("L21_V001", rows, window_seconds=60, stride_seconds=15)

        self.assertIn("inside", passages[0].text)
        self.assertNotIn("boundary", passages[0].text)
        self.assertIn("boundary", passages[1].text)


class SemanticPassageTests(unittest.TestCase):
    def test_empty_rows_produce_no_passages(self):
        self.assertEqual(make_semantic_passages("L21_V001", [], lambda texts: []), [])

    def test_a_similarity_drop_starts_a_new_passage(self):
        rows = [
            {"offset": 0.0, "token": " chủ đề một."},
            {"offset": 20.0, "token": " chủ đề hai."},
        ]

        def embed_cues(texts):
            self.assertEqual(len(texts), 2)
            return [[1.0, 0.0], [0.0, 1.0]]  # orthogonal -> similarity 0

        passages = make_semantic_passages("L21_V001", rows, embed_cues, similarity_threshold=0.5)

        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].start_sec, 0.0)
        self.assertEqual(passages[0].text, "chủ đề một.")
        self.assertEqual(passages[1].start_sec, 20.0)
        self.assertEqual(passages[1].text, "chủ đề hai.")

    def test_similar_cues_merge_into_one_passage(self):
        rows = [
            {"offset": 0.0, "token": " một."},
            {"offset": 20.0, "token": " hai."},
            {"offset": 40.0, "token": " ba."},
        ]

        def embed_cues(texts):
            return [[1.0, 0.0]] * 3  # identical -> similarity 1

        passages = make_semantic_passages("L21_V001", rows, embed_cues, similarity_threshold=0.5)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].text, "một. hai. ba.")
        self.assertEqual(passages[0].start_sec, 0.0)
        self.assertEqual(passages[0].end_sec, 42.0)
        self.assertEqual(passages[0].passage_id, "L21_V001:0")

    def test_sentence_fragments_are_rejoined_before_embedding(self):
        # build_transcript_cues splits on an 8s gap even mid-sentence; the
        # fragment ("một") must be rejoined with what follows ("hai.") into
        # one sentence before embed_cues ever sees it.
        rows = [
            {"offset": 0.0, "token": " một"},
            {"offset": 20.0, "token": " hai."},
            {"offset": 40.0, "token": " ba."},
        ]

        def embed_cues(texts):
            self.assertEqual(texts, ["một hai.", "ba."])
            return [[1.0, 0.0], [0.0, 1.0]]

        passages = make_semantic_passages("L21_V001", rows, embed_cues, similarity_threshold=0.5)

        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].text, "một hai.")

    def test_boundary_similarity_equal_to_threshold_merges(self):
        rows = [
            {"offset": 0.0, "token": " một."},
            {"offset": 20.0, "token": " hai."},
        ]

        def embed_cues(texts):
            return [[1.0, 0.0], [1.0, 0.0]]  # identical -> similarity exactly 1.0

        passages = make_semantic_passages("L21_V001", rows, embed_cues, similarity_threshold=1.0)

        self.assertEqual(len(passages), 1)


class ChangeScoreTests(unittest.TestCase):
    def test_fewer_than_two_vectors_produce_no_scores(self):
        self.assertEqual(audio_search._change_scores([]), [])
        self.assertEqual(audio_search._change_scores([[1.0, 0.0]]), [])

    def test_identical_vectors_have_zero_change(self):
        vectors = [[1.0, 0.0]] * 4

        scores = audio_search._change_scores(vectors, window_sentences=3)

        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertAlmostEqual(score, 0.0, places=5)

    def test_topic_shift_is_the_strongest_boundary(self):
        # Two sentences on one topic, two on another: the real shift, between
        # index 1 and 2, should score higher than either same-topic boundary.
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        scores = audio_search._change_scores(vectors, window_sentences=3)

        self.assertEqual(len(scores), 3)
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 1)


class BreakpointCandidateTests(unittest.TestCase):
    def test_too_few_scores_yields_no_candidates(self):
        # Below min_scores, a percentile over so few points is not a
        # meaningful statistic — this is the signal the assembler uses to
        # fall back to duration-only, sentence-aware chunking.
        scores = [0.1, 0.9, 0.2]

        self.assertEqual(
            audio_search._breakpoint_candidates(scores, percentile=15, min_scores=4), set()
        )

    def test_local_maxima_within_percentile_are_selected(self):
        scores = [0.1, 0.9, 0.15, 0.2, 0.85, 0.1, 0.12, 0.13]

        candidates = audio_search._breakpoint_candidates(scores, percentile=15, min_scores=4)

        self.assertEqual(candidates, {1, 4})


class BoundedAssemblerTests(unittest.TestCase):
    """Unit tests for `_assemble_bounded_chunks`, the duration-constrained
    boundary-selection policy, using hand-built sentence spans and scores so
    the decision logic is exercised directly, without a model."""

    def _config(self, **overrides):
        defaults = dict(min_duration_seconds=8, target_duration_seconds=25, max_duration_seconds=40)
        defaults.update(overrides)
        return SemanticChunkConfig(**defaults)

    def test_worked_example_picks_strongest_valley_over_closest_to_target(self):
        # Candidates at 11s/0.71, 19s/0.52, 27s/0.68, 36s/0.91; only 27s and
        # 36s fall in [target=25, max=40], and 36s is the stronger valley.
        sentences = [
            {"start": 0.0, "end": 11.0, "text": "s0"},
            {"start": 11.0, "end": 19.0, "text": "s1"},
            {"start": 19.0, "end": 27.0, "text": "s2"},
            {"start": 27.0, "end": 36.0, "text": "s3"},
            {"start": 36.0, "end": 38.0, "text": "s4"},
        ]
        scores = [0.71, 0.52, 0.68, 0.91]
        candidates = {0, 1, 2, 3}

        chunks = audio_search._assemble_bounded_chunks(sentences, scores, candidates, self._config())

        self.assertEqual(chunks[0][-1]["end"], 36.0)

    def test_growth_zone_ignores_a_strong_pre_target_candidate(self):
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 15.0, "text": "b"},
            {"start": 15.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 41.0, "text": "d"},
        ]
        scores = [0.99, 0.10, 0.05]
        config = self._config()

        with_early_candidate = audio_search._assemble_bounded_chunks(sentences, scores, {0}, config)
        without_it = audio_search._assemble_bounded_chunks(sentences, scores, set(), config)

        self.assertEqual(with_early_candidate, without_it)
        # And it is a real force-split near max, not a stop at the 5s mark.
        self.assertEqual(without_it[0][-1]["end"], 39.0)

    def test_no_candidate_forces_split_closest_to_max_duration(self):
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 15.0, "text": "b"},
            {"start": 15.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 41.0, "text": "d"},
        ]
        scores = [0.5, 0.5, 0.5]

        chunks = audio_search._assemble_bounded_chunks(sentences, scores, set(), self._config())

        self.assertEqual(chunks[0][-1]["end"], 39.0)

    def test_single_sentence_longer_than_max_is_emitted_alone(self):
        sentences = [
            {"start": 0.0, "end": 50.0, "text": "huge"},
            {"start": 50.0, "end": 65.0, "text": "next"},
        ]
        scores = [0.5]

        chunks = audio_search._assemble_bounded_chunks(sentences, scores, set(), self._config())

        self.assertEqual(chunks[0], [sentences[0]])
        self.assertEqual(len(chunks), 2)

    def test_undersized_trailing_tail_merges_into_previous_chunk(self):
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 30.0, "text": "b"},
            {"start": 30.0, "end": 33.0, "text": "c"},
            {"start": 33.0, "end": 35.0, "text": "d"},
        ]
        scores = [0.2, 0.6, 0.2]
        candidates = {1}  # boundary after "b", at cumulative 30s

        chunks = audio_search._assemble_bounded_chunks(sentences, scores, candidates, self._config())

        self.assertEqual(len(chunks), 2)
        # The short 2s tail ("d" alone) never appears as its own chunk.
        self.assertEqual(chunks[-1][0]["text"], "b")
        self.assertEqual(chunks[-1][-1]["end"], 35.0)

    def test_overlap_sentence_appears_in_both_adjacent_chunks(self):
        sentences = [
            {"start": 0.0, "end": 10.0, "text": "p1"},
            {"start": 10.0, "end": 28.0, "text": "p2"},
            {"start": 28.0, "end": 30.0, "text": "p3"},
        ]
        scores = [0.1, 0.8]
        candidates = {1}

        chunks = audio_search._assemble_bounded_chunks(
            sentences, scores, candidates, self._config(overlap_sentences=1)
        )

        self.assertEqual(len(chunks), 2)
        self.assertIs(chunks[0][-1], chunks[1][0])
        self.assertEqual(chunks[0][-1]["text"], "p2")

    def test_debug_records_semantic_reason_and_boundary_score(self):
        sentences = [
            {"start": 0.0, "end": 11.0, "text": "s0"},
            {"start": 11.0, "end": 19.0, "text": "s1"},
            {"start": 19.0, "end": 27.0, "text": "s2"},
            {"start": 27.0, "end": 36.0, "text": "s3"},
            {"start": 36.0, "end": 38.0, "text": "s4"},
        ]
        scores = [0.71, 0.52, 0.68, 0.91]
        candidates = {0, 1, 2, 3}
        debug = []

        audio_search._assemble_bounded_chunks(sentences, scores, candidates, self._config(), debug=debug)

        self.assertEqual(debug[0]["reason"], "semantic")
        self.assertEqual(debug[0]["boundary_score"], 0.91)
        self.assertEqual(debug[0]["duration"], 36.0)

    def test_debug_records_forced_reason_when_no_candidate(self):
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 15.0, "text": "b"},
            {"start": 15.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 41.0, "text": "d"},
        ]
        scores = [0.5, 0.5, 0.5]
        debug = []

        audio_search._assemble_bounded_chunks(sentences, scores, set(), self._config(), debug=debug)

        self.assertEqual(debug[0]["reason"], "forced")
        # A forced split still lands on a real boundary with its own score —
        # it just wasn't strong/eligible enough to be picked on its merits.
        self.assertEqual(debug[0]["boundary_score"], scores[2])

    def test_debug_records_end_of_video_when_never_forced_or_semantic(self):
        # Total duration never reaches target, so the chunk just runs out of
        # sentences rather than being cut by either policy.
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 12.0, "text": "b"},
        ]
        scores = [0.5]
        debug = []

        audio_search._assemble_bounded_chunks(sentences, scores, set(), self._config(), debug=debug)

        self.assertEqual(debug[0]["reason"], "end_of_video")

    def test_deterministic_output(self):
        sentences = [
            {"start": 0.0, "end": 11.0, "text": "s0"},
            {"start": 11.0, "end": 19.0, "text": "s1"},
            {"start": 19.0, "end": 27.0, "text": "s2"},
            {"start": 27.0, "end": 36.0, "text": "s3"},
            {"start": 36.0, "end": 38.0, "text": "s4"},
        ]
        scores = [0.71, 0.52, 0.68, 0.91]
        candidates = {0, 1, 2, 3}
        config = self._config()

        first = audio_search._assemble_bounded_chunks(sentences, scores, candidates, config)
        second = audio_search._assemble_bounded_chunks(sentences, scores, candidates, config)

        self.assertEqual(first, second)

    def test_pause_boost_changes_the_chosen_candidate(self):
        # No semantic candidates at all; a long pause between sentence 1 (b)
        # and sentence 2 (c) is the only signal that could end the chunk
        # inside [target, max]. Without it, force-split lands at 39s (c);
        # with it, the pause-boosted boundary after b (28s) wins instead.
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 28.0, "text": "b"},
            {"start": 38.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 50.0, "text": "d"},  # long enough that the
            # trailing tail after either split is >= min_duration, so the
            # tail-merge safety net doesn't also kick in and muddy this
            # comparison -- see test_undersized_trailing_tail_merges_into_previous_chunk
            # for that mechanism on its own.
        ]
        scores = [0.5, 0.5, 0.5]
        config = self._config()
        pauses = audio_search._pause_lengths(sentences)
        boosted = audio_search._pause_boosted_indices(pauses, percentile=75, min_scores=1)

        without_boost = audio_search._assemble_bounded_chunks(sentences, scores, set(), config)
        with_boost = audio_search._assemble_bounded_chunks(
            sentences, scores, set(), config, pause_boosted=boosted
        )

        self.assertEqual(without_boost[0][-1]["end"], 39.0)
        self.assertEqual(with_boost[0][-1]["end"], 28.0)

    def test_pause_boost_never_applies_in_the_growth_zone(self):
        # A boosted boundary at cum=5s (well below target=25) must not end
        # the chunk early just because it is pause-boosted -- mirrors the
        # existing growth-zone regression test for semantic candidates.
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 25.0, "end": 30.0, "text": "b"},
            {"start": 30.0, "end": 60.0, "text": "c"},
        ]
        scores = [0.5, 0.5]
        config = self._config(min_duration_seconds=8, target_duration_seconds=25, max_duration_seconds=40)

        without_boost = audio_search._assemble_bounded_chunks(sentences, scores, set(), config)
        with_early_boost = audio_search._assemble_bounded_chunks(
            sentences, scores, set(), config, pause_boosted={0}
        )

        self.assertEqual(without_boost, with_early_boost)

    def test_debug_flags_pause_boosted_reason(self):
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 28.0, "text": "b"},
            {"start": 38.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 50.0, "text": "d"},
        ]
        scores = [0.5, 0.5, 0.5]
        config = self._config()
        pauses = audio_search._pause_lengths(sentences)
        boosted = audio_search._pause_boosted_indices(pauses, percentile=75, min_scores=1)
        debug = []

        audio_search._assemble_bounded_chunks(
            sentences, scores, set(), config, pause_boosted=boosted, debug=debug
        )

        self.assertTrue(debug[0]["pause_boosted"])

    def test_gated_pause_promotes_a_borderline_semantic_candidate(self):
        # Same scenario as test_pause_boost_changes_the_chosen_candidate, but
        # now the pause-boosted index (1) is also a borderline semantic
        # candidate -- gating must still let it through.
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 28.0, "text": "b"},
            {"start": 38.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 50.0, "text": "d"},
        ]
        scores = [0.5, 0.5, 0.5]
        config = self._config()
        pauses = audio_search._pause_lengths(sentences)
        boosted = audio_search._pause_boosted_indices(pauses, percentile=75, min_scores=1)

        gated = audio_search._assemble_bounded_chunks(
            sentences, scores, set(), config, pause_boosted=boosted,
            borderline_candidates={1},
        )

        self.assertEqual(gated[0][-1]["end"], 28.0)

    def test_gated_pause_does_not_promote_a_boundary_outside_borderline_set(self):
        # Same pause-boosted index (1), but it is NOT in borderline_candidates
        # -- pause alone must not manufacture a boundary anymore. Falls back
        # to the same force-split as with no boost at all.
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 28.0, "text": "b"},
            {"start": 38.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 50.0, "text": "d"},
        ]
        scores = [0.5, 0.5, 0.5]
        config = self._config()
        pauses = audio_search._pause_lengths(sentences)
        boosted = audio_search._pause_boosted_indices(pauses, percentile=75, min_scores=1)

        gated = audio_search._assemble_bounded_chunks(
            sentences, scores, set(), config, pause_boosted=boosted,
            borderline_candidates=set(),
        )
        without_boost = audio_search._assemble_bounded_chunks(sentences, scores, set(), config)

        self.assertEqual(gated, without_boost)
        self.assertEqual(gated[0][-1]["end"], 39.0)

    def test_strong_semantic_candidate_is_eligible_even_outside_borderline_set(self):
        # A strong candidate (in `candidates`) must win regardless of the
        # gating machinery -- gating only constrains the pause path.
        sentences = [
            {"start": 0.0, "end": 11.0, "text": "s0"},
            {"start": 11.0, "end": 19.0, "text": "s1"},
            {"start": 19.0, "end": 27.0, "text": "s2"},
            {"start": 27.0, "end": 36.0, "text": "s3"},
            {"start": 36.0, "end": 38.0, "text": "s4"},
        ]
        scores = [0.71, 0.52, 0.68, 0.91]
        config = self._config()

        chunks = audio_search._assemble_bounded_chunks(
            sentences, scores, {3}, config,
            pause_boosted=set(), borderline_candidates=set(),
        )

        self.assertEqual(chunks[0][-1]["end"], 36.0)

    def test_debug_flags_borderline_semantic(self):
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 28.0, "text": "b"},
            {"start": 38.0, "end": 39.0, "text": "c"},
            {"start": 39.0, "end": 50.0, "text": "d"},
        ]
        scores = [0.5, 0.5, 0.5]
        config = self._config()
        pauses = audio_search._pause_lengths(sentences)
        boosted = audio_search._pause_boosted_indices(pauses, percentile=75, min_scores=1)
        debug = []

        audio_search._assemble_bounded_chunks(
            sentences, scores, set(), config, pause_boosted=boosted,
            borderline_candidates={1}, debug=debug,
        )

        self.assertTrue(debug[0]["borderline_semantic"])
        self.assertTrue(debug[0]["pause_boosted"])


class PauseBoostTests(unittest.TestCase):
    def test_pause_lengths_are_gaps_between_adjacent_sentences(self):
        sentences = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 8.0, "end": 12.0, "text": "b"},
            {"start": 12.5, "end": 15.0, "text": "c"},
        ]

        self.assertEqual(audio_search._pause_lengths(sentences), [3.0, 0.5])

    def test_pause_lengths_empty_for_a_single_sentence(self):
        self.assertEqual(
            audio_search._pause_lengths([{"start": 0.0, "end": 5.0, "text": "a"}]), []
        )

    def test_pause_boosted_indices_selects_the_long_pauses(self):
        pauses = [0.1, 0.2, 5.0, 0.1, 4.5, 0.2]

        boosted = audio_search._pause_boosted_indices(pauses, percentile=75, min_scores=4)

        self.assertEqual(boosted, {2, 4})

    def test_too_few_pauses_yields_no_boosted_indices(self):
        pauses = [0.1, 5.0, 0.2]

        self.assertEqual(
            audio_search._pause_boosted_indices(pauses, percentile=75, min_scores=4), set()
        )

    def test_no_pauses_yields_no_boosted_indices(self):
        self.assertEqual(audio_search._pause_boosted_indices([], percentile=75, min_scores=1), set())


class SentenceToPassageTests(unittest.TestCase):
    def test_timestamps_and_text_are_derived_from_the_sentence_span(self):
        sentences = [
            {"start": 5.0, "end": 10.0, "text": "a"},
            {"start": 10.0, "end": 16.0, "text": "b"},
        ]

        passage = audio_search._sentences_to_passage("L21_V001", sentences)

        self.assertEqual(passage.start_sec, 5.0)
        self.assertEqual(passage.end_sec, 16.0)
        self.assertEqual(passage.anchor_sec, 10.5)
        self.assertEqual(passage.passage_id, "L21_V001:5000")
        self.assertEqual(passage.text, "a b")


class BoundedSemanticPassageTests(unittest.TestCase):
    def test_empty_rows_produce_no_passages(self):
        self.assertEqual(make_bounded_semantic_passages("L21_V001", [], lambda texts: []), [])

    def test_single_sentence_transcript_never_calls_embed_cues(self):
        rows = [{"offset": 0.0, "token": " một câu duy nhất."}]

        def embed_cues(texts):
            raise AssertionError("embed_cues should not be called for a single sentence")

        passages = make_bounded_semantic_passages("L21_V001", rows, embed_cues)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].text, "một câu duy nhất.")

    def test_too_few_sentences_fall_back_to_duration_only_chunking(self):
        # Three sentences -> 2 boundary scores, below the default
        # min_scores_for_percentile=4. Even maximally different embeddings
        # must not split this into more than one passage on their own.
        rows = [
            {"offset": 0.0, "token": " một."},
            {"offset": 10.0, "token": " hai."},
            {"offset": 20.0, "token": " ba."},
        ]

        def embed_cues(texts):
            self.assertEqual(len(texts), 3)
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        passages = make_bounded_semantic_passages("L21_V001", rows, embed_cues)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].text, "một. hai. ba.")

    def test_config_json_round_trip(self):
        config = SemanticChunkConfig(
            context_window_sentences=2,
            breakpoint_percentile=20.0,
            min_scores_for_percentile=5,
            min_duration_seconds=6.0,
            target_duration_seconds=20.0,
            max_duration_seconds=35.0,
            overlap_sentences=2,
        )

        self.assertEqual(SemanticChunkConfig.from_json(config.to_json()), config)

    def test_from_json_fills_missing_keys_with_defaults(self):
        self.assertEqual(SemanticChunkConfig.from_json({}), SemanticChunkConfig())

    def test_pause_boost_is_disabled_by_default(self):
        self.assertFalse(SemanticChunkConfig().pause_boost_enabled)

    def test_pause_fields_round_trip_through_json(self):
        config = SemanticChunkConfig(
            pause_boost_enabled=True, pause_relative_percentile=80.0,
            pause_borderline_semantic_percentile=35.0,
        )

        self.assertEqual(SemanticChunkConfig.from_json(config.to_json()), config)

    def test_pause_borderline_semantic_percentile_default(self):
        self.assertEqual(SemanticChunkConfig().pause_borderline_semantic_percentile, 30.0)

    def test_pause_boost_enabled_runs_end_to_end_without_error(self):
        rows = [
            {"offset": 0.0, "token": " một."},
            {"offset": 10.0, "token": " hai."},
            {"offset": 25.0, "token": " ba."},
            {"offset": 45.0, "token": " bốn."},
        ]

        def embed_cues(texts):
            return [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]

        config = SemanticChunkConfig(
            pause_boost_enabled=True, min_duration_seconds=5,
            target_duration_seconds=10, max_duration_seconds=40,
        )

        passages = make_bounded_semantic_passages("L21_V001", rows, embed_cues, config=config)

        self.assertTrue(passages)
        self.assertEqual(sum(len(p.text) for p in passages) > 0, True)

    def test_pause_boost_always_gates_by_borderline_semantic_in_production(self):
        # A pause-ablation benchmark found the ungated form (pause alone
        # making a boundary eligible) lets pause dominate the large majority
        # of boundaries and reads as a false topic transition on manual
        # review roughly half the time. The only production entry point,
        # make_bounded_semantic_passages, must never call the assembler with
        # pause_boosted set but borderline_candidates left None.
        rows = [
            {"offset": 0.0, "token": " một."},
            {"offset": 10.0, "token": " hai."},
            {"offset": 25.0, "token": " ba."},
            {"offset": 45.0, "token": " bốn."},
        ]

        def embed_cues(texts):
            return [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]

        config = SemanticChunkConfig(
            pause_boost_enabled=True, min_duration_seconds=5,
            target_duration_seconds=10, max_duration_seconds=40,
        )

        with patch.object(
            audio_search, "_assemble_bounded_chunks",
            wraps=audio_search._assemble_bounded_chunks,
        ) as spy:
            make_bounded_semantic_passages("L21_V001", rows, embed_cues, config=config)

        _, kwargs = spy.call_args
        self.assertIsNotNone(kwargs.get("pause_boosted"))
        self.assertIsNotNone(kwargs.get("borderline_candidates"))


class FusionTests(unittest.TestCase):
    def test_rrf_uses_rank_not_incompatible_source_scores(self):
        fused = reciprocal_rank_fusion(
            [
                [("A", 0.01), ("B", 0.001)],
                [("B", 99_000), ("A", 98_000)],
            ],
            rank_constant=0,
        )

        self.assertEqual([key for key, _ in fused], ["A", "B"])
        self.assertAlmostEqual(dict(fused)["A"], 1.5)
        self.assertAlmostEqual(dict(fused)["B"], 1.5)

    def test_overlapping_hits_merge_but_distant_hits_remain(self):
        hits = [
            AudioHit("L21_V015", 840, 900, 870, "first", 1.0),
            AudioHit("L21_V015", 855, 915, 885, "better", 2.0),
            AudioHit("L21_V015", 1100, 1160, 1130, "later", 0.5),
        ]

        merged = merge_temporal_hits(hits)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].text, "better")
        self.assertEqual(merged[0].start_sec, 840)
        self.assertEqual(merged[0].end_sec, 915)


class ElasticsearchContractTests(unittest.TestCase):
    def test_index_mapping_supports_accented_text_and_2560_dimension_vectors(self):
        definition = build_index_definition()

        folded = definition["settings"]["analysis"]["analyzer"]["vi_folded"]
        embedding = definition["mappings"]["properties"]["embedding"]
        text = definition["mappings"]["properties"]["text"]
        self.assertEqual(folded["filter"], ["lowercase", "asciifolding"])
        self.assertEqual(text["fields"]["folded"]["analyzer"], "vi_folded")
        self.assertEqual(embedding["type"], "dense_vector")
        self.assertEqual(embedding["dims"], 2560)
        self.assertEqual(embedding["similarity"], "cosine")

    def test_basic_license_requests_bm25_and_knn_separately(self):
        bm25 = build_bm25_request("nghiên cứu cực quang", k=20)
        knn = build_knn_request([0.0] * 1024, k=20)

        should = bm25["query"]["bool"]["should"]
        best_fields_clauses = [c["multi_match"] for c in should if c["multi_match"]["type"] == "best_fields"]
        phrase = next(c["multi_match"] for c in should if c["multi_match"]["type"] == "phrase")
        best_fields = next(c for c in best_fields_clauses if "fuzziness" not in c)
        fuzzy = next(c for c in best_fields_clauses if "fuzziness" in c)
        self.assertEqual(best_fields["fields"], ["text^2", "text.folded"])
        self.assertEqual(phrase["fields"], ["text^4", "text.folded^2"])
        self.assertEqual(phrase["slop"], 2)
        self.assertEqual(fuzzy["fields"], ["text", "text.folded"])
        self.assertEqual(fuzzy["fuzziness"], "AUTO")
        self.assertEqual(fuzzy["prefix_length"], 1)
        self.assertEqual(bm25["query"]["bool"]["minimum_should_match"], 1)
        self.assertEqual(bm25["size"], 100)
        self.assertEqual(knn["knn"]["field"], "embedding")
        self.assertEqual(knn["knn"]["k"], 100)
        self.assertEqual(knn["size"], 100)
        self.assertEqual(build_knn_request([0.0] * 1024, k=10_000)["knn"]["num_candidates"], 10_000)
        self.assertNotIn("retriever", bm25)
        self.assertNotIn("retriever", knn)

    def test_transcript_path_serves_raw_asr_output_by_default(self):
        from backend import audio_search as module

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "step_data.parquet"
            clean = Path(directory) / "step_data_clean.parquet"
            raw.write_bytes(b"")
            clean.write_bytes(b"")
            with patch.object(module, "RAW_TRANSCRIPT_PATH", raw), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AIC_TRANSCRIPT_PATH", None)
                # A cleaned parquet sitting on disk must not change what is
                # served — writing a file is not deploying it.
                self.assertEqual(module.transcript_path(), raw)

    def test_a_cleaned_transcript_is_adopted_only_when_named_explicitly(self):
        from backend import audio_search as module

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "step_data.parquet"
            clean = Path(directory) / "generated" / "step_data_clean.parquet"
            with patch.object(module, "RAW_TRANSCRIPT_PATH", raw), \
                 patch.dict(os.environ, {"AIC_TRANSCRIPT_PATH": str(clean)}):
                self.assertEqual(module.transcript_path(), clean)

    def test_search_parses_and_merges_overlapping_elasticsearch_hits(self):
        class Client:
            calls = 0

            def search(self, **_kwargs):
                self.calls += 1
                return {
                    "hits": {
                        "hits": [
                            {
                                "_score": 2.0,
                                "_source": {
                                    "video_id": "L21_V015",
                                    "start_sec": 840,
                                    "end_sec": 900,
                                    "anchor_sec": 870,
                                    "text": "chuyến bay tư nhân",
                                },
                            },
                            {
                                "_score": 1.0,
                                "_source": {
                                    "video_id": "L21_V015",
                                    "start_sec": 855,
                                    "end_sec": 915,
                                    "anchor_sec": 885,
                                    "text": "ánh sáng cực quang",
                                },
                            },
                        ]
                    }
                }

        hits = search_audio(
            "private spacecraft aurora",
            k=10,
            client=Client(),
            embed=lambda _text: [0.0] * 1024,
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].bm25_rank, 1)
        self.assertEqual(hits[0].vector_rank, 1)
        self.assertEqual(hits[0].video_id, "L21_V015")
        self.assertEqual((hits[0].start_sec, hits[0].end_sec), (840, 915))

    def test_malformed_search_hit_is_ignored(self):
        response = {"hits": {"hits": [{"_score": 1.0, "_source": {"video_id": "L21"}}]}}

        self.assertEqual(parse_search_response(response), [])

    def test_semantic_search_prefers_exact_flat_index(self):
        import numpy as np

        class Flat:
            ntotal = 2

            def search(self, _query, _k):
                return np.asarray([[0.9, 0.8]]), np.asarray([[1, 0]])

        metadata = [
            {"passage_id": "a", "video_id": "L21_V001", "start_sec": 0,
             "end_sec": 10, "anchor_sec": 5, "text": "first"},
            {"passage_id": "b", "video_id": "L21_V002", "start_sec": 20,
             "end_sec": 30, "anchor_sec": 25, "text": "second"},
        ]
        with patch.object(audio_search, "load_audio_flat", return_value=(Flat(), metadata)):
            hits = audio_search.search_audio_semantic(
                "query", k=2, embed=lambda _text: [1.0, 0.0],
                client=object(),
            )

        self.assertEqual([hit.passage_id for hit in hits], ["b", "a"])
        self.assertEqual([hit.vector_rank for hit in hits], [1, 2])


if __name__ == "__main__":
    unittest.main()
