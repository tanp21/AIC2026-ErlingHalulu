import math
import random
import unittest

import numpy as np

from backend.subquery_fusion import (
    Hit, combine_vectors, shot_ranks, weighted_shot_rrf, weighted_shot_score,
)


def hits(*shots, video="V", scores=None):
    """Ranked hit list for shot indices in order; optional descending scores."""
    return [Hit((video, s), payload=(video, s, pos),
                score=None if scores is None else scores[pos])
            for pos, s in enumerate(shots)]


def order(results):
    return [g.group_id for g in results]


def descending(n, start=0.9, step=0.01):
    return [start - step * i for i in range(n)]


class WeightedRrfInvariantTests(unittest.TestCase):
    def test_p1_single_list_matches_shot_ranks(self):
        # Frame hits with repeated shots: fusion of one list is plain shot dedup.
        lst = hits(3, 3, 1, 4, 1, 2)
        expected = [g for g, _ in sorted(shot_ranks(lst).items(), key=lambda kv: kv[1][0])]
        self.assertEqual(order(weighted_shot_rrf([lst], [1.0], k=60)), expected)

    def test_p2_scale_invariance(self):
        lists = [hits(1, 2, 3), hits(2, 3, 4), hits(4, 1)]
        weights = [0.5, 0.25, 0.25]
        base = weighted_shot_rrf(lists, weights, k=10)
        for c in (2.0, 0.5, 4.0):  # exact binary scalings, no float tie flips
            scaled = weighted_shot_rrf(lists, [w * c for w in weights], k=10)
            self.assertEqual(order(scaled), order(base))
            self.assertEqual([g.rep_hit for g in scaled], [g.rep_hit for g in base])

    def test_p3_identical_lists_equal_single_list(self):
        lst = hits(5, 2, 2, 7, 1)
        single = order(weighted_shot_rrf([lst], [1.0], k=60))
        for weights in ([1.0, 1.0, 1.0], [0.5, 0.25, 0.25], [0.2, 0.7, 0.1]):
            self.assertEqual(order(weighted_shot_rrf([lst, lst, lst], weights, k=60)), single)

    def test_p4_permutation_invariance_with_equal_weights(self):
        rng = random.Random(7)
        for _ in range(100):
            lists = [hits(*rng.sample(range(10), rng.randint(1, 10))) for _ in range(4)]
            base = weighted_shot_rrf(lists, [0.25] * 4, k=10)
            perm = lists[:]
            rng.shuffle(perm)
            shuffled = weighted_shot_rrf(perm, [0.25] * 4, k=10)
            self.assertEqual(order(shuffled), order(base))
            self.assertEqual([g.score for g in shuffled], [g.score for g in base])

    def test_p5_monotone_in_rank(self):
        rng = random.Random(11)
        g = ("V", 9)
        for _ in range(200):
            others = rng.sample(range(9), rng.randint(1, 6))
            pos_worse = rng.randint(1, len(others))
            pos_better = rng.randint(0, pos_worse - 1)
            list1_worse = others[:pos_worse] + [9] + others[pos_worse:]
            list1_better = others[:pos_better] + [9] + others[pos_better:]
            list2 = hits(*rng.sample(range(10), rng.randint(1, 10)))
            worse = weighted_shot_rrf([hits(*list1_worse), list2], [0.5, 0.5], k=10)
            better = weighted_shot_rrf([hits(*list1_better), list2], [0.5, 0.5], k=10)
            w = {x.group_id: x for x in worse}
            b = {x.group_id: x for x in better}
            self.assertGreater(b[g].score, w[g].score)
            # Against any group whose per-list ranks did not change, g never falls behind.
            for gid in w:
                if gid == g or gid not in b or w[gid].ranks != b[gid].ranks:
                    continue
                if order(worse).index(g) < order(worse).index(gid):
                    self.assertLess(order(better).index(g), order(better).index(gid))

    def test_p6_bound(self):
        k = 60
        lists = [hits(1, 2), hits(1, 3), hits(1)]
        weights = [0.5, 0.25, 0.25]
        results = weighted_shot_rrf(lists, weights, k=k)
        bound = sum(weights) / (k + 1)
        for g in results:
            self.assertLessEqual(g.score, bound + 1e-15)
        top = {g.group_id: g.score for g in results}[("V", 1)]
        self.assertEqual(top, math.fsum(w / (k + 1) for w in weights))  # rank 1 everywhere

    def test_p7_canonical_output(self):
        lists = [hits(1, 1, 2, 3, 2), hits(3, 3, 4)]
        results = weighted_shot_rrf(lists, [0.5, 0.5], k=60)
        ids = order(results)
        self.assertEqual(len(ids), len(set(ids)))
        for g in results:
            self.assertEqual(g.rep_hit.group_id, g.group_id)
            self.assertTrue(any(r is not None for r in g.ranks))

    def test_p8_caption_wins_exact_tie_under_cap50(self):
        k = 60
        caption = hits(1)       # g1 = shot 1, only in caption
        para_a = hits(2)        # g2 = shot 2, only in both paraphrases
        para_b = hits(2)
        results = weighted_shot_rrf([caption, para_a, para_b], [0.5, 0.25, 0.25], k=k)
        scores = {g.group_id: g.score for g in results}
        self.assertEqual(scores[("V", 1)], scores[("V", 2)])  # exact tie: 0.5/(k+1)
        self.assertEqual(order(results)[0], ("V", 1))

    def test_p9_zero_weight_list_is_ignored(self):
        lists = [hits(1, 2), hits(3, 4)]
        with_zero = weighted_shot_rrf(lists + [hits(7, 8, 1)], [0.6, 0.4, 0.0], k=10)
        without = weighted_shot_rrf(lists, [0.6, 0.4], k=10)
        self.assertEqual(order(with_zero), order(without))
        self.assertEqual([g.score for g in with_zero], [g.score for g in without])
        self.assertNotIn(("V", 7), order(with_zero))

    def test_p10_deterministic_with_ties(self):
        lists = [hits(1, 2), hits(2, 1)]  # shots 1 and 2 tie exactly under equal weights
        runs = {tuple(order(weighted_shot_rrf(lists, [0.5, 0.5], k=10))) for _ in range(20)}
        self.assertEqual(len(runs), 1)
        self.assertEqual(list(next(iter(runs))), [("V", 1), ("V", 2)])  # str(group_id) tie-break

    def test_rejects_bad_weights_and_k(self):
        with self.assertRaises(ValueError):
            weighted_shot_rrf([hits(1)], [1.0, 1.0])
        with self.assertRaises(ValueError):
            weighted_shot_rrf([hits(1)], [-0.1])
        with self.assertRaises(ValueError):
            weighted_shot_rrf([hits(1)], [1.0], k=0)

    def test_empty_input_returns_empty(self):
        self.assertEqual(weighted_shot_rrf([], [], k=60), [])


class WeightedScoreInvariantTests(unittest.TestCase):
    def test_s1_single_list_matches_shot_ranks(self):
        lst = hits(3, 3, 1, 4, 1, 2, scores=[0.9, 0.9, 0.8, 0.8, 0.7, 0.6])
        expected = [g for g, _ in sorted(shot_ranks(lst).items(), key=lambda kv: kv[1][0])]
        self.assertEqual(order(weighted_shot_score([lst], [1.0])), expected)

    def test_s2_absence_never_beats_presence(self):
        a = hits(1, 2, 3, scores=[0.9, 0.5, 0.4])
        b = hits(4, 5, scores=[0.8, 0.3])
        results = {g.group_id: g for g in weighted_shot_score([a, b], [0.5, 0.5])}
        # shot 3 is present in a at 0.4 (a's lowest) and absent from b (fill = 0.3, b's lowest)
        self.assertAlmostEqual(results[("V", 3)].score, 0.5 * 0.4 + 0.5 * 0.3)
        # fill is b's minimum, never above any real score from b
        self.assertLessEqual(0.3, min(h.score for h in b))

    def test_s3_scale_identical_canonical_zero_weight(self):
        a = hits(1, 2, 2, 3, scores=[0.9, 0.8, 0.8, 0.7])
        b = hits(3, 1, scores=[0.85, 0.6])
        base = weighted_shot_score([a, b], [0.5, 0.5])
        self.assertEqual(order(weighted_shot_score([a, b], [1.0, 1.0])), order(base))
        single = order(weighted_shot_score([a], [1.0]))
        self.assertEqual(order(weighted_shot_score([a, a, a], [0.2, 0.3, 0.5])), single)
        ids = order(base)
        self.assertEqual(len(ids), len(set(ids)))
        for g in base:
            self.assertEqual(g.rep_hit.group_id, g.group_id)
        c = hits(9, scores=[0.99])
        with_zero = weighted_shot_score([a, b, c], [0.5, 0.5, 0.0])
        self.assertEqual(order(with_zero), order(base))

    def test_requires_scores(self):
        with self.assertRaises(ValueError):
            weighted_shot_score([hits(1, 2)], [1.0])


class CombineVectorsTests(unittest.TestCase):
    def test_e1_identical_vectors_return_that_vector(self):
        v = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        for weights in ([1, 1, 1], [0.5, 0.25, 0.25], [3, 1, 2]):
            np.testing.assert_allclose(combine_vectors([v, v, v], weights), v, atol=1e-6)

    def test_e2_unit_norm_and_weight_scale_invariance(self):
        rng = np.random.default_rng(0)
        vs = rng.normal(size=(4, 16)).astype(np.float32)
        out = combine_vectors(vs, [0.4, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)
        np.testing.assert_allclose(combine_vectors(vs, [4, 2, 2, 2]), out, atol=1e-6)

    def test_e3_one_hot_weight_returns_first_vector(self):
        rng = np.random.default_rng(1)
        vs = rng.normal(size=(3, 8)).astype(np.float32)
        expected = vs[0] / np.linalg.norm(vs[0])
        np.testing.assert_allclose(combine_vectors(vs, [1, 0, 0]), expected, atol=1e-6)

    def test_e4_invalid_inputs_raise(self):
        v = np.ones((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            combine_vectors(v, [0, 0])
        with self.assertRaises(ValueError):
            combine_vectors(v, [1, -1])
        with self.assertRaises(ValueError):
            combine_vectors(np.zeros((1, 3)), [1])
        with self.assertRaises(ValueError):
            combine_vectors(v, [1])


if __name__ == "__main__":
    unittest.main()
