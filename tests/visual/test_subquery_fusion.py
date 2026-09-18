import random
import unittest

from backend.subquery_fusion import Adjacency, FusedGroup, Hit, dedup_hits, fuse


def chain(video, n):
    """Linear shot chain (video, 1)..(video, n) with prev/next adjacency."""
    prev, nxt = {}, {}
    for i in range(1, n + 1):
        gid = (video, i)
        prev[gid] = (video, i - 1) if i > 1 else None
        nxt[gid] = (video, i + 1) if i < n else None
    return Adjacency(prev=prev, next=nxt)


def by_group(results):
    return {g.group_id: g for g in results}


class ScoringInvariantTests(unittest.TestCase):
    """Each test corresponds to an invariant (I1-I10) from the plan's scoring definition."""

    def test_i1_coverage_bound_mean(self):
        adj = chain("V", 5)
        aspects = [
            [Hit(("V", 1)), Hit(("V", 2))],
            [Hit(("V", 1)), Hit(("V", 3))],
            [Hit(("V", 2))],
        ]
        k = 60
        out = fuse([], aspects, adj, lam=0.0, k=k, agg="mean", beta=1.0)
        bound = 1.0 / (k + 1)
        self.assertTrue(out)
        for g in out:
            self.assertGreater(g.score, 0)
            self.assertLessEqual(g.score, bound + 1e-9)

    def test_i1_coverage_bound_top_m_mean(self):
        adj = chain("V", 5)
        aspects = [[Hit(("V", 1))], [Hit(("V", 1))], [Hit(("V", 1))], [Hit(("V", 2))]]
        k = 30
        out = fuse([], aspects, adj, lam=0.0, k=k, agg="top_m_mean", m=2, beta=1.0)
        bound = 1.0 / (k + 1)
        for g in out:
            self.assertLessEqual(g.score, bound + 1e-9)

    def test_i2_i3_neighbor_never_summed_and_capped(self):
        """Aspect A hits shot1 directly; aspect B only hits shot2, so shot1's credit from
        B must come from gamma-discounted neighbor evidence, never from summing with A."""
        adj = chain("V", 3)
        aspects = [[Hit(("V", 1))], [Hit(("V", 2))]]
        k = 60
        gamma = 1.0
        out = by_group(fuse([], aspects, adj, lam=0.0, k=k, agg="mean", gamma=gamma, beta=1.0))
        g = out[("V", 1)]
        direct = 1.0 / (k + 1)
        expected = (direct + gamma * direct) / 2  # mean of [e_A=direct, e_B=gamma*neighbor]
        self.assertAlmostEqual(g.score, expected, places=9)
        self.assertLessEqual(g.score, direct + 1e-9)  # never exceeds the per-query cap combined

        # gamma = 0 must equal neighbor-off exactly.
        out0 = by_group(fuse([], aspects, adj, lam=0.0, k=k, agg="mean", gamma=0.0, beta=1.0))
        self.assertAlmostEqual(out0[("V", 1)].score, direct / 2, places=9)

    def test_i4_neighbor_only_groups_never_become_candidates(self):
        adj = chain("V", 3)
        out = fuse([], [[Hit(("V", 2))]], adj, lam=0.0, k=60, gamma=1.0, beta=1.0)
        group_ids = {g.group_id for g in out}
        self.assertEqual(group_ids, {("V", 2)})  # shots 1 and 3 (neighbors) are absent

    def test_i4_output_groups_unique(self):
        adj = chain("V", 4)
        holistic = [Hit(("V", 1)), Hit(("V", 2))]
        aspects = [[Hit(("V", 2)), Hit(("V", 1))], [Hit(("V", 3))]]
        out = fuse(holistic, aspects, adj, lam=0.5, k=60)
        ids = [g.group_id for g in out]
        self.assertEqual(len(ids), len(set(ids)))

    def test_i5_holistic_absent_cap_and_tiebreak(self):
        adj = chain("V", 5)
        k = 60
        # h1: holistic rank-1, no aspect evidence. g: no holistic hit, 2 perfect aspects.
        holistic = [Hit(("V", 1))]
        aspects = [[Hit(("V", 3))], [Hit(("V", 3))]]
        bound_unit = 1.0 / (k + 1)

        for lam, expect_first in ((0.75, ("V", 1)), (0.25, ("V", 3))):
            out = fuse(holistic, aspects, adj, lam=lam, k=k, agg="mean", beta=1.0)
            groups = by_group(out)
            g = groups[("V", 3)]
            self.assertLessEqual(g.score, (1 - lam) * 1.0 * bound_unit + 1e-9)
            self.assertEqual(out[0].group_id, expect_first)

        # Exact tie at lam=0.5: both score (1-lam)*bound_unit == lam*bound_unit, tie-break
        # must favor the holistic-supported group.
        out = fuse(holistic, aspects, adj, lam=0.5, k=k, agg="mean", beta=1.0)
        groups = by_group(out)
        self.assertAlmostEqual(groups[("V", 1)].score, groups[("V", 3)].score, places=9)
        self.assertEqual(out[0].group_id, ("V", 1))

    def test_i6_beta_monotone(self):
        adj = chain("V", 5)
        k = 60
        holistic = [Hit(("V", 1))]  # shot1's direct rank (1) is <= T_h=2: strong support
        aspects = [[Hit(("V", 1)), Hit(("V", 3))], [Hit(("V", 3))]]
        prev_weak = None
        for beta in (1.0, 0.5, 0.0):
            out = by_group(fuse(holistic, aspects, adj, lam=0.5, k=k, T_h=2, beta=beta))
            strong = out[("V", 1)]
            self.assertAlmostEqual(strong.support, 1.0)
            weak = out.get(("V", 3))
            weak_score = weak.score if weak else 0.0
            if prev_weak is not None:
                self.assertLessEqual(weak_score, prev_weak + 1e-9)
            prev_weak = weak_score

    def test_i7_lam1_equals_holistic_only(self):
        adj = chain("V", 4)
        holistic = [Hit(("V", 2)), Hit(("V", 1))]
        aspects = [[Hit(("V", 3))], [Hit(("V", 1))]]
        out = fuse(holistic, aspects, adj, lam=1.0, k=60, beta=1.0)
        self.assertEqual([g.group_id for g in out], [("V", 2), ("V", 1)])  # H's own order
        for g in out:
            self.assertIsNotNone(g.holistic_rank)

    def test_i7_lam0_equals_aspect_ranking(self):
        adj = chain("V", 4)
        aspects = [[Hit(("V", 1)), Hit(("V", 3))], [Hit(("V", 3))]]
        out = fuse([], aspects, adj, lam=0.0, k=60, agg="mean", beta=1.0)
        # shot3 hit by both aspects (rank1 + rank2), shot1 hit by one (rank1) -> shot3 first.
        self.assertEqual(out[0].group_id, ("V", 3))

    def test_i8_rescue_with_soft_beta_and_hard_gate_drops_it(self):
        adj = chain("V", 5)
        k = 60
        aspects = [[Hit(("V", 4))], [Hit(("V", 4))]]  # both rank at shot4, H never hits it
        rescued = fuse([], aspects, adj, lam=0.5, k=k, T_h=0, beta=0.3)
        self.assertIn(("V", 4), {g.group_id for g in rescued})

        gated = fuse([], aspects, adj, lam=0.5, k=k, T_h=0, beta=0.0)
        self.assertNotIn(("V", 4), {g.group_id for g in gated})

    def test_i9_fallback_time_buckets_are_generic(self):
        # No shots at all: group ids are (video, "t", bucket_idx) time buckets.
        adj = Adjacency(
            prev={("V", "t", 1): ("V", "t", 0), ("V", "t", 0): None},
            next={("V", "t", 0): ("V", "t", 1), ("V", "t", 1): None},
        )
        holistic = [Hit(("V", "t", 0))]
        aspects = [[Hit(("V", "t", 1))]]
        out = fuse(holistic, aspects, adj, lam=0.5, k=60, gamma=1.0)
        self.assertTrue(out)  # runs with no special-casing for bucket-shaped group ids

    def test_i10_no_aspects_equals_holistic_ranking(self):
        adj = chain("V", 3)
        holistic = [Hit(("V", 2)), Hit(("V", 1))]
        out = fuse(holistic, [], adj, lam=0.5, k=60)
        self.assertEqual([g.group_id for g in out], [("V", 2), ("V", 1)])


class FusionBehaviorTests(unittest.TestCase):
    def test_deterministic_tiebreak_prefers_holistic_then_aspect_rank_then_group_id(self):
        adj = chain("V", 5)
        # Two groups, equal final score (both zero holistic, equal aspect evidence),
        # but different best aspect rank should decide the order.
        aspects = [[Hit(("V", 1)), Hit(("V", 2))]]  # shot1 rank1, shot2 rank2
        out = fuse([], aspects, adj, lam=0.0, k=60, beta=1.0)
        self.assertEqual([g.group_id for g in out], [("V", 1), ("V", 2)])

    def test_top_m_mean_with_fewer_aspects_than_m(self):
        adj = chain("V", 2)
        aspects = [[Hit(("V", 1))], [Hit(("V", 1))]]
        out = by_group(fuse([], aspects, adj, lam=0.0, k=60, agg="top_m_mean", m=5, beta=1.0))
        direct = 1.0 / 61
        self.assertAlmostEqual(out[("V", 1)].score, direct, places=9)

    def test_dedup_hits_drops_close_in_time_same_video_hits(self):
        hits = [Hit(("V", 1), time=0.0), Hit(("V", 2), time=1.0), Hit(("V", 3), time=10.0)]
        kept = dedup_hits(hits, window_seconds=2.0, video_of=lambda g: g[0])
        self.assertEqual([h.group_id for h in kept], [("V", 1), ("V", 3)])

    def test_dedup_seconds_changes_effective_rank_in_fuse(self):
        adj = chain("V", 3)
        aspects = [[Hit(("V", 1), time=0.0), Hit(("V", 2), time=1.0)]]
        without = by_group(fuse([], aspects, adj, lam=0.0, k=60, dedup_seconds=0.0))
        self.assertIn(("V", 2), without)
        with_dedup = by_group(fuse([], aspects, adj, lam=0.0, k=60, dedup_seconds=2.0))
        self.assertNotIn(("V", 2), with_dedup)  # shot2 hit was within 2s of shot1's, dropped

    def test_neighbor_scope_aspects_vs_all_support_uses_direct_rank_regardless(self):
        adj = chain("V", 3)
        holistic = [Hit(("V", 2))]  # holistic directly hits shot2, not shot1
        aspects = [[Hit(("V", 1))]]

        scoped = by_group(fuse(holistic, aspects, adj, lam=0.5, k=60, gamma=1.0,
                               neighbor_scope="aspects", T_h=0, beta=0.3))
        self.assertEqual(scoped[("V", 1)].holistic_rank, None)
        self.assertEqual(scoped[("V", 1)].support, 0.3)  # weak support: no direct H hit
        self.assertFalse(scoped[("V", 1)].used_neighbor["holistic"])

        all_scope = by_group(fuse(holistic, aspects, adj, lam=0.5, k=60, gamma=1.0,
                                  neighbor_scope="all", T_h=0, beta=0.3))
        self.assertEqual(all_scope[("V", 1)].holistic_rank, None)
        self.assertEqual(all_scope[("V", 1)].support, 0.3)  # support unaffected by neighbor_scope
        self.assertTrue(all_scope[("V", 1)].used_neighbor["holistic"])
        self.assertGreater(all_scope[("V", 1)].score, scoped[("V", 1)].score)

    def test_rejects_out_of_range_gamma_beta_and_bad_neighbor_scope(self):
        adj = chain("V", 2)
        with self.assertRaises(ValueError):
            fuse([], [], adj, lam=0.5, gamma=1.5)
        with self.assertRaises(ValueError):
            fuse([], [], adj, lam=0.5, beta=-0.1)
        with self.assertRaises(ValueError):
            fuse([], [], adj, lam=0.5, neighbor_scope="bogus")


class PropertyStyleTests(unittest.TestCase):
    """Randomized checks for I1, I2, I4 and I6 across many small synthetic scenarios."""

    def test_random_scenarios_hold_invariants(self):
        rng = random.Random(1234)
        k = 60
        bound = 1.0 / (k + 1)
        for _ in range(200):
            n_shots = rng.randint(2, 8)
            adj = chain("V", n_shots)
            n_aspects = rng.randint(0, 4)
            aspects = []
            for _ in range(n_aspects):
                n_hits = rng.randint(0, n_shots)
                shots = rng.sample(range(1, n_shots + 1), n_hits)
                aspects.append([Hit(("V", s)) for s in shots])
            has_holistic = rng.random() < 0.7
            holistic = [Hit(("V", s)) for s in
                       rng.sample(range(1, n_shots + 1), rng.randint(0, n_shots))] if has_holistic else []
            gamma = rng.choice([0.0, 0.3, 1.0])
            lam = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
            beta = rng.choice([0.0, 0.5, 1.0])
            agg = rng.choice(["mean", "top_m_mean"])

            out = fuse(holistic, aspects, adj, lam=lam, k=k, agg=agg, m=2, gamma=gamma,
                      beta=beta)

            # I4: unique group ids, every output group has >=1 direct hit somewhere.
            ids = [g.group_id for g in out]
            self.assertEqual(len(ids), len(set(ids)))
            direct_hit_ids = {h.group_id for h in holistic}
            for asp in aspects:
                direct_hit_ids |= {h.group_id for h in asp}
            for gid in ids:
                self.assertIn(gid, direct_hit_ids)

            # I1: bounded aspect contribution -> at lam=0, beta=1, score IS a(g).
            if lam == 0.0 and beta == 1.0 and aspects:
                for g in out:
                    self.assertLessEqual(g.score, bound + 1e-9)

            # I6: with beta=0, no group lacking direct holistic support can appear unless
            # lam contributes something (final relies only on lam*h(g) then, since support=0).
            if beta == 0.0:
                for g in out:
                    if g.holistic_rank is None:
                        self.assertLessEqual(g.score, lam * bound + 1e-9)


if __name__ == "__main__":
    unittest.main()
