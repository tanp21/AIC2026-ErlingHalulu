import unittest

from scripts.benchmarks.paraphrase_validation import (
    check_framing, check_protected, deterministic_checks, protected_tokens, run_validation_rounds,
    token_edit_distance,
)


def word_tokens(text):
    return len(text.split()) + 2


SOURCE = "A man in a red hat rides a bike"
VALID = [
    "A bike is ridden by a man in a red hat",
    "A man wearing a red hat is riding a bicycle",
    "Wearing a red hat, a man rides a bicycle",
    "A bicycle is being ridden by a man wearing a red hat",
]


class ProtectedTokenTests(unittest.TestCase):
    def test_digit_and_word_counts_are_equivalent(self):
        self.assertEqual(check_protected("Three cyclists cross the line",
                                         "The line is crossed by 3 cyclists"), [])

    def test_color_equivalence_map(self):
        self.assertEqual(check_protected("A golden dragon spins", "A gold dragon is spinning"), [])
        self.assertEqual(check_protected("A grey cat sleeps", "A gray cat is sleeping"), [])

    def test_changed_color_fails(self):
        self.assertTrue(check_protected("A man in a red hat", "A man in a blue hat"))

    def test_changed_count_fails(self):
        self.assertTrue(check_protected("Four drivers wait", "Five drivers are waiting"))

    def test_dropped_proper_noun_fails_and_moved_one_passes(self):
        source = "A club named FANA gives gifts in Hanoi"
        self.assertTrue(check_protected(source, "A club gives gifts in Hanoi"))
        self.assertEqual(check_protected(source, "In Hanoi, gifts are given by a club named FANA"), [])

    def test_changed_quoted_text_fails(self):
        self.assertTrue(check_protected('A sign reads "OPEN NOW"', 'A sign reads "OPEN TODAY"'))

    def test_sentence_initial_capital_is_not_a_proper_noun(self):
        tokens = protected_tokens("Dragons dance near the Temple. Crowds watch.")
        self.assertEqual(tokens["proper"], {"Temple"})

    def test_apostrophes_are_not_quoted_spans(self):
        self.assertEqual(protected_tokens("The man's hat and the child's toy")["quoted"], set())


class FramingTests(unittest.TestCase):
    def test_added_framing_word_fails(self):
        self.assertTrue(check_framing("A man rides a bike", "A photo of a man riding a bike"))
        self.assertTrue(check_framing("A man rides a bike", "Photos show a man riding a bike"))

    def test_framing_word_already_in_source_passes(self):
        self.assertEqual(check_framing("A video of a man riding", "A man riding, shown in a video"), [])


class DeterministicCheckTests(unittest.TestCase):
    def test_identity_after_normalization_is_hard(self):
        checks = deterministic_checks("A man rides a bike.", "a man, rides a BIKE", [], word_tokens)
        self.assertIn("identical to source", checks.hard)

    def test_identical_to_earlier_paraphrase_is_hard(self):
        checks = deterministic_checks(SOURCE, VALID[0], [VALID[0].upper()], word_tokens)
        self.assertIn("identical to an earlier paraphrase", checks.hard)

    def test_token_limit_boundaries(self):
        at_limit = deterministic_checks(SOURCE, VALID[0], [], lambda t: 72)
        self.assertFalse(any("tokens" in f for f in at_limit.hard))
        self.assertIn("over_target", at_limit.soft)
        over = deterministic_checks(SOURCE, VALID[0], [], lambda t: 73)
        self.assertTrue(any("tokens 73" in f for f in over.hard))
        normal = deterministic_checks(SOURCE, VALID[0], [], lambda t: 50)
        self.assertNotIn("over_target", normal.soft)

    def test_edit_distance_values(self):
        self.assertEqual(token_edit_distance("a b c d", "a b c d"), 0.0)
        self.assertEqual(token_edit_distance("a b c d", "a b c e"), 0.25)
        self.assertEqual(token_edit_distance("a b", "b a"), 1.0)
        self.assertEqual(token_edit_distance("a b c", "a b c d"), 0.25)

    def test_low_diversity_is_soft_only(self):
        checks = deterministic_checks(SOURCE, "A man in a red hat rides a bicycle", [], word_tokens)
        self.assertIn("low_diversity", checks.soft)
        self.assertEqual(checks.hard, [])


def always_equivalent(pairs):
    return {(p["id"], p["slot"]): {"equivalent": True, "added": [], "removed": [], "changed": []}
            for p in pairs}


class RetryLoopTests(unittest.TestCase):
    def cases(self, texts):
        return {"c1": {"source": SOURCE, "slots": [{"strategy": s, "text": t} for s, t in
                                                   zip(("reorder", "synonym", "restructure", "combined"), texts)]}}

    def test_all_valid_first_try(self):
        out = run_validation_rounds(self.cases(VALID), regenerate_batch=lambda r: {},
                                    judge_batch=always_equivalent, count_tokens=word_tokens)
        self.assertEqual(out["c1"]["n_effective"], 4)
        self.assertTrue(all(s["attempts"] == 1 and s["status"] == "ok" for s in out["c1"]["slots"]))

    def test_fail_once_then_pass(self):
        texts = [SOURCE] + VALID[1:]  # slot 0 identical to source
        calls = []

        def regenerate(requests):
            calls.append(requests)
            return {(r["id"], r["slot"]): VALID[0] for r in requests}

        out = run_validation_rounds(self.cases(texts), regenerate_batch=regenerate,
                                    judge_batch=always_equivalent, count_tokens=word_tokens)
        slot = out["c1"]["slots"][0]
        self.assertEqual((slot["status"], slot["attempts"]), ("ok", 2))
        self.assertIn("identical to source", calls[0][0]["failures"])
        self.assertEqual(out["c1"]["n_effective"], 4)

    def test_fail_every_time_is_rejected_after_two_retries(self):
        texts = [SOURCE] + VALID[1:]
        out = run_validation_rounds(
            self.cases(texts), regenerate_batch=lambda rs: {(r["id"], r["slot"]): SOURCE for r in rs},
            judge_batch=always_equivalent, count_tokens=word_tokens)
        slot = out["c1"]["slots"][0]
        self.assertEqual((slot["status"], slot["attempts"]), ("rejected", 3))
        self.assertEqual(len(slot["history"]), 3)
        self.assertEqual(out["c1"]["n_effective"], 3)

    def test_judge_rejection_is_retried(self):
        bad = VALID[2]

        def judge(pairs):
            verdicts = always_equivalent(pairs)
            for p in pairs:
                if p["paraphrase"] == bad:
                    verdicts[(p["id"], p["slot"])] = {"equivalent": False, "added": ["on a sunny day"],
                                                      "removed": [], "changed": []}
            return verdicts

        replacement = "A red hat is worn by a man who rides a bike"
        out = run_validation_rounds(
            self.cases(VALID), regenerate_batch=lambda rs: {(r["id"], r["slot"]): replacement for r in rs},
            judge_batch=judge, count_tokens=word_tokens)
        slot = out["c1"]["slots"][2]
        self.assertEqual((slot["status"], slot["attempts"], slot["text"]), ("ok", 2, replacement))
        self.assertTrue(any("judge added" in f for f in slot["history"][0]["failures"]))

    def test_judge_missing_key_raises(self):
        with self.assertRaises(ValueError):
            run_validation_rounds(self.cases(VALID), regenerate_batch=lambda r: {},
                                  judge_batch=lambda pairs: {}, count_tokens=word_tokens)


if __name__ == "__main__":
    unittest.main()
