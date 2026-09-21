"""A question that asks for an amount is answered by a sentence that states one.

Measured on the round-3 soy-sauce query (its video is L26_V448, and the answer line
reads "…hai muỗng cà phê nước tương Phú Sĩ…"): the plan already named the substance,
and searching that term alone ranked the video 30th of 84, because "nước tương Phú
Sĩ" is the show's sponsor and is said in every episode. Paired against six saved
plans, with only the query list varying, the two phrasings below moved the target's
fused rank 25->4, 2->1, 154->63, 187->26, 14->1 and 154->63, and brought it into the
verification set in two of the six cases where it had been outside.

End-to-end numbers cannot see this: three runs per arm are dominated by which plan the
planner wrote, and the same query measured 1/3 against 2/3 that way.
"""
import unittest

from agent.plan import BATCH_LIMIT, initial_queries, parse_plan, quantity_queries
from agent.tasks import task_from_text
from tests.agent.test_orchestrator import PLAN

QUESTION = "Trong quá trình nêm gia vị, lượng nước tương được sử dụng là bao nhiêu?"


def plan_for(question=QUESTION, answer_source="speech", transcript=("nước tương", "gia vị")):
    task = task_from_text("q-qa", "x", "qa")
    return parse_plan({**PLAN, "question_vi": question, "answer_source": answer_source,
                       "transcript_vi": list(transcript), "caption_en": []}, task)


class QuantityQueryTests(unittest.TestCase):
    def test_a_spoken_amount_becomes_two_audio_phrasings(self):
        queries = quantity_queries(plan_for())

        self.assertEqual([query["kind"] for query in queries], ["audio", "audio"])
        # The question's own quantity word with a bare measure, then the spelled-out
        # unit the answer sentence uses. Both were measured against the live index.
        self.assertEqual([query["text"] for query in queries],
                         ["lượng nước tương muỗng", "muỗng cà phê nước tương"])

    def test_a_count_answered_on_screen_is_left_alone(self):
        # "Vùng này gồm bao nhiêu tỉnh?" is answered by a slide, so the two audio
        # queries would displace the OCR ones that find it.
        queries = quantity_queries(plan_for("Vùng này gồm bao nhiêu tỉnh?",
                                            answer_source="on_screen_text"))

        self.assertEqual(queries, [])

    def test_a_question_that_asks_for_no_amount_is_left_alone(self):
        self.assertEqual(quantity_queries(plan_for("Người đó nói gì?")), [])

    def test_without_a_substance_phrase_there_is_nothing_to_pair(self):
        self.assertEqual(quantity_queries(plan_for(transcript=())), [])

    def test_they_take_the_semantic_twins_slot_and_one_more(self):
        # Measured with the plan held fixed, which is the only way to see this change:
        # across six saved plans the target's fused rank went 25->4, 2->1, 154->63,
        # 187->26, 14->1, 154->63, and it entered the verification set in two cases
        # where it had not. The pair costs one slot net -- the semantic twin gives up
        # its place and one more is taken from the batch's tail -- which is why the
        # evidence has to be worth it, and for this question it is.
        task = task_from_text("q-qa", "x", "qa")
        with_amount = initial_queries(plan_for(), task)
        without = initial_queries(plan_for(answer_source="on_screen_text"), task)

        ids = [query["id"] for query in with_amount]
        self.assertEqual(["r1.q0", "r1.q1"], [q for q in ids if q.startswith("r1.q")])
        self.assertNotIn("r1.a0s", ids)
        self.assertIn("r1.a0s", [query["id"] for query in without])
        self.assertEqual(len(with_amount), len(without) + 1)
        self.assertLessEqual(len(with_amount), BATCH_LIMIT)
        # The batch is truncated in order, so the target scene still leads.
        self.assertEqual(with_amount[0]["kind"], "visual")


if __name__ == "__main__":
    unittest.main()
