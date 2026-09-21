import unittest

from agent.plan import initial_queries, parse_plan
from agent.tasks import task_from_text
from tests.agent.test_orchestrator import PLAN

AMBIGUOUS = {
    **PLAN,
    "scenes": [
        {"description_vi": "hai người cầm điện thoại",
         "visual_en": ["two people holding a phone", "a phone held by two people"]},
        {"description_vi": "trả lời phỏng vấn",
         "visual_en": ["a young person speaks to an interviewer",
                       "an interviewer talks with a young person"]},
    ],
    "target_scene": 1,
    "readings_en": ["two people walking outdoors, one holding a phone up",
                    "two people sitting over a phone screen",
                    "a third reading over the cap"],
}


class ReadingQueryTests(unittest.TestCase):
    def test_readings_are_their_own_queries_right_after_the_target_scene(self):
        task = task_from_text("q-kis", "hai người cầm điện thoại")
        queries = initial_queries(parse_plan(AMBIGUOUS, task), task)

        self.assertEqual([query["id"] for query in queries[:3]], ["r1.s1", "r1.r0", "r1.r1"])
        # a reading is one caption on its own, never fused into the scene's vector
        self.assertNotIn("paraphrases", queries[1])
        self.assertEqual(queries[1]["text"], "two people walking outdoors, one holding a phone up")
        self.assertIn("paraphrases", queries[0])

    def test_at_most_two_readings_survive_the_plan(self):
        task = task_from_text("q-kis", "x")
        self.assertEqual(len(parse_plan(AMBIGUOUS, task).readings_en), 2)

    def test_trake_gets_no_readings(self):
        task = task_from_text("q-trake", "E1: a\nE2: b")
        plan = parse_plan({**AMBIGUOUS, "events_en": ["a", "b"]}, task)
        self.assertEqual(plan.readings_en, [])
        self.assertEqual([q for q in initial_queries(plan, task) if q["id"].startswith("r1.r")], [])

    def test_an_unambiguous_plan_adds_nothing(self):
        task = task_from_text("q-kis", "x")
        plan = parse_plan(PLAN, task)
        self.assertEqual(plan.readings_en, [])
        self.assertEqual([q for q in initial_queries(plan, task) if q["id"].startswith("r1.r")], [])


if __name__ == "__main__":
    unittest.main()
