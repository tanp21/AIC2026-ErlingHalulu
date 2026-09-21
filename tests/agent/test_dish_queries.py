"""The dish name a web lookup found, searched as a transcript query.

The lookup prompts already end with `TỪ KHÓA: <tên món>; <2-4 nguyên liệu đặc trưng>`,
and nothing read it: the prose went into the rewrite prompt as notes, and across runs
the rewrite used the quantities the question mentions ("hai muỗng nước tương") instead
of the name. Measured on the round-3 soy-sauce query, the name is the strongest key
available -- the lookup said "Cá hồi áp chảo xốt tiêu", the transcript says that dish
out loud at 48 s, and an audio search for the name returns the video first, where the
query's own terms ranked it 30th of 84.
"""
import unittest

from agent.orchestrator import _Run
from agent.tasks import task_from_text
from agent.web import dish_name
from tests.agent.test_orchestrator import FakeDeepSeek, FakeGemini, FakeSearch, HITS, make_agent

LOOKUP = """Tập chương trình và công thức khớp nhất với mô tả là món **Cá hồi áp chảo xốt tiêu**
trong chương trình "Món Ngon Mỗi Ngày", do đầu bếp Hùng Khang hướng dẫn.
TỪ KHÓA: Cá hồi áp chảo xốt tiêu; cá hồi, tiêu xanh, hành tây, nấm linh chi
ĐÁP ÁN: hai muỗng cà phê"""

#: Verbatim from real runs, where the model ignored the requested `TỪ KHÓA:` format.
#: A parser that only reads the asked-for line searched nothing at all in those runs.
REAL_PROSE = """Tập phát sóng và công thức khớp nhất với mô tả là món **Cá hồi áp chảo xốt tiêu** trong \
chương trình *Món Ngon Mỗi Ngày*:

* **Tên món ăn:** Cá hồi áp chảo xốt tiêu (hoặc Cá hồi áp chảo sốt tiêu xanh).
* **Nguyên liệu chính:** cá hồi phi lê, tiêu xanh, hành tây, nấm linh chi."""

REAL_BOLD_ONLY = """Món ăn trong tập phát sóng này là **Chả mực bánh xe chiên giòn** (chương trình \
*Món ngon mỗi ngày*). Theo công thức và quá trình chế biến, các gia vị gồm có: đầu hành băm \
(1 muỗng cà phê)."""


class DishNameTests(unittest.TestCase):
    def test_it_reads_the_dish_and_leaves_the_ingredients(self):
        # The ingredients are what the query already described; only the name is new.
        self.assertEqual(dish_name(LOOKUP), "Cá hồi áp chảo xốt tiêu")

    def test_the_name_is_read_from_the_prose_a_lookup_actually_writes(self):
        self.assertEqual(dish_name(REAL_PROSE), "Cá hồi áp chảo xốt tiêu")
        self.assertEqual(dish_name(REAL_BOLD_ONLY), "Chả mực bánh xe chiên giòn")

    def test_alternatives_are_cut_off_the_first_name(self):
        self.assertIn("(hoặc", REAL_PROSE)  # the case the split exists for
        self.assertEqual(dish_name("TỪ KHÓA: Bánh xèo (hoặc bánh khọt); bột gạo"), "Bánh xèo")

    def test_a_lookup_without_the_line_yields_nothing(self):
        self.assertIsNone(dish_name("Món này là cá hồi áp chảo."))
        self.assertIsNone(dish_name(None))

    def test_unaccented_and_lowercased_labels_still_match(self):
        text = "Tu khoa: Bánh xèo miền Tây; bột gạo, tôm\nDAP AN: không rõ"
        self.assertEqual(dish_name(text), "Bánh xèo miền Tây")

    def test_a_too_short_name_is_refused(self):
        self.assertIsNone(dish_name("TỪ KHÓA: A; b\nĐÁP ÁN: x"))


class DishQueryTests(unittest.TestCase):
    def run_for(self, lookups):
        agent = make_agent(FakeDeepSeek(), FakeGemini({("L01_V002", 200): 9}),
                           FakeSearch(HITS), mode="fast")
        run = _Run(agent, task_from_text("q-qa", "hỏi", "qa"))
        run.result.lookups.extend(lookups)
        return run

    def test_a_lookup_names_a_transcript_query(self):
        queries = self.run_for([{"question": "món ăn", "answer": LOOKUP,
                                 "about_the_query": True}])._dish_queries(room=2)

        self.assertEqual([query["kind"] for query in queries], ["audio"])
        self.assertEqual(queries[0]["text"], "Cá hồi áp chảo xốt tiêu")
        self.assertEqual(queries[0]["method"], "keyword")

    def test_a_lookup_about_a_video_it_already_chose_is_not_searched(self):
        # That one names whatever dish the video it was shown has, so searching it
        # spends a query reinforcing a choice that may be the mistake being corrected.
        # Real example: the recipe lookup for L26_V435 answered "Chả mực bánh xe chiên
        # giòn" while the query's video was the salmon one.
        chosen_video = {"question": "công thức (L26_V435)", "answer": REAL_BOLD_ONLY,
                        "about_the_query": False}

        self.assertEqual(self.run_for([chosen_video])._dish_queries(room=2), [])

    def test_the_same_dish_from_two_lookups_is_asked_once(self):
        queries = self.run_for([{"answer": LOOKUP, "about_the_query": True},
                                {"answer": LOOKUP, "about_the_query": True}])._dish_queries(room=2)

        self.assertEqual(len(queries), 1)

    def test_lookups_that_named_nothing_ask_nothing(self):
        self.assertEqual(self.run_for([{"answer": "không rõ", "about_the_query": True}])
                         ._dish_queries(room=2), [])
        self.assertEqual(self.run_for([])._dish_queries(room=2), [])


if __name__ == "__main__":
    unittest.main()
