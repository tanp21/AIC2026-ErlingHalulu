import re
import unittest

from agent import evidence as ev
from agent.evidence import (
    AFTER_SECONDS, BEFORE_SECONDS, EvidenceStore, QueryDetails, clusters, excerpts_of, query_details,
    select_excerpts, span_around, suggest_answers, summarize, summarize_all, words,
)
from agent.llm import LLMError
from agent.prompts import DETAILS_SYSTEM, MATCH_CAPTION_SYSTEM, MATCH_SYSTEM, SUGGEST_SYSTEM, VISION_MATCH_SYSTEM
from agent.review import HumanReview
from agent.tasks import task_from_text
from tests.agent.test_orchestrator import HITS, PLAN, FakeDeepSeek, FakeSearch
from tests.agent.test_review import ForbiddenVision, make_agent

CUES = [
    {"start": 0.0, "end": 5.0, "text": "xin chào các bạn"},
    {"start": 30.0, "end": 36.0, "text": "hôm nay nấu món nghêu hấp sả"},
    {"start": 90.0, "end": 96.0, "text": "cho thêm hai muỗng nước tương"},
]
SHOTS = [
    {"shot_idx": 1, "frame_idx": 100, "start": 0.0, "end": 20.0, "time": 10.0, "text": "MÓN NGON MỖI NGÀY"},
    {"shot_idx": 2, "frame_idx": 500, "start": 20.0, "end": 60.0, "time": 40.0, "text": "NGHÊU HẤP SẢ"},
    {"shot_idx": 3, "frame_idx": 1500, "start": 60.0, "end": 100.0, "time": 80.0, "text": "  "},
]
# The real gold video of a cooking query, in miniature. What the query describes is spread over
# the shots *before* the frame it asks for, and inside a caption it sits deep in the paragraph
# (character ~2,200 of ~3,900 for the apron and the vase): a caption cut to its head loses it.
PADDING = "The kitchen has a wooden cabinet, a steel sink and a shelf of jars. " * 30
WINDOWS = [
    {"shot_idx": 5, "window_idx": 0, "frame_idx": None, "start": 6.5, "end": 12.0, "time": 9.2,
     "text": "A salmon fillet rests on a board. It is sprinkled with pepper."},
    {"shot_idx": 6, "window_idx": 0, "frame_idx": None, "start": 12.0, "end": 19.3, "time": 15.6,
     "text": PADDING + "A woman wearing a white apron stands beside a vase of red ginger flowers. She smiles."},
    {"shot_idx": 7, "window_idx": 0, "frame_idx": None, "start": 19.2, "end": 22.0, "time": 20.6,
     "text": "Hands place four clams on a white plate. The plate is on a steel counter."},
    {"shot_idx": 8, "window_idx": 0, "frame_idx": None, "start": 22.0, "end": 45.0, "time": 33.5,
     "text": "A pan sizzles on a stove."},
    {"shot_idx": 9, "window_idx": 0, "frame_idx": None, "start": 300.0, "end": 310.0, "time": 305.0,
     "text": "A woman in a white apron in some far away shot."},
]
DETAILS = QueryDetails(
    [{"vi": "cô gái đeo tạp dề trắng", "terms_en": ["apron", "white apron"], "terms_vi": ["tạp dề"],
      "meaning_en": "A woman wearing a white apron.", "same_en": ["woman in a white apron"], "different_en": ["red apron", "black apron"]},
     {"vi": "đặt bốn con nghêu lên dĩa", "terms_en": ["clam", "shellfish", "oyster"], "terms_vi": ["nghêu"]},
     {"vi": "một chi tiết không ai nhắc", "terms_en": ["unicorn"], "terms_vi": ["kỳ lân"]}],
    {"terms_en": ["clam"], "terms_vi": ["nghêu"]},
)


# A detail with no quantity in it, which speech can support outright.
DISH = QueryDetails([{"vi": "món nghêu", "terms_en": ["clam"], "terms_vi": ["nghêu"]}])


class Search(FakeSearch):
    """A server whose caption *index* is not built: `captions` (index) fails, `caption_windows` (disk) works."""

    def __init__(self, hits=HITS):
        super().__init__(hits)
        self.calls = []

    def transcript(self, video_id):
        self.calls.append(("transcript", video_id))
        return CUES

    def ocr(self, video_id):
        self.calls.append(("ocr", video_id))
        return SHOTS

    def captions(self, video_id):
        raise AssertionError("the caption index is not what evidence reads")

    def caption_windows(self, video_id):
        self.calls.append(("caption_windows", video_id))
        return WINDOWS


def item(cid, time, video="L01_V001"):
    return {"id": cid, "video_id": video, "time": time}


class Recorder(FakeDeepSeek):
    """DeepSeek stand-in: answers the evidence prompts by their system message.

    The judge marks every excerpt shown under a detail as exact, unless `judge` says
    otherwise, which is enough to see exactly what reached it.
    """

    def __init__(self, plan=PLAN, fail=(), caption_judge=None, text_judge=None):
        super().__init__(plan)
        self.fail, self.calls, self.caption_judge, self.text_judge = set(fail), [], caption_judge, text_judge

    def chat_json(self, messages, **kwargs):
        system = messages[0]["content"]
        # "match_caption" judges the English captions, "match" the Vietnamese speech and OCR.
        for name, prompt in (("details", DETAILS_SYSTEM), ("match_caption", MATCH_CAPTION_SYSTEM),
                             ("match", MATCH_SYSTEM), ("suggest", SUGGEST_SYSTEM)):
            if system == prompt:
                self.calls.append((name, messages[-1]["content"]))
                if name in self.fail:
                    raise LLMError("deepseek down")
                return getattr(self, "_match_text" if name == "match" else f"_{name}")(messages[-1]["content"])
        return super().chat_json(messages, **kwargs)

    def prompt(self, name):
        return next(prompt for called, prompt in self.calls if called == name)

    def _details(self, _):
        return {"details": [{k: v for k, v in d.items()} for d in DETAILS.details],
                "question": DETAILS.question}

    def _match_text(self, prompt):
        """Speech and OCR are judged by nobody unless a test brings a `text_judge`: captions are what most tests are about."""
        return self.text_judge(prompt) if self.text_judge else {"details": []}

    def _match_caption(self, prompt):
        if self.caption_judge:
            return self.caption_judge(prompt)
        blocks = re.split(r"^(?=\d+\. )", prompt, flags=re.MULTILINE)[1:]
        return {"details": [{"detail": int(block.split(".", 1)[0]), "judgements": [
            {"id": excerpt, "why": "matches", "note_vi": "khớp", "verdict": "exact"} for excerpt in re.findall(r"^\s+(e\d+) \[", block, re.MULTILINE)]}
            for block in blocks]}

    def _suggest(self, prompt):
        first = re.search(r"^(e\d+) \[", prompt, re.MULTILINE)
        return {"answers": [{"answer": "  nghêu  ", "excerpt": first.group(1), "confidence": 8}] if first else []}


class CountTests(unittest.TestCase):
    def test_quantities_of_two_or_more_are_found_in_digits_and_in_both_languages(self):
        self.assertEqual(ev.counts_in("đặt lên 1 dĩa trắng 4 con X"), {4})          # one is "a"
        self.assertEqual(ev.counts_in("cầm lên hai con, ba quả"), {2, 3})
        self.assertEqual(ev.counts_in("a pair of hands and four clams, 12 in all"), {2, 4, 12})
        self.assertEqual(ev.counts_in("sau đó nam giới bay lên"), set())            # ordinary words, not numbers

    def test_only_a_detail_that_names_a_quantity_can_be_caught_out(self):
        self.assertIsNone(ev.count_mismatch("cô gái đeo tạp dề", "A woman with two hands"))
        self.assertIsNone(ev.count_mismatch("cầm lên 2 con X", "She is holding two shell-like items"))
        self.assertEqual(ev.count_mismatch("đặt 4 con X", "a plate holds two shrimp"), "chi tiết cần 4, đoạn nói 2")
        self.assertEqual(ev.count_mismatch("đặt 4 con X", "a plate holds shrimp"), "chi tiết cần 4, đoạn không nêu số lượng")
        self.assertIsNone(ev.count_mismatch("đặt 4 con X", "two hosts put four clams on the plate"))

    def test_a_graphic_marker_drawn_onto_the_footage_is_never_caught_out_on_its_count(self):
        """Reported on p2-6: "2 vòng tròn đỏ xuất hiện để khoanh vùng vị trí 2 thanh niên" was read like
        "4 con nghêu trên đĩa" -- the same circle drawn once around each rider, seconds apart, only ever
        shows one at a time in whatever frames were sampled, so the guard kept calling the true
        candidate a conflict. How many times a broadcast-drawn marker (a circle, arrow, box, highlight)
        is used is a count across the whole story, not something one still can total.
        """
        detail = "2 vòng tròn đỏ xuất hiện để khoanh vùng vị trí 2 thanh niên"
        self.assertIsNone(ev.count_mismatch(detail, "chỉ thấy một vòng tròn đỏ"))
        self.assertIsNone(ev.count_mismatch(detail, "không thấy vòng tròn nào"))
        self.assertTrue(ev._is_marker_detail(detail))
        self.assertTrue(ev._is_marker_detail("a red circle highlights the man"))  # English works too
        self.assertFalse(ev._is_marker_detail("đặt 4 con nghêu lên đĩa trắng"))   # a real object count
        self.assertIn("graphic mark", VISION_MATCH_SYSTEM.lower())


class WordTests(unittest.TestCase):
    def test_words_are_folded_and_a_short_ending_still_matches(self):
        self.assertEqual(words("Đèo Tà Pứa, 15.000đ"), ["deo", "ta", "pua", "15", "000d"])
        self.assertTrue(ev._word_matches("clam", "clams"))
        self.assertTrue(ev._word_matches("hold", "holding"))
        self.assertFalse(ev._word_matches("clam", "clamshell"))  # more than a short ending
        self.assertFalse(ev._word_matches("two", "twos"))       # too short to take an ending

    def test_a_multi_word_term_needs_every_word_in_the_sentence(self):
        terms = ev._term_words(["white apron"])
        self.assertEqual(ev._matched(terms, set(words("A woman in a white apron"))), 1)
        self.assertEqual(ev._matched(terms, set(words("A white plate and a red apron"))), 1)  # any order
        self.assertEqual(ev._matched(terms, set(words("A white plate"))), 0)


class SpanTests(unittest.TestCase):
    def test_a_span_holds_every_shot_overlapping_it_not_just_the_one_holding_a_moment(self):
        span = span_around(EvidenceStore(Search()), "L01_V001", 21.0 - BEFORE_SECONDS, 21.0 + AFTER_SECONDS)

        self.assertEqual([w["start"] for w in span["captions"]], [6.5, 12.0, 19.2, 22.0])  # not 300 s
        self.assertEqual([c["text"] for c in span["speech"]], ["xin chào các bạn", "hôm nay nấu món nghêu hấp sả"])
        self.assertEqual([o["text"] for o in span["ocr"]], ["MÓN NGON MỖI NGÀY", "NGHÊU HẤP SẢ"])

    def test_a_video_costs_one_fetch_per_kind_and_a_missing_one_is_empty(self):
        class Broken(Search):
            def ocr(self, video_id):
                raise RuntimeError("no ocr index")

        search = Broken()
        store = EvidenceStore(search)
        span_around(store, "L01_V001", 0, 50)
        span_around(store, "L01_V001", 60, 100)

        self.assertEqual(sorted(search.calls), [("caption_windows", "L01_V001"), ("transcript", "L01_V001")])
        self.assertEqual(span_around(store, "L01_V001", 0, 50)["ocr"], [])
        self.assertEqual(span_around(EvidenceStore(FakeSearch(HITS)), "L01_V001", 0, 50)["captions"], [])  # no such method

    def test_candidates_of_one_video_share_a_call_until_far_apart_best_ranked_first(self):
        found = clusters([item("C1", 500, "L02_V001"), item("C2", 10), item("C3", 30), item("C4", 400),
                          item("C5", 20, "L02_V001"), item("C6", 460)])

        self.assertEqual([[i["id"] for i in group] for group in found],
                         [["C1"], ["C2", "C3"], ["C4", "C6"], ["C5"]])
        self.assertEqual([len(g) for g in clusters([item(f"C{n}", n) for n in range(20)])], [8, 8, 4])


class ExcerptTests(unittest.TestCase):
    def excerpts(self):
        span = span_around(EvidenceStore(Search()), "L01_V001", -25, 46)
        return excerpts_of(span, [ev._term_sets(entry) for entry in DETAILS.details])

    def test_only_sentences_holding_a_term_are_kept_however_deep_in_the_paragraph(self):
        found = self.excerpts()
        texts = {e["text"] for e in found}

        self.assertIn("A woman wearing a white apron stands beside a vase of red ginger flowers.", texts)  # char ~2,000
        self.assertIn("Hands place four clams on a white plate.", texts)
        self.assertNotIn("A pan sizzles on a stove.", texts)
        self.assertNotIn("She smiles.", texts)
        # a Vietnamese term reaches speech and OCR; an English one only reaches a caption
        self.assertEqual({e["source"] for e in found if 1 in e["scores"]}, {"caption", "speech", "ocr"})
        self.assertEqual({e["source"] for e in found if 0 in e["scores"]}, {"caption"})

    def test_each_candidate_gets_its_own_best_sentences_within_reach(self):
        found = self.excerpts()
        near = select_excerpts(found, [item("C1", 21.0)], 3, 1)
        self.assertEqual([e["id"] for e in near], [f"e{n}" for n in range(1, len(near) + 1)])
        self.assertTrue(all(21 - BEFORE_SECONDS <= e["at"] <= 21 + AFTER_SECONDS for e in near))
        self.assertEqual(len(near), 2)  # one per detail that has anything; the unicorn detail has none
        both = select_excerpts(found, [item("C1", 21.0), item("C2", 21.5)], 3, 1)
        self.assertEqual(len(both), len(near))  # a sentence two candidates share is kept once


class ActionAndPurposeTests(unittest.TestCase):
    """The reported bug: "carries the pumpkin away, to hide it" was marked partial for a caption
    that showed the carrying but not the hiding, which happens off-screen or in a later shot."""

    def test_the_prompts_treat_an_unseen_purpose_as_no_evidence_against_the_action(self):
        for prompt in (MATCH_CAPTION_SYSTEM, MATCH_SYSTEM):
            self.assertIn("purpose" if prompt is MATCH_CAPTION_SYSTEM else "mục đích", prompt)
        self.assertIn("running across the stage carrying the pumpkin", MATCH_CAPTION_SYSTEM)
        self.assertIn("need not also show him hiding it", MATCH_CAPTION_SYSTEM)

    def test_a_caption_showing_only_the_carrying_is_exact_for_a_detail_that_also_names_the_purpose(self):
        details = QueryDetails([{"vi": "một người mang vật giống trái bí đỏ đi giấu",
                                 "terms_en": ["carry", "pumpkin", "hide", "away"], "terms_vi": [],
                                 "distinctiveness": 0.9}])
        store = EvidenceStore(Search())
        windows = [{"shot_idx": 1, "window_idx": 0, "frame_idx": None, "start": 10.0, "end": 20.0, "time": 15.0,
                    "text": "He is running across the stage while carrying the pumpkin, which looks orange."}]

        class Carry(Search):
            def caption_windows(self, video_id):
                return windows

        found = summarize(Recorder(), details, EvidenceStore(Carry()), [item("C1", 15.0)])
        check = found["C1"]["checks"][0]
        self.assertEqual(check["status"], "supported")
        self.assertIn("carry", check["quote"].lower())


class RareWordsTests(unittest.TestCase):
    """The reported bug (p2-1): the sentence that named the lion lost to generic ones from the next shot."""

    def test_a_sentence_with_a_rare_term_outranks_one_with_only_common_terms(self):
        captions = [
            {"start": 0.0, "end": 10.0, "text": "A yellow flower is next to a yellow sign. A yellow cup sits next to a yellow lamp. "
                                                "A yellow chair stands next to a yellow table. A yellow mat is next to a yellow wall."},
            {"start": 10.0, "end": 20.0, "text": "A long lion costume lies on the floor."},
        ]
        span = {"speech": [], "ocr": [], "captions": [{**c, "text": c["text"]} for c in captions]}
        terms = [{"text": [], "caption": ev._term_words(["yellow", "next to", "lion"])}]

        found = ev.excerpts_of(span, terms)
        by_text = {e["text"]: e["scores"][0] for e in found}
        self.assertGreater(by_text["A long lion costume lies on the floor."], max(v for t, v in by_text.items() if "lion" not in t))
        chosen = select_excerpts(found, [item("C1", 12.0)], 1, 1)
        self.assertEqual(chosen[0]["text"], "A long lion costume lies on the floor.")

    def test_the_search_plans_reading_of_the_query_reaches_the_details_step(self):
        """"A yellow animal" is a lion in a lion-dance corpus; the plan already knows, the checklist has to as well."""
        class Scene:
            visual_en = ["a yellow lion lies on a blue mat", "a lion costume"]

        class Plan:
            scenes = [Scene()]
            caption_en = ["five men beside a yellow lion costume"]
            readings_en = []

        seen = []

        class Spy(Recorder):
            def chat_json(self, messages, **kwargs):
                seen.append(messages[-1]["content"])
                return super().chat_json(messages, **kwargs)

        query_details(Spy(), task_from_text("q", "nhóm người bên cạnh một con vật màu vàng"), Plan())
        self.assertIn("Cách kế hoạch tìm kiếm hiểu truy vấn", seen[0])
        self.assertIn("- a yellow lion lies on a blue mat", seen[0])
        self.assertIn("- five men beside a yellow lion costume", seen[0])
        query_details(Spy(), task_from_text("q", "x"))                       # no plan: no section, no error
        self.assertNotIn("Cách kế hoạch", seen[1])

    def test_the_prompts_say_a_different_thing_that_shares_a_word_is_unrelated_not_a_conflict(self):
        for prompt in (MATCH_CAPTION_SYSTEM, MATCH_SYSTEM):
            self.assertIn("unrelated" if prompt is MATCH_CAPTION_SYSTEM else "KHÔNG phải \"conflict\"", prompt)
        self.assertIn("a yellow flower for \"a yellow animal\"", MATCH_CAPTION_SYSTEM)
        self.assertIn("Never call those \"conflict\"", MATCH_CAPTION_SYSTEM)
        self.assertIn("yellow lion costume lies", MATCH_CAPTION_SYSTEM)       # the worked example that is exact
        self.assertIn("TÁCH", DETAILS_SYSTEM)                                  # one thing per detail
        self.assertIn("không phải nội dung", DETAILS_SYSTEM)                   # its placeholders are not content


class QueryDetailsTests(unittest.TestCase):
    def test_a_detail_carries_what_it_means_what_is_the_same_thing_and_what_only_looks_like_it(self):
        """The reported bug: red ginger was called "not riềng tía", yellow daisies were called "close"."""
        class Botanist(Recorder):
            def _details(self, _):
                return {"details": [{"vi": "lọ hoa riềng tía", "meaning_en": "  A vase of red ginger (torch ginger) flowers. ",
                                     "same_en": ["red ginger", "torch ginger", "a", "b", "c", "d", "e"],
                                     "different_en": ["lavender", "", "yellow daisy"], "terms_en": ["ginger"], "terms_vi": ["riềng"]}]}

        found = query_details(Botanist(), task_from_text("q", "x")).details[0]
        self.assertEqual(found["meaning_en"], "A vase of red ginger (torch ginger) flowers.")
        self.assertEqual(found["same_en"], ["red ginger", "torch ginger", "a", "b", "c"])       # five at most
        self.assertEqual(found["different_en"], ["lavender", "yellow daisy"])
        self.assertIn("tên tiếng Anh chuẩn", DETAILS_SYSTEM)   # asks for the canonical English name
        self.assertNotIn("riềng", DETAILS_SYSTEM)                # without an example from a past query to copy


    def test_details_carry_their_terms_are_unique_capped_and_never_the_question(self):
        class Verbose(Recorder):
            def _details(self, _):
                return {"details": ["a chi tiết chỉ có chữ", {"vi": "a chi tiết chỉ có chữ"}, {"vi": "  b  ", "terms_en": [f"t{n}" for n in range(20)]},
                                    {"vi": ""}, *({"vi": f"d{i}"} for i in range(12))],
                        "question": {"terms_en": [], "terms_vi": []}}

        found = query_details(Verbose(), task_from_text("q", "x"))
        self.assertEqual([d["vi"] for d in found.details][:2], ["a chi tiết chỉ có chữ", "b"])
        self.assertEqual(len(found.details), 8)
        self.assertEqual(len(found.details[1]["terms_en"]), 8)
        self.assertIsNone(found.question)               # a question with no words to look for is none
        self.assertIn("KHÔNG đưa câu hỏi", DETAILS_SYSTEM)

    def test_distinctiveness_is_clamped_and_defaults_to_distinctive_when_the_model_omits_it(self):
        class Scored(Recorder):
            def _details(self, _):
                return {"details": [{"vi": "a", "distinctiveness": 0.3}, {"vi": "b", "distinctiveness": 5},
                                    {"vi": "c", "distinctiveness": -1}, {"vi": "d", "distinctiveness": "oops"},
                                    {"vi": "e"}]}

        found = query_details(Scored(), task_from_text("q", "x")).details
        self.assertEqual([d["distinctiveness"] for d in found], [0.3, 1.0, 0.0, 1.0, 1.0])
        self.assertIn("distinctiveness", DETAILS_SYSTEM)
        self.assertIn("ĐẶC TRƯNG", DETAILS_SYSTEM)

    def test_a_failure_is_an_empty_falsy_result(self):
        self.assertFalse(query_details(Recorder(fail={"details"}), task_from_text("q", "x")))

        class Garbage(Recorder):
            def _details(self, _):
                return ["not", "an", "object"]

        self.assertFalse(query_details(Garbage(), task_from_text("q", "x")))


def label_id(label: str) -> str:
    """The candidate id a vision image label carries: `"[C1] shot later in the same story"` -> `C1`.

    One candidate sends several images now (`vision_summarize` labels each of them by candidate),
    so a reply names the candidate once, not once per image.
    """
    return label.split("]", 1)[0].strip("[ ")


class VisionRecorder(Recorder):
    """A `Recorder` whose `.judge()` is the actual vision-call signature (images, prompt, schema, ...),
    not the text-judge stand-in `Recorder.judge` stores as a plain attribute -- the two must not collide.
    """

    def __init__(self, *args, vision_reply=None, vision_fail=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_reply, self.vision_fail, self.vision_calls = vision_reply, vision_fail, []

    def judge(self, images, prompt, schema, **kwargs):
        self.vision_calls.append((images, prompt))
        if self.vision_fail:
            raise LLMError("deepseek vision down")
        if self.vision_reply is not None:
            return self.vision_reply(images, prompt)
        ids = list(dict.fromkeys(label_id(label) for label, _ in images))
        return {"candidates": [{"id": cid, "details": [
            {"detail": 0, "why": "seen", "verdict": "exact", "note_vi": "thấy"}]} for cid in ids]}


class VisionSearch(Search):
    def image(self, video_id, frame_idx, width=640):
        return f"jpeg:{video_id}:{frame_idx}".encode()


def vitem(cid, time, video="L01_V001", frame_idx=None):
    """A candidate `item()` with the `frame_idx` a vision call fetches its frame by."""
    return {**item(cid, time, video), "frame_idx": int(time * 10) if frame_idx is None else frame_idx}


class VisionJudgeTests(unittest.TestCase):
    DETAIL = QueryDetails([{"vi": "cô gái đeo tạp dề trắng", "terms_en": ["apron"], "terms_vi": ["tạp dề"],
                            "meaning_en": "A woman wearing a white apron.", "same_en": ["white apron"],
                            "different_en": ["red apron"]}])

    def test_the_exact_frame_and_shots_around_it_are_fetched_labelled_and_the_prompt_carries_the_detail(self):
        deepseek = VisionRecorder()
        found = ev.vision_summarize(deepseek, VisionSearch(), self.DETAIL, [vitem("C1", 21.0, "L01_V001", frame_idx=210)])

        self.assertEqual(len(deepseek.vision_calls), 1)
        images, prompt = deepseek.vision_calls[0]
        # its own frame, then VisionSearch's (FakeSearch's) neighbours -20/-10/+10/+20 in time order,
        # each labelled with how far it is from the candidate's moment, the candidate's own frame excluded
        self.assertEqual(images, [
            ("[C1] this candidate's own moment", b"jpeg:L01_V001:210"),
            ("[C1] shot 2s earlier in the same story", b"jpeg:L01_V001:190"),
            ("[C1] shot 1s earlier in the same story", b"jpeg:L01_V001:200"),
            ("[C1] shot 1s later in the same story", b"jpeg:L01_V001:220"),
            ("[C1] shot 2s later in the same story", b"jpeg:L01_V001:230"),
        ])
        self.assertIn("A woman wearing a white apron.", prompt)
        self.assertIn("Same: white apron", prompt)
        self.assertIn("Different: red apron", prompt)
        self.assertEqual(found, {"C1": {0: ("supported", "thấy")}})

    def test_context_frames_are_spread_over_the_window_when_there_are_no_shots(self):
        """The fallback (no shot boundaries): spread over the raw keyframe neighbourhood, bounded by
        `seconds` -- labelling is covered by the test above, this is the reach and the bound.
        """
        class Wide(VisionSearch):
            def neighbours(self, video_id, frame_idx, radius=6):
                # one keyframe per second for five minutes either side, plus one far outlier
                return [{"video_id": video_id, "frame_idx": frame_idx + step * 10, "time": (frame_idx + step * 10) / 10}
                        for step in list(range(-300, 301)) + [3000]]

        narrow = ev._context_frames(Wide(), "L01_V001", 1000, 100.0, seconds=45.0)
        self.assertEqual(len(narrow), 2 * ev.VISION_CONTEXT_FRAMES)
        self.assertNotIn(1000, [f["frame_idx"] for f in narrow])                 # never the candidate's own frame
        self.assertNotIn(31000, [f["frame_idx"] for f in narrow])                # nor the far outlier
        self.assertEqual([f["frame_idx"] for f in narrow], sorted(f["frame_idx"] for f in narrow))
        self.assertLessEqual(max(f["time"] for f in narrow) - 100.0, 45.0)
        self.assertLessEqual(100.0 - min(f["time"] for f in narrow), 45.0)

        wide = ev._context_frames(Wide(), "L01_V001", 1000, 100.0)               # the production default (180s)
        self.assertGreater(max(f["time"] for f in wide), max(f["time"] for f in narrow))
        self.assertLessEqual(max(f["time"] for f in wide) - 100.0, ev.VISION_CONTEXT_SECONDS)

    def test_a_search_that_cannot_find_neighbours_still_sends_the_candidates_own_frame(self):
        class NoNeighbours(VisionSearch):
            def neighbours(self, video_id, frame_idx, radius=6):
                raise RuntimeError("no frame map")

        deepseek = VisionRecorder()
        found = ev.vision_summarize(deepseek, NoNeighbours(), self.DETAIL, [vitem("C1", 21.0, "L01_V001", frame_idx=210)])
        self.assertEqual(deepseek.vision_calls[0][0], [("[C1] this candidate's own moment", b"jpeg:L01_V001:210")])
        self.assertEqual(found, {"C1": {0: ("supported", "thấy")}})

    def test_none_no_entries_or_no_group_asks_nothing(self):
        self.assertEqual(ev.vision_summarize(None, VisionSearch(), self.DETAIL, [vitem("C1", 21.0)]), {})
        self.assertEqual(ev.vision_summarize(VisionRecorder(), VisionSearch(), QueryDetails(), [vitem("C1", 21.0)]), {})
        self.assertEqual(ev.vision_summarize(VisionRecorder(), VisionSearch(), self.DETAIL, []), {})

    def test_a_reply_that_names_the_wrong_candidate_still_lands_on_the_one_whose_images_they_are(self):
        """Measured live, p2-6: a call whose images were labelled `[GOLD]` came back `{"id": "C1"}`, and
        the reply was thrown away whole -- the candidate kept a text-only checklist and nothing said so.
        `VISION_CHUNK` is 1, so which candidate a reply is about is never a question the model has to
        answer: the label on the images already said it.
        """
        def wrong_id(images, prompt):
            return {"candidates": [{"id": "C9", "details": [
                {"detail": 0, "why": "one red circle around the rider", "verdict": "exact",
                 "note_vi": "có một vòng tròn đỏ"}]}]}

        deepseek = VisionRecorder(vision_reply=wrong_id)
        found = ev.vision_summarize(deepseek, VisionSearch(), self.DETAIL,
                                    [vitem("C1", 21.0, "L01_V001", frame_idx=210)])
        self.assertEqual(found, {"C1": {0: ("supported", "có một vòng tròn đỏ")}})

    def test_a_chunk_of_several_candidates_is_still_told_apart_by_the_ids_it_answers_with(self):
        """A call that carries more than one candidate needs the reply's ids -- and an id nobody asked
        about is dropped, never guessed onto one of the candidates it was asked about.
        """
        original = ev.VISION_CHUNK
        ev.VISION_CHUNK = 2
        try:
            deepseek = VisionRecorder(vision_reply=lambda images, prompt: {"candidates": [
                {"id": "C2", "details": [{"detail": 0, "why": "seen", "verdict": "exact", "note_vi": "thấy"}]},
                {"id": "C9", "details": [{"detail": 0, "why": "seen", "verdict": "exact", "note_vi": "thấy"}]},
            ]})
            found = ev.vision_summarize(deepseek, VisionSearch(), self.DETAIL,
                                        [vitem("C1", 21.0, frame_idx=210), vitem("C2", 22.0, frame_idx=220)])
        finally:
            ev.VISION_CHUNK = original
        self.assertEqual(found, {"C2": {0: ("supported", "thấy")}})

    def test_a_failed_chunk_or_backend_without_vision_is_no_result_never_an_error(self):
        failing = ev.vision_summarize(VisionRecorder(vision_fail=True), VisionSearch(), self.DETAIL, [vitem("C1", 21.0)])
        self.assertEqual(failing, {})

        class NoVision(Recorder):
            pass  # no .judge() method at all: like a fake that never implemented it

        self.assertEqual(ev.vision_summarize(NoVision(), VisionSearch(), self.DETAIL, [vitem("C1", 21.0)]), {})

        class BrokenImages(VisionSearch):
            def image(self, video_id, frame_idx, width=640):
                raise RuntimeError("ffmpeg failed")

        self.assertEqual(ev.vision_summarize(VisionRecorder(), BrokenImages(), self.DETAIL, [vitem("C1", 21.0)]), {})

    def test_only_known_ids_and_detail_indices_and_verdicts_are_kept(self):
        def reply(images, prompt):
            return {"candidates": [
                {"id": "C1", "details": [
                    {"detail": 0, "why": "w", "verdict": "conflict", "note_vi": "sai"},
                    {"detail": 9, "why": "w", "verdict": "exact"},                # no such detail
                    {"detail": 0, "why": "w", "verdict": "definitely"},          # not a verdict
                ]},
                {"id": "C99", "details": [{"detail": 0, "why": "w", "verdict": "exact"}]},   # never asked about
            ]}

        found = ev.vision_summarize(VisionRecorder(vision_reply=reply), VisionSearch(), self.DETAIL, [vitem("C1", 21.0)])
        self.assertEqual(found, {"C1": {0: ("contradicted", "sai")}})

    def test_a_count_the_judge_itself_reports_seeing_caps_its_verdict_at_partial(self):
        """The excerpt judges have had this guard since the "4 con X" report; vision needs it too.

        A live p2-30 run marked "places 4 X on a white plate" supported from a frame showing two clams,
        because a verdict read off pixels bypassed `count_mismatch` -- which only ever saw the words.
        """
        detail = QueryDetails([{"vi": "đặt lên dĩa trắng 4 con nghêu", "terms_en": ["clam"],
                                "meaning_en": "Places four clams on a white plate."}])

        def reply(why):
            return lambda images, prompt: {"candidates": [{"id": "C1", "details": [
                {"detail": 0, "why": why, "verdict": "exact", "note_vi": "thấy"}]}]}

        two = ev.vision_summarize(VisionRecorder(vision_reply=reply("A hand puts two clams on the white plate.")),
                                  VisionSearch(), detail, [vitem("C1", 21.0)])
        self.assertEqual(two["C1"][0][0], "partial")
        self.assertIn("chi tiết cần 4", two["C1"][0][1])
        four = ev.vision_summarize(VisionRecorder(vision_reply=reply("Four clams lie on the white plate.")),
                                   VisionSearch(), detail, [vitem("C1", 21.0)])
        self.assertEqual(four["C1"][0][0], "supported")
        quiet = ev.vision_summarize(VisionRecorder(vision_reply=reply("Clams lie on the white plate.")),
                                    VisionSearch(), detail, [vitem("C1", 21.0)])
        self.assertEqual(quiet["C1"][0][0], "supported")            # no count stated: nothing to check against
        self.assertIn("SAY THE NUMBER", VISION_MATCH_SYSTEM)

    def test_one_candidate_a_call_each_with_its_own_story(self):
        # a candidate now carries the keyframes of the shots around it, so a call is one candidate's
        # story rather than two candidates mixed into one judgement
        items = [vitem(f"C{n}", float(n), "L01_V001", frame_idx=n) for n in range(9)]
        deepseek = VisionRecorder()
        found = ev.vision_summarize(deepseek, VisionSearch(), self.DETAIL, items)
        self.assertEqual(len(deepseek.vision_calls), 9)
        self.assertEqual({label.split("]", 1)[0] for call, _ in deepseek.vision_calls for label, _ in call},
                         {f"[C{n}" for n in range(9)})
        self.assertEqual(len(found), 9)


class VisionWeightsTests(unittest.TestCase):
    """Asked for on p2-1, once vision does the checking: a criterion must still count by how much it
    tells candidates apart (the same `detail_weights` a caption checklist feeds), and a detail only the
    shots around the candidate show must count as that candidate's own moment.
    """

    DETAIL = QueryDetails([
        {"vi": "một con vật màu vàng", "terms_en": ["animal"], "terms_vi": [],
         "meaning_en": "A yellow animal.", "same_en": [], "different_en": [], "distinctiveness": 0.2},
        {"vi": "mang vật giống trái bí đỏ đi giấu", "terms_en": ["pumpkin"], "terms_vi": [],
         "meaning_en": "Carries an object that looks like a pumpkin away to hide it.",
         "same_en": [], "different_en": [], "distinctiveness": 1.0},
    ])

    class Wordless(Search):
        """No transcript, no on-screen text, no captions: vision is the only thing that can check."""

        def transcript(self, video_id):
            return []

        def ocr(self, video_id):
            return []

        def caption_windows(self, video_id):
            return []

    def judge(self, reply, items):
        vision = VisionRecorder(vision_reply=reply)
        return summarize(vision, self.DETAIL, EvidenceStore(self.Wordless()), items, vision=vision, search=VisionSearch())

    def test_a_criterion_vision_confirms_is_weighted_by_its_distinctiveness_like_a_captions(self):
        def reply(images, prompt):
            return {"candidates": [
                {"id": "C1", "details": [{"detail": 0, "why": "w", "verdict": "exact", "note_vi": "thấy"}]},
                {"id": "C2", "details": [{"detail": 1, "why": "w", "verdict": "exact", "note_vi": "thấy"}]}]}

        found = self.judge(reply, [vitem("C1", 20.0), vitem("C2", 30.0)])
        weights, _ = ev.detail_weights(list(found.values()), 2, self.DETAIL.details)
        self.assertLess(weights[0], weights[1])                        # "a yellow animal" is the guessable one
        self.assertTrue(all(check["source"] == "vision" and check["detail"] in (0, 1)
                            for summary in found.values() for check in summary["checks"]
                            if check["status"] != "not_mentioned"))

        scored = {cid: ev.evidence_of(found[cid], weights, time=time) for cid, time in (("C1", 20.0), ("C2", 30.0))}
        self.assertGreater(scored["C2"]["score"], scored["C1"]["score"])   # the story detail leads, not the guessable one

    def test_a_detail_only_a_neighbouring_shot_shows_is_credited_at_the_candidates_own_moment(self):
        def reply(images, prompt):
            return {"candidates": [{"id": "C1", "details": [
                {"detail": 1, "why": "a later shot shows him hiding it", "verdict": "exact", "note_vi": "thấy"}]}]}

        found = self.judge(reply, [vitem("C1", 20.0)])
        check = found["C1"]["checks"][1]
        self.assertEqual((check["status"], check["source"], check["at"]), ("supported", "vision", 20.0))
        weights, _ = ev.detail_weights([found["C1"]], 2, self.DETAIL.details)
        self.assertAlmostEqual(ev.evidence_of(found["C1"], weights, time=20.0)["share"],
                               weights[1] / sum(weights), places=4)       # no proximity loss for the story around it


class ShotContextTests(unittest.TestCase):
    """The reported p2-1 story again: its beats live in whole shots, so the context is chosen per shot.

    In L24_V035 the moment the search returned (frame 628, 20.9 s) sits in shot 1; "the man wakes up"
    (a man on the floor beside the lying lion) is in shot 4 and "he shakes the lion" in shot 5, 22 s and
    36 s later. Every sampling by time -- over the raw keyframe neighbourhood, and over the seconds of
    the window -- stepped over one or both of them.
    """

    SHOTS = [
        (0, 0, 486, 0.0, 16.23, [(72, 2.4), (243, 8.1), (413, 13.77)]),
        (1, 487, 654, 16.23, 21.83, [(512, 17.07), (570, 19.0), (628, 20.93)]),
        (2, 655, 1073, 21.83, 35.8, [(717, 23.9), (864, 28.8), (1010, 33.67)]),
        (3, 1074, 1253, 35.8, 41.8, [(1100, 36.67), (1163, 38.77), (1226, 40.87)]),
        (4, 1254, 1639, 41.8, 54.67, [(1311, 43.7), (1446, 48.2), (1581, 52.7)]),
        (5, 1640, 1788, 54.67, 59.63, [(1662, 55.4), (1714, 57.13), (1765, 58.83)]),
        (6, 1789, 3149, 59.63, 105.0, [(1993, 66.43), (2469, 82.3), (2945, 98.17)]),
    ]

    class Shots(VisionSearch):
        """A server with shot boundaries; `rows` lets a test bend them."""

        def __init__(self, rows=None):
            super().__init__()
            self.rows, self.shot_calls = rows if rows is not None else ShotContextTests.SHOTS, 0

        def shots(self, video_id):
            self.shot_calls += 1
            return [{"index": index, "start": start, "end": end, "start_frame": first, "end_frame": last,
                     "keyframes": [{"video_id": video_id, "frame_idx": frame, "time": time}
                                   for frame, time in keys]}
                    for index, first, last, start, end, keys in self.rows]

    def test_the_nearest_shot_of_each_side_is_dense_the_rest_are_one_frame_reaching_further(self):
        """The nearest shot keeps every keyframe (a beat that is a change needs two states visible,
        measured on this candidate against the same prompt: one frame per shot had the judge answer
        "unrelated" to "he wakes the animal up" in every run where three had it answer `exact` every
        time); shots past that keep one each, so the same frame budget reaches further out in time --
        which is what the p2-2 report needed (see `DenseVideoTests`). The candidate's own shot's other
        two keyframes (512, 570 -- both before it; it is the shot's last) come first of all, ahead of
        even the nearest other shot -- what the p2-6 report needed (see `EmptyKeyframeShotTests`).
        """
        deepseek = VisionRecorder()
        search = self.Shots()
        ev.vision_summarize(deepseek, search, VisionJudgeTests.DETAIL, [vitem("C1", 20.933, "L24_V035", frame_idx=628)])

        frames = [data.decode().split(":")[-1] for _, data in deepseek.vision_calls[0][0]]
        self.assertEqual(frames[0], "628")                       # its own moment first
        self.assertEqual(frames[1:], ["72", "243", "413",       # the nearest shot before it: dense
                                      "512", "570",              # the candidate's own shot's other frames
                                      "717", "864", "1010",     # the nearest shot after it: dense too
                                      "1100",                   # every shot past that: one frame each,
                                      "1311", "1662", "1993"])  # reaching the far end of the 180s window
        self.assertEqual(search.shot_calls, 1)                   # one request per video, however many candidates
        ev.vision_summarize(VisionRecorder(), search, VisionJudgeTests.DETAIL,
                            [vitem("C2", 20.933, "L24_V035", frame_idx=628)])
        self.assertEqual(search.shot_calls, 1)                   # still one: the shot list is cached per video

    def test_a_shot_that_reaches_into_the_window_without_a_frame_inside_it_is_left_out(self):
        rows = [row for row in self.SHOTS[:-1]]
        rows.append((6, 1789, 3149, 59.63, 105.0, [(1993, 66.43)]))
        frames = [f["frame_idx"] for f in ev._context_frames(self.Shots(rows), "L24_V035", 628, 20.933, seconds=45.0)]
        self.assertNotIn(1993, frames)                            # 45.5s out: just past this narrower window
        self.assertIn(1662, frames)

    def test_a_server_without_shot_boundaries_still_uses_the_keyframe_neighbourhood(self):
        """`shots()` is absent on older deployments (and on the test fakes above): the old path stays."""
        kept = [f["frame_idx"] for f in ev._context_frames(VisionSearch(), "L01_V001", 210, 21.0)]
        self.assertEqual(kept, [190, 200, 220, 230])             # FakeSearch's ±10/±20 s, nearest spread
        self.assertIsNone(ev._shot_frames(VisionSearch(), "L01_V001", 210, 21.0, 45.0, 1, 16))

        class Broken(VisionSearch):
            def shots(self, video_id):
                raise RuntimeError("no shot map for this video")

        self.assertEqual([f["frame_idx"] for f in ev._context_frames(Broken(), "L01_V001", 210, 21.0)],
                         [190, 200, 220, 230])                    # a failure falls back, it does not raise


class EmptyKeyframeShotTests(unittest.TestCase):
    """The reported p2-6 bug: a shot with NO indexed keyframe was invisible to the checklist.

    On the live candidate the 37 s shot right before the one holding it (nothing in it stood out
    enough for the offline embedding pass to sample a keyframe from) held the query's "two red circles"
    beat, and dropping keyframe-less shots at `_shots_of` made that whole stretch structurally
    unreachable -- not thinned, not deprioritised, absent. A shot with no keyframe still gets its own
    midpoint frame, fetched directly rather than picked from an index.
    """

    #: own shot (has keyframes) at 0-10s; a 37 s keyframe-less shot right before it, mirroring the
    #: live gap (790.6-827.5s before the candidate's 827.5-837.4s shot); one further keyframed shot
    #: further back so the "several shots deep" walk still has something past the gap.
    SHOTS = [
        (0, 0, 250, -47.0, -10.0, [(50, -42.0), (150, -27.0), (240, -11.2)]),
        (1, 251, 1174, -10.0, 27.0, []),
        (2, 1175, 1425, 27.0, 37.0, [(1200, 28.0), (1300, 32.0), (1400, 36.0)]),
    ]

    class Shots(VisionSearch):
        def __init__(self, rows=None):
            super().__init__()
            self.rows = rows if rows is not None else EmptyKeyframeShotTests.SHOTS

        def shots(self, video_id):
            return [{"index": i, "start": s, "end": e, "start_frame": sf, "end_frame": ef,
                     "keyframes": [{"video_id": video_id, "frame_idx": f, "time": t} for f, t in keys]}
                    for i, sf, ef, s, e, keys in self.rows]

    def test_a_shot_with_no_keyframe_still_contributes_its_own_midpoint(self):
        kept = ev._context_frames(self.Shots(), "L22_V024", 1300, 32.0)
        # shot 1 has no keyframe of its own: (251+1174)//2 = 712, at its midpoint time (-10-10.0+27.0... )
        midpoint_frame = (251 + 1174) // 2
        self.assertIn(midpoint_frame, [f["frame_idx"] for f in kept])

    def test_the_candidates_own_moment_can_still_be_found_inside_a_keyframe_less_shot(self):
        # a candidate landing inside shot 1 (no keyframes of its own) must still be placed correctly,
        # not silently mistaken for the nearest keyframed shot
        found = ev._context_frames(self.Shots(), "L22_V024", 700, 5.0)   # frame_idx 700 sits in shot 1
        self.assertNotIn(700, [f["frame_idx"] for f in found])           # never the candidate's own frame
        self.assertIn(50, [f["frame_idx"] for f in found])                # reaches shot 0's own keyframes

    def test_a_keyframe_less_shot_outside_the_window_is_still_excluded(self):
        rows = [(0, 0, 250, 100.0, 137.0, [(150, 118.0)]),
                (1, 251, 30251, 137.0, 1337.0, [])]                       # a 20-minute empty shot
        kept = ev._context_frames(self.Shots(rows), "L22_V024", 150, 118.0, seconds=45.0)
        self.assertNotIn((251 + 30251) // 2, [f["frame_idx"] for f in kept])   # its midpoint is minutes away


class OwnShotKeyframesTests(unittest.TestCase):
    """The rest of the reported p2-6 bug: the candidate's OWN shot's other keyframes were dropped.

    `vision_summarize` sends only the candidate's own exact frame for "this candidate's own moment";
    the rest of that same shot came only through `_context_frames`, which used to skip the shot the
    candidate is IN entirely (`shot is own: continue`). On the live candidate the query's second
    red-circle beat sat in the shot's own second keyframe, 3.5 s after the candidate's own -- never a
    neighbouring shot at all, just the rest of the one it is already in -- and it was never asked for.
    """

    #: own shot (frame 30, candidate) has two more keyframes either side of it; a further shot on each
    #: side too, so "own shot first, then outward" is the property under test, not just "own shot only".
    SHOTS = [
        (0, 0, 250, -40.0, -10.0, [(80, -35.0)]),
        (1, 251, 750, -10.0, 20.0, [(255, -9.0), (300, 0.0), (700, 15.0)]),
        (2, 751, 1000, 20.0, 40.0, [(800, 25.0)]),
    ]

    class Shots(VisionSearch):
        def shots(self, video_id):
            return [{"index": i, "start": s, "end": e, "start_frame": sf, "end_frame": ef,
                     "keyframes": [{"video_id": video_id, "frame_idx": f, "time": t} for f, t in keys]}
                    for i, sf, ef, s, e, keys in OwnShotKeyframesTests.SHOTS]

    def test_the_rest_of_the_candidates_own_shot_reaches_the_model(self):
        deepseek = VisionRecorder()
        ev.vision_summarize(deepseek, self.Shots(), VisionJudgeTests.DETAIL,
                            [vitem("C1", 0.0, "L22_V024", frame_idx=300)])   # the shot's middle keyframe
        frames = [data.decode().split(":")[-1] for _, data in deepseek.vision_calls[0][0]]
        self.assertEqual(frames[0], "300")                          # its own exact frame first
        # both of its own shot's other keyframes reach the model, not just the nearest OTHER shot's
        self.assertEqual(sorted(frames[1:], key=int), ["80", "255", "700", "800"])

    def test_the_own_shots_keyframes_are_not_crowded_out_by_a_tight_budget(self):
        # a budget too small for both the own shot's extra keyframe and the next shot's: the own shot,
        # being literally the candidate's own, wins the slot -- it is not a "neighbour" competing on
        # the same footing.
        kept = [f["frame_idx"] for f in ev._context_frames(self.Shots(), "L22_V024", 300, 0.0, most=1)]
        self.assertIn(255, kept)                                     # own shot's nearer keyframe: kept
        self.assertNotIn(80, kept)                                   # shot 0: crowded out by the tight budget

    def test_a_candidate_at_the_edge_of_its_shot_has_no_own_side_to_add(self):
        # frame 700 is the LAST keyframe of shot 1: nothing after it in that shot, so the later side
        # goes straight to the next shot, and nothing raises over an empty own-side list.
        kept = [f["frame_idx"] for f in ev._context_frames(self.Shots(), "L22_V024", 700, 15.0)]
        self.assertEqual(kept, [80, 255, 300, 800])


class DenseVideoTests(unittest.TestCase):
    """The reported p2-2 bug: capping by shot COUNT lost reach on a densely-cut video.

    Query: "the clip starts with a photo of the rhino painting, ends with a photo of the monkey
    graffiti on a bridge." The candidate sat in a shot at 747.8s; the second beat was 19s later --
    comfortably inside any reasonable time budget -- but seven 1-2s shots away, past the old
    `VISION_CONTEXT_SHOTS = 4` cap, so the checklist never saw it and called the detail a conflict
    instead of leaving it not_mentioned. Reproduces that spacing (one shot a second) rather than a
    hand-picked-friendly layout; the target here sits ten shots out, comfortably past that old cap and
    comfortably inside the frame budget.
    """

    #: (index, start_frame, end_frame, start, end, keyframe seconds) -- one shot a second, matching
    #: the live video's own density around the reported candidate.
    SHOTS = [(i, i * 30, i * 30 + 29, float(i), float(i) + 1, [float(i) + 0.5]) for i in range(30)]

    class Dense(VisionSearch):
        def shots(self, video_id):
            return [{"index": i, "start": s, "end": e, "start_frame": sf, "end_frame": ef,
                     "keyframes": [{"video_id": video_id, "frame_idx": int(t * 30), "time": t} for t in keys]}
                    for i, sf, ef, s, e, keys in DenseVideoTests.SHOTS]

    def test_a_shot_count_cap_would_have_missed_it_a_frame_budget_reaches_it(self):
        # own moment: shot 0 (0-1s); the target beat sits in shot 10 (10-11s), ten shots away -- past
        # the old `VISION_CONTEXT_SHOTS = 4` cap, well inside a 180s time budget.
        kept = [f["frame_idx"] for f in ev._context_frames(self.Dense(), "X_V001", 15, 0.5)]
        self.assertIn(10 * 30 + 15, kept)                         # the far beat's shot is reached
        self.assertLessEqual(len(kept), ev.VISION_CONTEXT_MAX * 2)  # the frame budget still bounds the total

    def test_the_reported_candidate_ticks_the_far_detail_a_shot_count_cap_could_not_reach(self):
        def reply(images, prompt):
            labels = [label for label, _ in images]
            far = any("10" in label and "later" in label for label in labels[1:])
            return {"candidates": [{"id": "C1", "details": [
                {"detail": 0, "why": "seen", "verdict": "exact" if far else "unrelated", "note_vi": "thấy"}]}]}

        found = ev.vision_summarize(VisionRecorder(vision_reply=reply), self.Dense(),
                                    QueryDetails([{"vi": "cảnh cuối", "meaning_en": "the ending scene"}]),
                                    [vitem("C1", 0.5, "X_V001", frame_idx=15)])
        self.assertEqual(found["C1"][0][0], "supported")


class LooksLikeTests(unittest.TestCase):
    """Reported mid-query: the detail is "an object that looks like a pumpkin", not a pumpkin.

    Dropping the resemblance would make a prop or a decoration a conflict; keeping it is what lets the
    person's own words decide whether the thing on screen is the one the query means.
    """

    def test_both_checking_prompts_keep_a_resemblance_a_resemblance(self):
        self.assertIn('"giống"', DETAILS_SYSTEM)
        self.assertIn("looks like", DETAILS_SYSTEM.lower())
        self.assertIn("looks like", VISION_MATCH_SYSTEM.lower())


class AmbiguousXanhTests(unittest.TestCase):
    """Reported on p2-6: the gold candidate's car and jacket are green, and the checklist called both
    a conflict against a query that only ever says "xanh" -- never "xanh dương" or "xanh lá".

    Vietnamese "xanh" on its own is the classic grue word: it covers blue and green both, and a query
    that does not add "dương"/"lá"/"lục"/"biển"/"da trời" is not claiming one of them, so a candidate
    that turns out to be the other is not wrong.
    """

    def test_the_details_prompt_treats_bare_xanh_as_either_colour(self):
        self.assertIn('"xanh"', DETAILS_SYSTEM)
        self.assertIn("mơ hồ", DETAILS_SYSTEM)
        self.assertIn('"blue", "green"', DETAILS_SYSTEM)
        # a query that DOES qualify the colour is left alone -- only the bare word is ambiguous
        self.assertIn("xanh dương", DETAILS_SYSTEM)
        self.assertIn("xanh lá", DETAILS_SYSTEM)

    def test_a_bare_xanh_detail_with_both_colours_in_same_en_is_not_a_conflict_on_a_green_frame(self):
        """What the reported candidate needed: `same_en` carrying both colours reaches the judge as one
        "Same:" line, so a frame the model sees as green is `exact`, not `conflict`, against a detail
        the query only called "xanh"."""
        detail = QueryDetails([{"vi": "một chiếc ô tô màu xanh", "terms_en": ["car"],
                                "meaning_en": "A blue or green car.",
                                "same_en": ["blue", "green", "teal"], "different_en": ["red", "white", "black"]}])

        def reply(images, prompt):
            self.assertIn("Same: blue; green; teal", prompt)
            return {"candidates": [{"id": "C1", "details": [
                {"detail": 0, "why": "a green car is visible", "verdict": "exact", "note_vi": "xe xanh lá"}]}]}

        found = ev.vision_summarize(VisionRecorder(vision_reply=reply), VisionSearch(), detail, [vitem("C1", 21.0)])
        self.assertEqual(found["C1"][0][0], "supported")


class CaptionSuspendedTests(unittest.TestCase):
    """The reported request: prioritise vision, stop asking a second model to re-describe a frame
    a caption model already described once, offline, without the query in mind."""

    DETAIL = QueryDetails([{"vi": "một con vật màu vàng", "terms_en": ["animal", "yellow"], "terms_vi": [],
                            "meaning_en": "A yellow animal.", "same_en": [], "different_en": []}])

    def caption_only_search(self):
        class CaptionOnly(Search):
            def caption_windows(self, video_id):
                return [{"shot_idx": 1, "window_idx": 0, "frame_idx": None, "start": 0.0, "end": 40.0,
                        "time": 20.0, "text": "A large yellow lion costume lies on the mat."}]

            def transcript(self, video_id):
                return []

            def ocr(self, video_id):
                return []
        return CaptionOnly()

    def test_the_caption_call_never_runs_while_vision_is_available(self):
        deepseek = Recorder()  # `_match_caption` would answer "exact" for everything by default
        found = summarize(deepseek, self.DETAIL, EvidenceStore(self.caption_only_search()),
                          [vitem("C1", 20.0)], vision=VisionRecorder(), search=VisionSearch())
        self.assertEqual([name for name, _ in deepseek.calls], [])                 # the caption prompt was never sent
        self.assertEqual(found["C1"]["checks"][0]["source"], "vision")             # vision covered it instead

    def test_turning_checklist_vision_off_brings_caption_back(self):
        deepseek = Recorder()
        found = summarize(deepseek, self.DETAIL, EvidenceStore(self.caption_only_search()), [item("C1", 20.0)])
        self.assertEqual([name for name, _ in deepseek.calls], ["match_caption"])
        self.assertEqual(found["C1"]["checks"][0]["source"], "caption")

    def test_a_detail_vision_has_no_opinion_on_still_gets_nothing_not_a_caption_fallback(self):
        """Suspended means suspended: a detail vision leaves silent stays not_mentioned, not caption's guess."""
        def only_index_0(images, prompt):
            return {"candidates": [{"id": "C1", "details": []}]}   # vision looked, said nothing about detail 0

        found = summarize(Recorder(), self.DETAIL, EvidenceStore(self.caption_only_search()), [vitem("C1", 20.0)],
                          vision=VisionRecorder(vision_reply=only_index_0), search=VisionSearch())
        self.assertEqual(found["C1"]["checks"][0]["status"], "not_mentioned")


class ProductionWiringTests(unittest.TestCase):
    """The candidate dicts `_kick_summaries` builds must carry what `vision_summarize` fetches by.

    Every other vision test here builds its own candidate items and so cannot catch a mismatch
    between what the orchestrator actually constructs and what `vision_summarize` reads: the
    checklist ran end to end in earlier manual checks only because a hand-built dict happened to
    include `frame_idx`, while `_kick_summaries` built `{"id", "video_id", "time"}` and nothing else
    -- every vision call in every real run silently fetched no image at all (`search.image` raised
    `KeyError` on the missing key, caught, and the chunk returned `None` before ever calling the
    model), and stayed silent because a "nothing to check" return logs nothing.
    """

    def test_a_human_run_actually_sends_the_frame_not_just_the_id_and_time(self):
        deepseek = VisionRecorder()
        review, seen = HumanReview(judge="human"), []

        def watch(snapshot):
            seen.append(snapshot)
            if snapshot["stage"] == "awaiting_review" and not snapshot["summarizing"]:
                review.finish()

        agent = make_agent(deepseek, ForbiddenVision(), VisionSearch(HITS), mode="fast", judge="human")
        agent.run(task_from_text("q-kis", "thuyền đỏ"), review=review, progress=watch)

        self.assertTrue(deepseek.vision_calls, "vision_summarize never actually called the model")
        images, _ = deepseek.vision_calls[0]
        self.assertTrue(all(data.startswith(b"jpeg:") for _, data in images))     # a real image, not a KeyError swallowed
        last = next(s for s in reversed(seen) if s["stage"] == "awaiting_review")
        self.assertTrue(any(c["summary"] and any(chk["source"] == "vision" for chk in c["summary"]["checks"])
                            for c in last["candidates"]))

    def test_the_checklist_reads_the_frames_with_the_runs_own_deepseek_client(self):
        deepseek = VisionRecorder()
        review = HumanReview(judge="human")

        def watch(snapshot):
            if snapshot["stage"] == "awaiting_review" and not snapshot["summarizing"]:
                review.finish()

        make_agent(deepseek, ForbiddenVision(), VisionSearch(HITS), mode="fast",
                   judge="human").run(task_from_text("q-kis", "thuyền đỏ"), review=review, progress=watch)
        self.assertTrue(deepseek.vision_calls)

    def test_checklist_vision_off_sends_no_frame_at_all(self):
        deepseek = VisionRecorder()
        review = HumanReview(judge="human")

        def watch(snapshot):
            if snapshot["stage"] == "awaiting_review" and not snapshot["summarizing"]:
                review.finish()

        make_agent(deepseek, ForbiddenVision(), VisionSearch(HITS), mode="fast", judge="human",
                  checklist_vision=False).run(task_from_text("q-kis", "thuyền đỏ"), review=review, progress=watch)
        self.assertEqual(deepseek.vision_calls, [])


class ChecklistTests(unittest.TestCase):
    def test_the_details_are_found_in_the_shots_before_the_frame_and_deep_in_a_caption(self):
        """The reported bug: a candidate at 21 s showed almost nothing supported."""
        deepseek = Recorder()
        found = summarize(deepseek, DETAILS, EvidenceStore(Search()), [item("C2", 21.0)])

        prompt = deepseek.prompt("match_caption")
        self.assertIn("A woman wearing a white apron stands beside a vase of red ginger flowers.", prompt)
        self.assertNotIn("far away shot", prompt)
        self.assertNotIn("A pan sizzles", prompt)      # holds no term: never sent
        self.assertLess(len(prompt), 3000)            # excerpts, not the ~2,300-character paragraph
        checks = found["C2"]["checks"]
        self.assertEqual((checks[0]["status"], checks[0]["source"], checks[0]["at"]), ("supported", "caption", 12.0))
        self.assertIn("white apron", checks[0]["quote"])
        self.assertEqual(checks[1]["status"], "supported")
        self.assertEqual(checks[2], {"detail": 2, "status": "not_mentioned", "source": None, "at": None, "quote": "", "note": ""})
        self.assertEqual(found["C2"]["summary"], "Khớp hẳn 2/3 chi tiết: cô gái đeo tạp dề trắng; đặt bốn con nghêu lên dĩa")
        self.assertEqual(checks[0]["note"], "khớp")

    def test_captions_are_judged_in_english_against_the_details_english_meaning(self):
        deepseek = Recorder()
        summarize(deepseek, DETAILS, EvidenceStore(Search()), [item("C2", 21.0)])

        prompt = deepseek.prompt("match_caption")
        self.assertIn("0. A woman wearing a white apron.\n   Same: woman in a white apron\n   Different: red apron; black apron\n", prompt)
        self.assertNotIn("cô gái đeo tạp dề", prompt)             # the Vietnamese phrase is not in the English call
        self.assertIn("1. đặt bốn con nghêu lên dĩa\n   e", prompt)   # a detail with no meaning falls back to its text
        self.assertRegex(prompt, r"\n   e\d+ \[\d+s\] ")           # excerpts are numbered, timed, and only captions
        self.assertNotIn("hôm nay nấu món nghêu", prompt)         # speech is not here
        self.assertNotIn("Bạn", MATCH_CAPTION_SYSTEM)              # the rubric itself is English
        for rule in ("Same", "Different", "conflict", "the action applies to another object", "note_vi"):
            self.assertIn(rule, MATCH_CAPTION_SYSTEM)
        self.assertNotIn("tía ≈", MATCH_CAPTION_SYSTEM + MATCH_SYSTEM)   # the old "colours may drift" licence

    def test_speech_and_ocr_are_judged_in_vietnamese_against_the_details_as_written(self):
        deepseek = Recorder()
        summarize(deepseek, DETAILS, EvidenceStore(Search()), [item("C2", 21.0)])

        prompt = deepseek.prompt("match")
        self.assertIn("1. đặt bốn con nghêu lên dĩa\n", prompt)
        self.assertIn("[speech 30s] hôm nay nấu món nghêu hấp sả", prompt)
        self.assertIn("[ocr 20s] NGHÊU HẤP SẢ", prompt)
        self.assertNotIn("Same:", prompt)                          # no English meaning lines here
        self.assertNotIn("Hands place four clams", prompt)        # no captions here
        self.assertNotIn("\n0. ", prompt)                          # the apron detail has no speech or OCR excerpt: not asked
        self.assertIn("LỜI THOẠI", MATCH_SYSTEM)

    def test_what_speech_says_reaches_the_checklist_from_its_own_call(self):
        def heard(prompt):
            speech = re.search(r"^\s+(e\d+) \[speech", prompt, re.MULTILINE).group(1)
            return {"details": [{"detail": 0, "judgements": [{"id": speech, "why": "nói món nghêu", "verdict": "exact"}]}]}

        found = summarize(Recorder(caption_judge=lambda p: {"details": []}, text_judge=heard), DISH, EvidenceStore(Search()), [item("C2", 32.0)])
        check = found["C2"]["checks"][0]
        self.assertEqual((check["status"], check["source"], check["note"]), ("supported", "speech", "nói món nghêu"))

    def test_the_count_guard_holds_for_speech_too(self):
        def heard(prompt):
            speech = re.search(r"^\s+(e\d+) \[speech", prompt, re.MULTILINE).group(1)
            return {"details": [{"detail": 1, "judgements": [{"id": speech, "why": "w", "verdict": "exact"}]}]}

        found = summarize(Recorder(caption_judge=lambda p: {"details": []}, text_judge=heard), DETAILS, EvidenceStore(Search()), [item("C2", 32.0)])
        self.assertEqual(found["C2"]["checks"][1]["status"], "partial")   # "bốn con" is in the detail, no number is in the speech

    def test_the_reviewer_reads_vietnamese_so_a_caption_verdict_carries_a_vietnamese_note(self):
        found = summarize(Recorder(), DETAILS, EvidenceStore(Search()), [item("C2", 21.0)])
        self.assertEqual(found["C2"]["checks"][0]["note"], "khớp")     # note_vi, not the English why

        def no_note(prompt):
            ids = re.findall(r"^\s+(e\d+) \[", prompt, re.MULTILINE)
            return {"details": [{"detail": 0, "judgements": [{"id": ids[0], "why": "only English", "verdict": "exact"}]}]}

        found = summarize(Recorder(caption_judge=no_note), DETAILS, EvidenceStore(Search()), [item("C2", 21.0)])
        self.assertEqual(found["C2"]["checks"][0]["note"], "only English")   # falls back rather than showing nothing

    def test_one_language_failing_keeps_what_the_other_judged_and_both_failing_is_no_summary(self):
        def heard(prompt):
            speech = re.search(r"^\s+(e\d+) \[speech", prompt, re.MULTILINE).group(1)
            return {"details": [{"detail": 0, "judgements": [{"id": speech, "why": "w", "verdict": "exact"}]}]}

        store = EvidenceStore(Search())
        only_speech = summarize(Recorder(fail={"match_caption"}, text_judge=heard), DISH, store, [item("C2", 32.0)])
        self.assertEqual(only_speech["C2"]["checks"][0]["status"], "supported")   # the captions could not be read, the speech could
        only_captions = summarize(Recorder(fail={"match"}), DETAILS, store, [item("C2", 21.0)])
        self.assertEqual(only_captions["C2"]["checks"][0]["status"], "supported")
        self.assertEqual(summarize(Recorder(fail={"match", "match_caption"}), DETAILS, store, [item("C2", 21.0)]), {})

    def test_captions_come_from_the_disk_windows_not_the_caption_index(self):
        search = Search()   # `captions` (the index) raises if it is ever touched
        summarize(Recorder(), DETAILS, EvidenceStore(search), [item("C2", 21.0)])
        self.assertIn(("caption_windows", "L01_V001"), search.calls)

    def test_a_candidate_far_from_the_evidence_does_not_inherit_it(self):
        found = summarize(Recorder(), DETAILS, EvidenceStore(Search()), [item("C1", 21.0), item("C9", 300.0)])
        self.assertEqual(found["C1"]["checks"][0]["status"], "supported")
        self.assertEqual([c["status"] for c in found["C9"]["checks"]], ["supported", "not_mentioned", "not_mentioned"])
        self.assertEqual(found["C9"]["checks"][0]["at"], 300.0)   # its own neighbourhood's apron, not the one at 12 s

    def test_the_judge_decides_and_a_verdict_it_could_not_have_given_is_ignored(self):
        def judge(prompt):
            ids = dict(re.findall(r"^\s+(e\d+) \[[^\]]*\] (.*)$", prompt, re.MULTILINE))
            apron = next(i for i, text in ids.items() if "white apron" in text)
            clams = next(i for i, text in ids.items() if "four clams" in text)
            return {"details": [
                {"detail": 0, "judgements": [{"id": apron, "why": "đúng tạp dề trắng", "verdict": "exact"},
                                             {"id": "e999", "why": "never shown", "verdict": "exact"}]},
                {"detail": 1, "judgements": [{"id": clams, "why": "nói 2 con, không phải 4", "verdict": "conflict"}]},
                {"detail": 2, "judgements": [{"id": apron, "why": "x", "verdict": "definitely"}]},   # not a verdict
                {"detail": "x", "judgements": []}, "junk"]}

        found = summarize(Recorder(caption_judge=judge), DETAILS, EvidenceStore(Search()), [item("C2", 21.0)])
        first, second, third = found["C2"]["checks"]
        self.assertEqual((first["status"], first["source"], first["note"]), ("supported", "caption", "đúng tạp dề trắng"))
        self.assertEqual((second["status"], second["quote"], second["note"]),
                         ("contradicted", "Hands place four clams on a white plate.", "nói 2 con, không phải 4"))
        self.assertEqual(third["status"], "not_mentioned")
        self.assertEqual(found["C2"]["summary"], "Khớp hẳn 1/3 chi tiết: cô gái đeo tạp dề trắng. Mâu thuẫn: đặt bốn con nghêu lên dĩa")

    def test_partial_is_what_a_sentence_about_the_right_subject_gets_and_it_is_not_counted_as_a_match(self):
        def judge(prompt):
            ids = re.findall(r"^\s+(e\d+) \[", prompt, re.MULTILINE)
            return {"details": [{"detail": 0, "judgements": [
                {"id": ids[0], "why": "chỉ nói áo, không nói tạp dề", "verdict": "partial"}]}]}

        found = summarize(Recorder(caption_judge=judge), DETAILS, EvidenceStore(Search()), [item("C2", 21.0)])
        self.assertEqual(found["C2"]["checks"][0]["status"], "partial")
        self.assertEqual(found["C2"]["summary"], "Khớp hẳn 0/3 chi tiết. Gần đúng: cô gái đeo tạp dề trắng")

    def test_a_count_the_sentence_does_not_state_caps_the_verdict_at_partial(self):
        """The reported bug: "4 con X" was ticked for a plate holding two shrimp."""
        windows = [{"shot_idx": 1, "window_idx": 0, "frame_idx": None, "start": 12.0, "end": 19.0, "time": 15.0,
                    "text": "A second white plate holds two cooked shrimp on the counter."}]

        class TwoShrimp(Search):
            def caption_windows(self, video_id):
                return windows

        details = QueryDetails([{"vi": "đặt lên 1 dĩa trắng 4 con tôm", "terms_en": ["shrimp", "plate"], "terms_vi": []}])
        found = summarize(Recorder(), details, EvidenceStore(TwoShrimp()), [item("C1", 15.0)])   # the judge says exact
        check = found["C1"]["checks"][0]
        self.assertEqual(check["status"], "partial")
        self.assertIn("chi tiết cần 4, đoạn nói 2", check["note"])

        windows[0]["text"] = "A second white plate holds four cooked shrimp on the counter."
        found = summarize(Recorder(), details, EvidenceStore(TwoShrimp()), [item("C1", 15.0)])
        self.assertEqual(found["C1"]["checks"][0]["status"], "supported")

    def test_partial_beats_a_conflict_and_a_match_beats_both(self):
        windows = [
            {"shot_idx": 1, "window_idx": 0, "frame_idx": None, "start": 5.0, "end": 9.0, "time": 7.0, "text": "A woman wears a red apron."},
            {"shot_idx": 2, "window_idx": 0, "frame_idx": None, "start": 10.0, "end": 14.0, "time": 12.0, "text": "A woman in an apron stands there."},
            {"shot_idx": 3, "window_idx": 0, "frame_idx": None, "start": 15.0, "end": 19.0, "time": 17.0, "text": "A woman wears a white apron."},
        ]

        class Three(Search):
            def caption_windows(self, video_id):
                return windows

        def judge_with(*verdicts):
            def judge(prompt):
                ids = re.findall(r"^\s+(e\d+) \[", prompt, re.MULTILINE)
                return {"details": [{"detail": 0, "judgements": [
                    {"id": i, "why": "w", "verdict": v} for i, v in zip(ids, verdicts)]}]}
            return judge

        details = QueryDetails([{"vi": "áo tạp dề trắng", "terms_en": ["apron"], "terms_vi": []}])
        for verdicts, expected in ((("conflict", "partial", "exact"), "supported"), (("conflict", "partial", "unrelated"), "partial"),
                                   (("conflict", "unrelated", "unrelated"), "contradicted")):
            found = summarize(Recorder(caption_judge=judge_with(*verdicts)), details, EvidenceStore(Three()), [item("C1", 12.0)])
            self.assertEqual(found["C1"]["checks"][0]["status"], expected, verdicts)

    def test_a_video_with_words_but_no_term_needs_no_call_and_says_so(self):
        deepseek = Recorder()
        nothing = QueryDetails([{"vi": "kỳ lân", "terms_en": ["unicorn"], "terms_vi": ["kỳ lân"]}])
        found = summarize(deepseek, nothing, EvidenceStore(Search()), [item("C1", 21.0)])
        self.assertEqual(deepseek.calls, [])
        self.assertEqual(found["C1"]["checks"][0]["status"], "not_mentioned")
        self.assertIn("Chưa thấy chi tiết nào", found["C1"]["summary"])

    def test_a_failed_call_or_a_video_with_no_words_is_no_summary_not_an_error(self):
        store = EvidenceStore(Search())
        self.assertEqual(summarize(Recorder(fail={"match", "match_caption"}), DETAILS, store, [item("C1", 21.0)]), {})
        self.assertEqual(summarize(Recorder(), QueryDetails(), store, [item("C1", 21.0)]), {})
        empty = Recorder()
        self.assertEqual(summarize(empty, DETAILS, EvidenceStore(object()), [item("C1", 21.0)]), {})
        self.assertEqual(empty.calls, [])  # nothing to read, nothing to pay for

    def test_one_call_per_video_and_each_reported_as_it_finishes(self):
        items = [item("C1", 20), item("C2", 30), item("C3", 40, "L02_V001"), item("C4", 300), item("C5", 25)]
        deepseek, seen = Recorder(), []
        found = summarize_all(deepseek, DETAILS, EvidenceStore(Search()), items,
                              on_batch=lambda part: seen.append(sorted(part)))
        self.assertEqual(sorted(found), ["C1", "C2", "C3", "C4", "C5"])
        self.assertEqual(sorted(seen), [["C1", "C2", "C5"], ["C3"], ["C4"]])
        self.assertEqual(len([c for c in deepseek.calls if c[0] == "match_caption"]), 3)


class SuggestTests(unittest.TestCase):
    def test_answers_come_with_the_sentence_they_were_read_from_and_the_nearest_candidate(self):
        found = suggest_answers(Recorder(), "X là con gì?", DETAILS, EvidenceStore(Search()), [item("C1", 33.0), item("C2", 200.0)])
        self.assertEqual([(a["answer"], a["confidence"], a["source"]) for a in found][0][:2], ("nghêu", 8.0))
        self.assertIn(found[0]["cid"], ("C1",))
        self.assertTrue(found[0]["quote"])

    def test_no_words_to_search_or_a_failed_call_is_no_suggestion(self):
        self.assertEqual(suggest_answers(Recorder(fail={"suggest"}), "q", DETAILS, EvidenceStore(Search()), [item("C1", 33.0)]), [])
        self.assertEqual(suggest_answers(Recorder(), "q", DETAILS, EvidenceStore(object()), [item("C1", 33.0)]), [])


def summary_at(status, at):
    return {"summary": "x", "checks": [{"detail": 0, "status": status, "source": "caption", "at": at, "quote": "", "note": ""}]}


def summary(*statuses):
    return {"summary": "x", "checks": [{"detail": i, "status": status, "source": None, "at": None, "quote": "", "note": ""}
                                       for i, status in enumerate(statuses)]}


class EvidenceScoreTests(unittest.TestCase):
    def test_only_support_counts_a_contradiction_lowers_nothing(self):
        """Captions get colours and counts wrong; on ten queries ranking by contradiction hurt more than it helped."""
        weights = [1.0, 1.0]
        said = ev.evidence_of(summary("supported", "supported"), weights)
        half = ev.evidence_of(summary("partial", "not_mentioned"), weights)
        silent = ev.evidence_of(summary("not_mentioned", "not_mentioned"), weights)
        against = ev.evidence_of(summary("contradicted", "not_mentioned"), weights)

        self.assertEqual((said["share"], said["score"], said["tier"]), (1.0, 0.5, 0))    # score = half own + half the video's
        self.assertEqual((half["share"], half["tier"]), (0.125, 0))                      # a partial is worth a quarter
        self.assertEqual((silent["share"], silent["score"], silent["tier"]), (0.0, 0.0, 1))
        self.assertEqual(against, silent)                                                # the same as saying nothing

    def test_a_check_counts_for_less_the_further_its_shot_is_from_the_moment(self):
        weights = [1.0]
        self.assertEqual(ev.evidence_of(summary_at("supported", None), weights, time=30.0)["share"], 1.0)   # no time: full weight
        self.assertEqual(ev.evidence_of(summary_at("supported", 30.0), weights, time=30.0)["share"], 1.0)
        self.assertAlmostEqual(ev.evidence_of(summary_at("supported", 50.0), weights, time=30.0)["share"], 0.5, places=3)
        self.assertAlmostEqual(ev.evidence_of(summary_at("supported", 130.0), weights, time=30.0)["share"], 1 / 6, places=3)
        self.assertAlmostEqual(ev.proximity(20.0), 0.5)
        self.assertEqual(ev.proximity(0.0), 1.0)

    def test_a_video_is_credited_with_every_detail_any_of_its_read_moments_says(self):
        """A story is spread over shots: no one frame says it all, the right video says most of it somewhere."""
        weights = [1.0, 1.0, 2.0]
        coverage = ev.video_coverage([
            ("L01_V001", summary("supported", "not_mentioned", "not_mentioned")),
            ("L01_V001", summary("not_mentioned", "partial", "supported")),
            ("L01_V001", summary("contradicted", "not_mentioned", "not_mentioned")),      # says nothing for the video
            ("L01_V002", summary("partial", "not_mentioned", "not_mentioned")),
        ], weights)
        self.assertEqual(coverage["L01_V001"], (1.0 + 0.25 + 2.0) / 4)
        self.assertEqual(coverage["L01_V002"], 0.25 / 4)
        self.assertNotIn("L01_V003", coverage)

    def test_the_video_lifts_a_moment_that_says_little_itself(self):
        weights = [1.0, 1.0]
        alone = ev.evidence_of(summary("partial", "not_mentioned"), weights, coverage=0.0)
        in_a_video_that_says_it = ev.evidence_of(summary("partial", "not_mentioned"), weights, coverage=1.0)
        self.assertEqual((alone["video"], in_a_video_that_says_it["video"]), (0.0, 1.0))
        self.assertGreater(in_a_video_that_says_it["score"], alone["score"])
        self.assertEqual(in_a_video_that_says_it["share"], alone["share"])                # the bar is the moment's own

    def test_a_candidate_nobody_read_has_no_evidence_and_its_video_can_still_speak_for_a_read_one(self):
        self.assertIsNone(ev.evidence_of(None, [1.0]))
        self.assertIsNone(ev.evidence_of(summary("supported"), []))
        self.assertEqual(ev.evidence_of(summary("not_mentioned"), [1.0], coverage=0.5)["tier"], 0)   # its video says something

    def test_a_detail_nearly_every_candidate_says_weighs_little_once_enough_have_been_read(self):
        few = [summary("supported", "not_mentioned")] * 5
        weights, common = ev.detail_weights(few, 2)
        self.assertEqual((weights, common), ([1.0, 1.0], [False, False]))     # five is too few to say

        many = [summary("supported", "supported" if n == 0 else "not_mentioned") for n in range(8)]
        weights, common = ev.detail_weights(many + [None], 2)                 # an unread candidate is not counted
        self.assertEqual(common, [True, False])
        self.assertEqual(weights, [ev.MIN_WEIGHT, 0.875])                     # 1 - 1/8: only the first candidate says detail 1
        # said by everyone: the floor, not zero, so it still counts for something
        self.assertEqual(ev.detail_weights([summary("supported")] * 8, 1)[0], [ev.MIN_WEIGHT])

    def test_the_reported_fix_a_guessable_detail_counts_for_less_from_the_start(self):
        """p2-1: "a yellow animal" is guessable as the lion before any candidate is read; the story-specific
        details ("carries the pumpkin away", "wakes the animal") should already outweigh it."""
        entries = [{"vi": "một con vật màu vàng", "distinctiveness": 0.3},
                   {"vi": "mang vật giống trái bí đỏ đi giấu", "distinctiveness": 0.9}]
        weights, _ = ev.detail_weights([], 2, entries)                        # nobody has been read yet
        self.assertEqual(weights, [round(ev._prior_weight(entries[0]), 4), round(ev._prior_weight(entries[1]), 4)])
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(weights[0], 0.44, places=2)                    # 0.2 + 0.8*0.3
        self.assertAlmostEqual(weights[1], 0.92, places=2)                    # 0.2 + 0.8*0.9

    def test_the_prior_and_the_empirical_factor_both_apply_once_there_is_enough_data(self):
        entries = [{"vi": "a", "distinctiveness": 0.5}, {"vi": "b", "distinctiveness": 0.5}]
        read = [summary("supported", "supported" if n == 0 else "not_mentioned") for n in range(8)]  # detail 0: everyone; 1: one
        weights, common = ev.detail_weights(read, 2, entries)
        prior = ev._prior_weight(entries[0])
        self.assertAlmostEqual(weights[0], round(prior * ev.MIN_WEIGHT, 4))    # common AND generic: both factors shrink it
        self.assertAlmostEqual(weights[1], round(prior * 0.875, 4))
        self.assertEqual(common, [True, False])

    def test_with_no_entries_every_detail_gets_the_same_prior_as_before(self):
        self.assertEqual(ev.detail_weights([summary("supported")] * 8, 1), ev.detail_weights([summary("supported")] * 8, 1, None))




class TiesTests(unittest.TestCase):
    def test_within_a_band_own_share_settles_it_before_retrieval_does(self):
        """Two candidates whose blended evidence differs by a hair must not swap places on THAT -- but
        the one whose own moment supports more of the query still leads; only a further tie on own
        share falls back to retrieval order.
        """
        from agent.orchestrator import _Run

        class Candidate:
            def __init__(self, cid):
                self.cid, self.verdict, self.video_id, self.time = cid, None, "L01_V001", 0.0

        class Stub:
            details = QueryDetails([{"vi": "a"}, {"vi": "b"}, {"vi": "c"}, {"vi": "d"}])
            summaries = {"C1": summary("partial", "partial", "not_mentioned", "not_mentioned"),         # own .125
                         "C2": summary("partial", "partial", "partial", "not_mentioned"),               # own .1875: inside the band, and ahead on share
                         "C3": summary("supported", "supported", "supported", "not_mentioned")}         # own .75: bands up
            _split = staticmethod(_Run._split)

        ranking = [Candidate("C1"), Candidate("C2"), Candidate("C3")]      # retrieval order
        ordered = _Run._by_evidence(Stub(), ranking, [1.0] * 4, {})
        self.assertEqual([c.cid for c in ordered], ["C3", "C2", "C1"])     # C2 leads C1 on share, not retrieval
        self.assertGreaterEqual(ev.TIE_BAND, 0.1)                            # evidence is coarse on purpose

    def test_the_reported_case_a_thinner_video_does_not_outrank_a_fuller_own_moment(self):
        """p2-6, with the numbers the reported run measured. Two broadcasts of the same real event:
        L21_V018 has no red circle at all -- every one of its candidates reads `not_mentioned` for that
        detail, never a false "supported" -- but it reads FULLER on the other details, so its video
        coverage (0.7549) passes L22_V024's (0.6977) and its own moment (.625) covers more of them.
        Blended: .690 against .677 -- the duplicate is ahead on the blend and found first by retrieval,
        and the two land in the same band, where retrieval order used to settle the tie and put the
        wrong video first. Only the gold moment's own share breaks it (.656 against .625), and that is
        what the tie must be settled by: video coverage lifts a moment that says little itself, it does
        not overrule one that already says more.
        """
        from agent.orchestrator import _Run

        class Candidate:
            def __init__(self, cid, video_id):
                self.cid, self.verdict, self.video_id, self.time = cid, None, video_id, 0.0

        class Stub:
            details = QueryDetails([{"vi": d} for d in "abcdefgh"])
            summaries = {
                # own .65625: five of the query's eight details, including the circle the other lacks
                "C14": summary("supported", "supported", "supported", "supported", "supported",
                               "partial", "not_mentioned", "not_mentioned"),
                # own .625: five full too, but none of them the circle -- the thinner own moment
                "C6": summary("supported", "supported", "supported", "supported", "supported",
                              "not_mentioned", "not_mentioned", "not_mentioned"),
            }
            _split = staticmethod(_Run._split)

        coverage = {"L22_V024": 0.6977, "L21_V018": 0.7549}                     # the reported run's
        ranking = [Candidate("C6", "L21_V018"), Candidate("C14", "L22_V024")]   # retrieval found L21_V018 first
        ordered = _Run._by_evidence(Stub(), ranking, [1.0] * 8, coverage)
        self.assertEqual([c.cid for c in ordered], ["C14", "C6"])   # the fuller own moment leads anyway

        blended = {cid: 0.5 * share + 0.5 * coverage[video] for cid, share, video in
                   (("C14", 0.65625, "L22_V024"), ("C6", 0.625, "L21_V018"))}
        self.assertGreater(blended["C6"], blended["C14"])           # ...even though the blend says otherwise
        self.assertEqual(round(blended["C6"] / ev.TIE_BAND), round(blended["C14"] / ev.TIE_BAND))  # one band



class OrderByEvidenceTests(unittest.TestCase):
    """The searches only find candidates; the words around each decide what leads.

    The agent's own (retrieval) order is L01_V001, L01_V002, L01_V003. V003's captions say the detail,
    V002's contradict it, V001's say nothing.
    """

    CAPTIONS = {
        "L01_V001": [{"shot_idx": 1, "window_idx": 0, "frame_idx": None, "start": 5.0, "end": 15.0, "time": 10.0,
                      "text": "A kitchen counter with a steel sink and jars."}],
        "L01_V002": [{"shot_idx": 1, "window_idx": 0, "frame_idx": None, "start": 15.0, "end": 25.0, "time": 20.0,
                      "text": "A woman stands in a red apron by the stove."}],
        "L01_V003": [{"shot_idx": 1, "window_idx": 0, "frame_idx": None, "start": 25.0, "end": 35.0, "time": 30.0,
                      "text": "A woman stands in a white apron by the stove."}],
    }
    APRON = QueryDetails([{"vi": "cô gái đeo tạp dề trắng", "terms_en": ["apron"], "terms_vi": [],
                           "meaning_en": "A woman wearing a white apron.", "same_en": [], "different_en": ["red apron"]}])

    class PerVideo(Search):
        def caption_windows(self, video_id):
            return OrderByEvidenceTests.CAPTIONS[video_id]

        def transcript(self, video_id):
            return []

        def ocr(self, video_id):
            return []

    @staticmethod
    def judge(prompt):
        blocks = re.findall(r"^\s+(e\d+) \[\d+s\] (.*)$", prompt, re.MULTILINE)
        return {"details": [{"detail": 0, "judgements": [
            {"id": eid, "why": "w", "note_vi": "n", "verdict": "conflict" if "red apron" in text else "exact"} for eid, text in blocks]}]}

    def run_human(self, script, hits=HITS):
        """`checklist_vision=False`: this fixture's captions carry the evidence (see the class
        docstring), so vision -- on by default, and this fixture has no working vision fake -- is
        turned off here to test the caption pathway it exercises specifically.
        """
        class Apron(Recorder):
            def _details(self, _):
                return {"details": [dict(d) for d in OrderByEvidenceTests.APRON.details], "question": None}

        review, seen = HumanReview(judge="human"), []

        def watch(snapshot):
            seen.append(snapshot)
            if snapshot["stage"] == "awaiting_review":
                script(snapshot, review)

        agent = make_agent(Apron(caption_judge=self.judge), ForbiddenVision(), self.PerVideo(hits), mode="fast",
                           judge="human", checklist_vision=False)
        return agent.run(task_from_text("q-kis", "cô gái đeo tạp dề trắng"), review=review, progress=watch), seen

    def test_what_the_evidence_supports_leads_and_a_contradiction_lowers_nothing(self):
        def script(snapshot, review):
            if snapshot["summarizing"] or not all(c["summary"] for c in snapshot["candidates"]):
                return
            review.finish()

        result, seen = self.run_human(script)
        ranked = next(s for s in reversed(seen) if s["stage"] == "awaiting_review")

        self.assertEqual([c["video_id"] for c in ranked["candidates"]], ["L01_V003", "L01_V001", "L01_V002"])
        self.assertEqual([c["agent_rank"] for c in ranked["candidates"]], [3, 1, 2])   # retrieval had them the other way
        self.assertEqual([c["evidence"]["tier"] for c in ranked["candidates"]], [0, 1, 1])   # V002's contradiction is not a demotion
        self.assertEqual([c["summary"]["checks"][0]["status"] for c in ranked["candidates"]], ["supported", "not_mentioned", "contradicted"])
        self.assertEqual([c.video_id for c in result.ranking], ["L01_V003", "L01_V001", "L01_V002"])   # and so does the export

    def test_the_order_holds_still_until_the_checklist_has_been_read(self):
        _, seen = self.run_human(lambda snapshot, review: review.finish() if not snapshot["summarizing"] else None)
        first = next(s for s in seen if s["stage"] == "awaiting_review")
        self.assertTrue(first["summarizing"])
        self.assertEqual([c["video_id"] for c in first["candidates"]], ["L01_V001", "L01_V002", "L01_V003"])   # retrieval, for now

    def test_what_the_reviewer_settled_is_never_moved_by_the_evidence(self):
        def script(snapshot, review):
            review.decide("C1", "accept")         # V001: the evidence says nothing about it
            review.decide("C3", "reject")         # V003: the evidence supports it
            if not snapshot["summarizing"]:
                review.finish()

        result, _ = self.run_human(script)
        self.assertEqual([c.video_id for c in result.ranking], ["L01_V001", "L01_V002", "L01_V003"])

    def test_a_candidate_with_no_checklist_at_all_is_ordered_by_retrieval_among_its_kind(self):
        class NothingRead(Search):
            def caption_windows(self, video_id):
                return []

        review = HumanReview(judge="human")
        agent = make_agent(Recorder(), ForbiddenVision(), NothingRead(HITS), mode="fast", judge="human")

        def watch(snapshot):
            if snapshot["stage"] == "awaiting_review" and not snapshot["summarizing"]:
                review.finish()

        result = agent.run(task_from_text("q-kis", "x"), review=review, progress=watch)
        self.assertEqual([c.video_id for c in result.ranking], ["L01_V001", "L01_V002", "L01_V003"])

    def test_a_snapshot_carries_the_weights_the_bar_and_the_dimming_use(self):
        _, seen = self.run_human(lambda snapshot, review: review.finish() if not snapshot["summarizing"] else None)
        last = next(s for s in reversed(seen) if s["stage"] == "awaiting_review")
        self.assertEqual(last["detail_weights"], [1.0])      # three read candidates are too few to call a detail common
        self.assertEqual(last["common_details"], [False])


class HumanRunChecklistTests(unittest.TestCase):
    def run_human(self, task, deepseek, script):
        review, seen = HumanReview(judge="human"), []

        def watch(snapshot):
            seen.append(snapshot)
            if snapshot["stage"] == "awaiting_review":
                script(snapshot, review)

        # `checklist_vision=False`: this class exercises the caption channel specifically
        # (`Recorder._match_caption`), and its fake deepseek has no `.judge()` vision method.
        agent = make_agent(deepseek, ForbiddenVision(), Search(HITS), mode="fast", judge="human", checklist_vision=False)
        return agent.run(task, review=review, progress=watch), seen

    def test_candidates_arrive_with_a_checklist_against_the_query_details(self):
        def script(snapshot, review):
            if snapshot["summarizing"] or len(snapshot["candidates"]) and not snapshot["candidates"][0]["summary"]:
                return  # keep waiting: the person is still reading
            review.finish()

        result, seen = self.run_human(task_from_text("q-kis", "cô gái đeo tạp dề"), Recorder(), script)
        last = next(s for s in reversed(seen) if s["stage"] == "awaiting_review")
        self.assertEqual(last["details"], DETAILS.texts)      # plain strings for the view
        first = last["candidates"][0]
        self.assertEqual([c["status"] for c in first["summary"]["checks"]][:2], ["supported", "supported"])
        self.assertEqual(first["summary"]["checks"][0]["source"], "caption")
        self.assertIsNone(result.error)

    def test_the_words_lit_in_the_view_include_the_words_the_checklist_looked_for(self):
        def script(snapshot, review):
            if snapshot["details"]:
                review.finish()

        _, seen = self.run_human(task_from_text("q-kis", "cô gái đeo tạp dề"), Recorder(), script)
        keywords = next(s for s in reversed(seen) if s["details"])["plan"]["keywords"]
        self.assertIn("tạp dề", keywords["vi"])
        self.assertIn("white apron", keywords["en"])

    def test_no_deepseek_evidence_means_no_summaries_and_no_delay(self):
        deepseek = Recorder(fail={"details"})
        result, seen = self.run_human(task_from_text("q-kis", "x"), deepseek,
                                      lambda snapshot, review: (not snapshot["summarizing"]) and review.finish())
        self.assertEqual([name for name, _ in deepseek.calls], ["details"])
        self.assertTrue(all(c["summary"] is None for s in seen for c in s["candidates"]))
        self.assertIsNone(result.error)

    def test_a_qa_run_offers_answers_and_uses_one_when_the_person_submits_without_typing(self):
        task = task_from_text("q-qa", "X là con gì?")
        plan = {**PLAN, "question_vi": "X là con gì?", "answer_source": "speech"}

        def script(snapshot, review):
            if snapshot["suggested_answers"]:
                review.finish()

        result, seen = self.run_human(task, Recorder(plan), script)
        offered = next(s for s in seen if s.get("suggested_answers"))["suggested_answers"]
        self.assertEqual((offered[0]["answer"], offered[0]["source"]), ("nghêu", "caption"))
        self.assertEqual((result.answer, result.answer_evidence), ("nghêu", "suggested from caption (unconfirmed)"))

    def test_a_typed_answer_beats_a_suggestion(self):
        task = task_from_text("q-qa", "X là con gì?")
        plan = {**PLAN, "question_vi": "X là con gì?", "answer_source": "speech"}

        def script(snapshot, review):
            if snapshot["suggested_answers"]:
                review.set_answer("sò")
                review.finish()

        result, _ = self.run_human(task, Recorder(plan), script)
        self.assertEqual((result.answer, result.answer_evidence), ("sò", "typed by the reviewer"))


if __name__ == "__main__":
    unittest.main()
