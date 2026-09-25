import re
import unittest

from agent.checker import (
    CHECK_CONTEXT_MAX, CHECK_IMAGE_WIDTH, CHECK_SAID_CHARS, CHECK_SCHEMA, CheckError, CheckQuery,
    CheckTarget, _said, check_moment, check_schema,
)
from agent.evidence import EvidenceStore, QueryDetails

DETAILS = QueryDetails([
    {"vi": "4 con nghêu trên đĩa trắng", "terms_en": ["clam", "plate"], "terms_vi": ["nghêu", "đĩa"],
     "meaning_en": "four clams on a white plate", "same_en": [], "different_en": [],
     "distinctiveness": 0.9},
    {"vi": "người phụ nữ mặc tạp dề", "terms_en": ["apron"], "terms_vi": ["tạp dề"],
     "meaning_en": "a woman wearing an apron", "same_en": [], "different_en": [],
     "distinctiveness": 0.3},
])
FPS = 25.0
TARGET = CheckTarget("L26_V433", 12000, 480.0)


class FakeSearch:
    """Enough of `SearchClient` for the checker: shots, keyframes and their pixels."""

    def __init__(self, shots=None, fail_images=False):
        self._shots = shots if shots is not None else _shots()
        self.fail_images = fail_images
        self.asked = []

    def shots(self, video_id):
        return self._shots

    def neighbours(self, video_id, frame_idx, radius=6):
        return [key for shot in self._shots for key in shot["keyframes"]]

    def image(self, video_id, frame_idx, width=640):
        self.asked.append((video_id, int(frame_idx), width))
        if self.fail_images:
            raise RuntimeError("ffmpeg failed")
        return f"jpeg-{frame_idx}".encode()

    # the three EvidenceStore kinds
    def transcript(self, video_id):
        return [{"start": 478.0, "end": 484.0, "text": "Chúng ta xếp hai con nghêu lên đĩa."},
                {"start": 484.0, "end": 489.0, "text": "Chị mặc tạp dề đứng bên bếp."}]

    def ocr(self, video_id):
        return [{"start": 470.0, "end": 500.0, "text": "MÓN NGON MỖI NGÀY"}]

    def caption_windows(self, video_id):
        return [{"start": 470.0, "end": 500.0, "text": "A woman in an apron holds a white plate."}]


def _shots(length=20.0, span=400.0):
    """Shots of `length` seconds either side of the target, with three keyframes each.

    The clock and the frame numbers are one video at `FPS`, and the shot boundaries fall on
    multiples of `length` from the first one, so `TARGET` always lands on a shot start and is
    genuinely inside the shot the checker will find for it.
    """
    shots, start, index = [], max(0.0, TARGET.time - span), 0
    while start < TARGET.time + span:
        end = start + length
        start_frame, end_frame = int(start * FPS), int(end * FPS) - 1
        shots.append({
            "index": index, "start": start, "end": end,
            "start_frame": start_frame, "end_frame": end_frame,
            "keyframes": [{"video_id": "L26_V433", "frame_idx": start_frame + int(offset * FPS),
                           "time": start + offset}
                          for offset in (0.0, length / 3, 2 * length / 3)],
        })
        start, index = end, index + 1
    return shots


class FakeJudge:
    """A model that records what it was shown and answers whatever it was told to.

    `text_verdicts` is `{detail index: verdict}`; the excerpt ids are read back out of the
    prompt rather than guessed, because they are numbered by `select_excerpts` and which
    sentences it picks is part of what these tests are checking.
    """

    def __init__(self, vision_reply=None, text_verdicts=None, fail=None):
        self.vision_reply = vision_reply if vision_reply is not None else {
            "details": [{"detail": 0, "why": "four clams on a white plate", "verdict": "exact",
                         "note_vi": "bốn con nghêu trên đĩa"},
                        {"detail": 1, "why": "no apron visible", "verdict": "unrelated",
                         "note_vi": "không thấy tạp dề"}],
            "overall": {"score": 8, "reason": "đúng cảnh xếp nghêu lên đĩa"},
        }
        self.text_verdicts = text_verdicts or {}
        self.fail = fail
        self.images = []
        self.prompts = []
        self.chats = []

    def judge(self, images, prompt, schema, **kwargs):
        if self.fail:
            raise self.fail
        self.images.append(images)
        self.prompts.append(prompt)
        return self.vision_reply

    def chat_json(self, messages, **kwargs):
        self.chats.append(messages)
        asked = "\n".join(message["content"] for message in messages)
        answers = []
        for index, verdict in self.text_verdicts.items():
            shown = _excerpt_ids(asked, index)
            if shown:
                answers.append({"detail": index, "judgements": [
                    {"id": shown[0], "why": "nói rõ", "verdict": verdict}]})
        return {"details": answers}


def _excerpt_ids(prompt, detail):
    """The `e1`-style ids listed under one detail's number in an `_ask` prompt."""
    found, collecting = [], False
    for line in prompt.splitlines():
        if re.match(rf"^{detail}\.\s", line):
            collecting = True
            continue
        if re.match(r"^\d+\.\s", line):
            collecting = False
        if collecting and (match := re.match(r"^\s+(e\d+)\s", line)):
            found.append(match.group(1))
    return found


def run(judge, search=None, query=None, details=DETAILS, window=300.0, target=TARGET):
    search = search or FakeSearch()
    return check_moment(judge, search, EvidenceStore(search),
                        query=query or CheckQuery(visual_text="4 con nghêu trên đĩa trắng", kind="kis"),
                        target=target, details=details, window_seconds=window)


class ImageSetTests(unittest.TestCase):
    def test_the_moment_comes_first_and_the_shots_around_it_follow_labelled_by_offset(self):
        judge = FakeJudge()
        run(judge)

        labels = [label for label, _ in judge.images[0]]
        self.assertEqual(labels[0], "[M] this moment itself")
        self.assertTrue(all(label.startswith("[M] ") for label in labels))
        self.assertTrue(any("earlier in the same story" in label for label in labels))
        self.assertTrue(any("later in the same story" in label for label in labels))

    def test_every_frame_is_fetched_at_the_judging_width(self):
        search = FakeSearch()
        run(FakeJudge(), search)

        self.assertTrue(search.asked)
        self.assertEqual({width for _, _, width in search.asked}, {CHECK_IMAGE_WIDTH})
        self.assertEqual(search.asked[0][1], TARGET.frame_idx)

    def test_the_frame_budget_is_per_side_so_one_call_stays_bounded(self):
        judge = FakeJudge()
        run(judge, FakeSearch(_shots(length=5.0)), window=300.0)

        self.assertLessEqual(len(judge.images[0]), 2 * CHECK_CONTEXT_MAX + 1)

    def test_a_frame_that_cannot_be_cut_is_skipped_not_raised(self):
        class HalfBroken(FakeSearch):
            def image(self, video_id, frame_idx, width=640):
                if frame_idx != TARGET.frame_idx:
                    raise RuntimeError("ffmpeg failed")
                return b"jpeg"

        judge = FakeJudge()
        result = run(judge, HalfBroken())

        self.assertEqual(len(judge.images[0]), 1)
        self.assertEqual(result["frames_seen"], 1)

    def test_no_readable_pixels_at_all_is_the_one_real_failure(self):
        # The person pressed a button that means "look at this": with nothing to look at there
        # is no answer to give, unlike the agent's checklist where vision is the optional extra.
        with self.assertRaises(CheckError):
            run(FakeJudge(), FakeSearch(fail_images=True))

    def test_a_video_with_no_shot_boundaries_falls_back_to_the_keyframe_neighbourhood(self):
        judge = FakeJudge()
        result = run(judge, FakeSearch(shots=[]))

        self.assertGreaterEqual(result["frames_seen"], 1)
        self.assertEqual(judge.images[0][0][0], "[M] this moment itself")


class VerdictTests(unittest.TestCase):
    def test_the_checklist_the_overall_verdict_and_the_weights_come_back_together(self):
        result = run(FakeJudge())

        self.assertEqual(result["verdict"], {"score": 8.0, "reason": "đúng cảnh xếp nghêu lên đĩa"})
        self.assertEqual([check["status"] for check in result["checks"]],
                         ["supported", "not_mentioned"])
        self.assertEqual(result["checks"][0]["source"], "vision")
        self.assertEqual(result["support"]["supported"], 1)
        self.assertEqual(result["support"]["total"], 2)
        self.assertEqual([detail["distinctiveness"] for detail in result["details"]], [0.9, 0.3])

    def test_a_distinctive_detail_is_worth_more_than_a_guessable_one(self):
        # 0.2 + 0.8 * distinctiveness: 0.92 against 0.44, so supporting the distinctive one
        # earns more of the share than supporting the common one would.
        distinctive = run(FakeJudge())["support"]["share"]
        guessable = run(FakeJudge(vision_reply={
            "details": [{"detail": 0, "why": "x", "verdict": "unrelated"},
                        {"detail": 1, "why": "a woman in an apron", "verdict": "exact"}],
            "overall": {"score": 5, "reason": "r"},
        }))["support"]["share"]

        self.assertGreater(distinctive, guessable)

    def test_a_contradiction_is_counted_and_shown_but_lowers_nothing(self):
        result = run(FakeJudge(vision_reply={
            "details": [{"detail": 0, "why": "two clams, not four", "verdict": "conflict",
                         "note_vi": "chỉ hai con"},
                        {"detail": 1, "why": "a woman in an apron", "verdict": "exact"}],
            "overall": {"score": 3, "reason": "sai số lượng"},
        }))
        only_the_apron = run(FakeJudge(vision_reply={
            "details": [{"detail": 0, "why": "nothing like it", "verdict": "unrelated"},
                        {"detail": 1, "why": "a woman in an apron", "verdict": "exact"}],
            "overall": {"score": 3, "reason": "sai"},
        }))

        self.assertEqual(result["support"]["contradicted"], 1)
        self.assertEqual(result["checks"][0]["status"], "contradicted")
        self.assertEqual(result["support"]["share"], only_the_apron["support"]["share"])

    def test_a_judge_whose_own_words_state_a_different_count_is_capped_at_partial(self):
        # Measured on a live p2-30 run: "places 4 X on a white plate" came back supported from
        # a frame showing two. The number is the one part checkable without a model.
        result = run(FakeJudge(vision_reply={
            "details": [{"detail": 0, "why": "two clams on a white plate", "verdict": "exact",
                         "note_vi": "hai con nghêu"}],
            "overall": {"score": 6, "reason": "r"},
        }))

        self.assertEqual(result["checks"][0]["status"], "partial")
        self.assertIn("chi tiết cần 4", result["checks"][0]["note"])

    def test_a_verdict_about_a_detail_that_does_not_exist_is_dropped(self):
        result = run(FakeJudge(vision_reply={
            "details": [{"detail": 7, "why": "x", "verdict": "exact"},
                        {"detail": 0, "why": "four clams", "verdict": "exact"}],
            "overall": {"score": 9, "reason": "r"},
        }))

        self.assertEqual(len(result["checks"]), 2)
        self.assertEqual(result["checks"][0]["status"], "supported")

    def test_a_reply_with_no_usable_overall_still_answers(self):
        result = run(FakeJudge(vision_reply={"details": [], "overall": "nonsense"}))

        self.assertEqual(result["verdict"]["score"], 0.0)
        self.assertEqual([check["status"] for check in result["checks"]],
                         ["not_mentioned", "not_mentioned"])


class WordsTests(unittest.TestCase):
    def test_speech_supports_a_detail_when_the_pixels_are_silent_about_it(self):
        judge = FakeJudge(vision_reply={"details": [], "overall": {"score": 5, "reason": "r"}},
                          text_verdicts={1: "exact"})
        result = run(judge)

        apron = result["checks"][1]
        self.assertEqual(apron["status"], "supported")
        self.assertEqual(apron["source"], "speech")
        self.assertIn("tạp dề", apron["quote"])
        self.assertEqual(apron["at"], 484.0)

    def test_a_sentence_stating_a_different_count_cannot_fully_support_the_detail(self):
        # The transcript says "hai con nghêu" where the detail asks for four. The number is the
        # one part checkable without a model, so the judge saying "exact" is capped at partial.
        judge = FakeJudge(vision_reply={"details": [], "overall": {"score": 5, "reason": "r"}},
                          text_verdicts={0: "exact"})
        clams = run(judge)["checks"][0]

        self.assertEqual(clams["status"], "partial")
        self.assertIn("chi tiết cần 4", clams["note"])

    def test_vision_only_ever_upgrades_what_the_words_already_said(self):
        judge = FakeJudge(
            vision_reply={"details": [{"detail": 1, "why": "not visible", "verdict": "partial"}],
                          "overall": {"score": 5, "reason": "r"}},
            text_verdicts={1: "exact"})
        result = run(judge)

        # The words said "supported"; a weaker vision verdict must not pull it down.
        self.assertEqual(result["checks"][1]["status"], "supported")
        self.assertEqual(result["checks"][1]["source"], "speech")

    def test_the_words_around_the_moment_ride_along_to_the_vision_call(self):
        judge = FakeJudge()
        run(judge)

        self.assertIn("nghêu", judge.prompts[0])
        self.assertIn("MÓN NGON MỖI NGÀY", judge.prompts[0])

    def test_captions_are_not_asked_about_while_the_pixels_are_available(self):
        # A caption is one model's guess about a frame, made offline without the query in mind;
        # the same call judging the pixels has the query in hand. `evidence.summarize` suspends
        # the caption channel for the same reason.
        judge = FakeJudge()
        run(judge)

        asked = "\n".join(message["content"] for chat in judge.chats for message in chat)
        self.assertNotIn("A woman in an apron holds a white plate.", asked)

    def test_a_failed_text_call_costs_the_check_its_words_and_nothing_else(self):
        class Grumpy(FakeJudge):
            def chat_json(self, messages, **kwargs):
                raise RuntimeError("no")

        result = run(Grumpy(text_verdicts={1: "exact"}))
        self.assertEqual(result["checks"][0]["status"], "supported")   # vision still answered
        self.assertEqual(result["checks"][1]["status"], "not_mentioned")


class AnswerTests(unittest.TestCase):
    def test_a_question_gets_its_answer_from_the_same_call(self):
        # The images are already assembled and paid for, so the answer rides along instead of
        # costing the second round trip the agent's own Q&A reading does.
        judge = FakeJudge(vision_reply={
            "details": [{"detail": 0, "why": "four clams", "verdict": "exact"}],
            "overall": {"score": 9, "reason": "r"},
            "answer": {"text": "nghêu", "confidence": 8, "source": "frame", "quote": ""},
        })
        result = run(judge, query=CheckQuery(visual_text="Món ăn được nấu là gì?", kind="qa"))

        self.assertEqual(len(judge.images), 1)
        self.assertEqual(result["answer"], {"text": "nghêu", "confidence": 8.0,
                                            "source": "frame", "quote": ""})
        self.assertIn("The question:", judge.prompts[0])

    def test_a_query_that_is_not_a_question_is_not_asked_one(self):
        judge = FakeJudge()
        result = run(judge)

        self.assertIsNone(result["answer"])
        self.assertNotIn("The question:", judge.prompts[0])

    def test_a_moment_that_does_not_answer_says_so_instead_of_guessing(self):
        judge = FakeJudge(vision_reply={
            "details": [], "overall": {"score": 2, "reason": "r"},
            "answer": {"text": "có lẽ là cá", "confidence": 1, "source": "unknown", "quote": ""},
        })
        result = run(judge, query=CheckQuery(visual_text="Món ăn là gì?", kind="qa"))

        self.assertEqual(result["answer"]["source"], "unknown")
        self.assertEqual(result["answer"]["text"], "")

    def test_an_explicit_question_beats_the_kind(self):
        judge = FakeJudge()
        run(judge, query=CheckQuery(visual_text="cảnh bếp", kind="kis", question="Có mấy con nghêu?"))

        self.assertIn("Có mấy con nghêu?", judge.prompts[0])

    def test_a_qa_answer_is_never_asked_for_from_on_screen_text(self):
        """The Colombia case: the lower-third caption named the COUNTRY and the festival, the
        question asked for the CITY, and a model reading the caption offered "Silletero" and
        "COLOMBIA". The answer comes from the keyframes or the speech, so the prompt says so and
        the schema stops offering "ocr" as a source at all."""
        judge = FakeJudge()
        run(judge, query=CheckQuery(visual_text="cảnh lễ hội", kind="qa", question="Thành phố nào?"))

        self.assertIn("Do NOT take the answer from on-screen text", judge.prompts[0])
        self.assertEqual(CHECK_SCHEMA["properties"]["answer"]["properties"]["source"]["enum"],
                         ["frame", "speech", "unknown"])

    def test_the_answer_is_committed_before_the_prose_when_a_question_is_asked(self):
        # Measured live: one reply wrote the whole answer object INTO `overall.reason` and left
        # `answer` missing, so the person was shown "This moment does not answer the question" for
        # a question the model had answered. The field now comes first in property order and is
        # required, so it is filled before the model is deep inside the overall prose.
        asked, plain = check_schema(True), check_schema(False)
        self.assertEqual(list(asked["properties"])[0], "answer")
        self.assertIn("answer", asked["required"])
        self.assertNotIn("answer", plain["required"])   # a KIS check asks nothing

    def test_a_good_answer_mislabelled_as_ocr_is_kept_not_thrown_away(self):
        # Measured three runs out of three on the Colombia query: the narration said "tại
        # Medellin", the reply carried it as the answer, and it labelled its own source "ocr".
        # The label is the model's report about itself and is not reliable enough to drop text for.
        judge = FakeJudge(vision_reply={
            "details": [], "overall": {"score": 9, "reason": "r"},
            "answer": {"text": "Medellín", "confidence": 9, "source": "ocr", "quote": ""},
        })
        result = run(judge, query=CheckQuery(visual_text="cảnh lễ hội", kind="qa", question="Thành phố nào?"))

        self.assertEqual(result["answer"]["text"], "Medellín")


class SaidTests(unittest.TestCase):
    """`_said` feeds the answer-reading half of the vision call. A dense window (a fast-cut news
    bulletin can pack a dozen unrelated stories into +-5 minutes) can hold far more text than the
    budget -- see the long note on `_said` for the real query this reproduces: the line that
    named the answer sat well past a from-the-start cutoff, so the model never saw it."""

    def test_keeps_the_line_closest_to_the_moment_when_the_window_does_not_fit(self):
        span = {"speech": [{"start": 0.0, "text": "xa mấy phút trước"},
                           {"start": 690.0, "text": "Medellin"}],
               "ocr": []}

        said = _said(span, target_time=692.0, limit=25)

        self.assertIn("Medellin", said)
        self.assertNotIn("xa mấy phút trước", said)

    def test_output_stays_in_time_order_even_though_the_cut_is_by_proximity(self):
        # Also covers the previous bug: sorting the rendered STRINGS put every "on screen" line
        # before every "speech" line (o < s) and read "[speech 100s]" before "[speech 20s]".
        span = {"speech": [{"start": 690.0, "text": "gần sau"}, {"start": 680.0, "text": "gần trước"}],
               "ocr": [{"start": 300.0, "text": "XA"}]}

        said = _said(span, target_time=685.0, limit=1000)

        self.assertLess(said.index("XA"), said.index("gần trước"))
        self.assertLess(said.index("gần trước"), said.index("gần sau"))

    def test_a_window_that_fits_is_unchanged(self):
        span = {"speech": [{"start": 480.0, "text": "nghêu"}], "ocr": [{"start": 470.0, "text": "MÓN NGON"}]}

        said = _said(span, target_time=480.0, limit=2400)

        self.assertIn("nghêu", said)
        self.assertIn("MÓN NGON", said)

    def test_furniture_on_screen_cannot_crowd_the_answer_out_of_the_speech(self):
        """The Colombia case: 169 OCR rows of clocks, logos and "coming up next" banners carrying
        exactly one caption about the story, against the 7 speech lines that held the answer
        ("...tại Medellin."). On one shared budget the furniture took 1598 of 2400 characters and
        the model answered "which city" with "Silletero" -- a word off the caption's festival name.
        Speech keeps its own share, so the answer survives however loud the overlays are.
        """
        span = {
            "speech": [{"start": 690.0, "text": "hoạt động truyền thống tại Medellin"}],
            "ocr": [{"start": 689.0 + index, "text": f"HD 18:41:{index:02d} TIẾP THEO TRÊN Phim hoạt hình"}
                    for index in range(40)],
        }

        said = _said(span, target_time=692.0, limit=2400)

        self.assertIn("Medellin", said)
        self.assertLess(sum(len(line) for line in said.splitlines() if line.startswith("[on screen")),
                        CHECK_SAID_CHARS)   # the on-screen side is capped by the remainder, not the whole pool


class WindowTests(unittest.TestCase):
    def test_the_requested_window_is_reported_back_and_clamped(self):
        self.assertEqual(run(FakeJudge(), window=300.0)["window_seconds"], 300.0)
        self.assertEqual(run(FakeJudge(), window=5.0)["window_seconds"], 30.0)
        self.assertEqual(run(FakeJudge(), window=99999.0)["window_seconds"], 600.0)

    def test_a_narrow_window_shows_the_judge_fewer_shots_than_a_wide_one(self):
        wide, narrow = FakeJudge(), FakeJudge()
        run(wide, window=300.0)
        run(narrow, window=30.0)

        self.assertGreater(len(wide.images[0]), len(narrow.images[0]))


class NoChecklistTests(unittest.TestCase):
    def test_a_query_the_details_call_could_not_split_still_gets_a_verdict(self):
        judge = FakeJudge(vision_reply={"details": [], "overall": {"score": 7, "reason": "hợp"}})
        result = run(judge, details=QueryDetails())

        self.assertEqual(result["verdict"]["score"], 7.0)
        self.assertEqual(result["checks"], [])
        self.assertEqual(result["support"]["share"], 0.0)
        self.assertIn("no detail list", judge.prompts[0])


class ChannelTests(unittest.TestCase):
    """A search can be visual, audio (spoken words to find) or OCR (text to find) -- these mean
    different things to the judge, so they travel as separate labelled channels, never flattened
    into one sentence read as if it were always a scene description.
    """

    def test_the_vision_prompt_labels_each_channel_in_english(self):
        judge = FakeJudge()
        run(judge, query=CheckQuery(visual_text="a person walking", audio_text="cá nhám",
                                    ocr_text="Nà Ní", kind="kis"))

        prompt = judge.prompts[0]
        self.assertIn("Visual description (English): a person walking", prompt)
        self.assertIn("Spoken words to find (Vietnamese speech): cá nhám", prompt)
        self.assertIn("On-screen text to find: Nà Ní", prompt)

    @staticmethod
    def _searched_with(prompt: str) -> str:
        """The actual channel values sent, as opposed to the fixed legend explaining what each
        channel label means (which mentions all three labels regardless of what was searched)."""
        return prompt.split("What the person searched with:", 1)[1].split("\n\n", 1)[0]

    def test_an_empty_channel_is_left_out_of_the_prompt_entirely(self):
        judge = FakeJudge()
        run(judge, query=CheckQuery(visual_text="a person walking", kind="kis"))

        searched = self._searched_with(judge.prompts[0])
        self.assertNotIn("Spoken words to find", searched)
        self.assertNotIn("On-screen text to find", searched)

    def test_an_ocr_only_query_still_produces_a_verdict(self):
        # A bare on-screen string has no "scene" to describe -- the checker must not require a
        # visual channel to exist.
        judge = FakeJudge(vision_reply={"details": [], "overall": {"score": 8, "reason": "khớp chữ"}})
        result = run(judge, query=CheckQuery(ocr_text="Nà Ní", kind="kis"))

        self.assertEqual(result["verdict"]["score"], 8.0)
        searched = self._searched_with(judge.prompts[0])
        self.assertIn("On-screen text to find: Nà Ní", searched)
        self.assertNotIn("Visual description", searched)

    def test_the_details_call_gets_vietnamese_labels_matching_check_details_note(self):
        # `query.description` (Vietnamese labels) is what `query_details`/`CHECK_DETAILS_SYSTEM`
        # read -- `DETAILS_SYSTEM` is a Vietnamese prompt, unlike the English vision prompt.
        query = CheckQuery(ocr_text="Nà Ní", kind="kis")
        self.assertEqual(query.description, "Chữ trên màn hình cần tìm: Nà Ní")

        query = CheckQuery(visual_text="a person walking", audio_text="cá nhám", kind="kis")
        self.assertEqual(query.description,
                         "Mô tả hình ảnh (tiếng Anh): a person walking\nLời thoại cần tìm: cá nhám")

    def test_a_qa_question_falls_back_to_whatever_channels_are_given(self):
        # No explicit `question`: for a Q&A query, the search channels themselves are read as
        # the question (matches the pre-channel behaviour where the query text doubled as one).
        query = CheckQuery(audio_text="có bao nhiêu con nghêu", kind="qa")
        self.assertEqual(query.asked, "Lời thoại cần tìm: có bao nhiêu con nghêu")

    def test_an_explicit_question_beats_every_channel(self):
        query = CheckQuery(visual_text="cảnh bếp", kind="qa", question="Có mấy con nghêu?")
        self.assertEqual(query.asked, "Có mấy con nghêu?")

    def test_a_kis_query_is_never_asked_as_a_question(self):
        query = CheckQuery(visual_text="cảnh bếp", kind="kis")
        self.assertIsNone(query.asked)


if __name__ == "__main__":
    unittest.main()
