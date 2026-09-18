import json
import re
import unittest

from agent.llm import GroundedAnswer, LLMError
from agent.orchestrator import Agent, AgentOptions
from agent.tasks import task_from_text
from agent.vision import ANSWER_SCHEMA, PICK_SCHEMA, SEQUENCE_SCHEMA

PLAN = {
    "summary_vi": "tìm cảnh",
    "scenes": [{"description_vi": "thuyền", "visual_en": ["a red boat", "boat"]}],
    "transcript_vi": ["thuyền đỏ"],
    "ocr": [],
    "lookups": [],
}


class FakeDeepSeek:
    def __init__(self, plan=PLAN, turns=()):
        self.plan = plan
        self.turns = list(turns)
        self.chats = []
        self.tool_names = []

    def chat_json(self, messages, **_):
        if isinstance(self.plan, Exception):
            raise self.plan
        return self.plan

    def chat(self, messages, tools=None, **_):
        self.chats.append([dict(m) for m in messages])
        self.tool_names.append([tool["function"]["name"] for tool in tools or []])
        if not self.turns:
            return {"role": "assistant", "content": "done"}
        return self.turns.pop(0)


def tool_call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call-{name}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}},
    ]}


class FakeGemini:
    """Scores candidates from `scores` keyed by (video, frame); everything else 2."""

    def __init__(self, scores=None, answers=None, pick="F2", reading="42", event_scores=None):
        self.scores = scores or {}
        self.event_scores = event_scores or {}
        self.reading = reading
        self.answers = answers or {}
        self.pick = pick
        self.judged = []
        self.answer_calls = 0

    def judge(self, images, prompt, schema, **_):
        if schema is PICK_SCHEMA:
            return {"id": self.pick, "confidence": 8, "reason": "bắt đầu"}
        if schema is SEQUENCE_SCHEMA:
            video = re.match(r"\[E1\] (\S+) ", images[0][0]).group(1)
            score = self.event_scores.get(video, 5)
            return {"events": [{"id": f"E{i + 1}", "score": score} for i in range(len(images))]}
        if schema is ANSWER_SCHEMA:
            self.answer_calls += 1
            self.answer_prompt = prompt
            return {"answer": self.reading, "confidence": 7, "evidence": "trên bảng"}
        # the follow-up question asked via `inspect` in the neighbours test
        follow_up = "Chữ trên biển" in prompt
        verdicts = []
        for label, _ in images:
            cid, video = re.match(r"\[(C\d+)\] (\S+) ", label).group(1, 2)
            frame = self.frame_of[cid]
            self.judged.append(cid)
            verdicts.append({"id": cid, "score": 1 if follow_up else self.scores.get((video, frame), 2),
                             "reason": "ok", "answer": self.answers.get((video, frame), "")})
        return {"candidates": verdicts}

    def web_search(self, question):
        return GroundedAnswer("cá mập", (), ())


class FakeSearch:
    def __init__(self, visual_hits, temporal=None):
        self.visual_hits = visual_hits
        self.temporal = temporal or []
        self.batches = []

    def search(self, queries):
        self.batches.append(queries)
        results = {}
        for query in queries:
            if query["kind"] == "visual":
                hits = self.visual_hits
                if query.get("video_ids"):
                    hits = [h for h in hits if h[0] in query["video_ids"]]
                results[query["id"]] = {"kind": "visual", "hits": [
                    {"rank": rank, "video_id": v, "frame_idx": f, "time": t, "score": 0.3}
                    for rank, (v, f, t) in enumerate(hits, 1)]}
            elif query["kind"] == "temporal":
                results[query["id"]] = {"kind": "temporal", "hits": self.temporal}
            else:
                results[query["id"]] = {"kind": query["kind"], "error": "unavailable"}
        return results

    def image(self, video_id, frame_idx, width=640):
        return b"jpeg"

    def neighbours(self, video_id, frame_idx, radius=6):
        return [{"video_id": video_id, "frame_idx": frame_idx + step * 10, "time": (frame_idx + step * 10) / 10}
                for step in range(-2, 3)]

    def transcript(self, video_id):
        return [{"start": 0.0, "end": 5.0, "text": "xin chào"}]


def make_agent(deepseek, gemini, search, **options):
    agent = Agent(deepseek, gemini, search, AgentOptions(**options))
    # the fake gemini needs each candidate's frame to look up its scripted score
    original = agent.inspector.verify

    def verify(task, plan, candidates, question=None):
        gemini.frame_of = {c.cid: c.frame_idx for c in candidates}
        return original(task, plan, candidates, question)

    agent.inspector.verify = verify
    return agent


HITS = [("L01_V001", 100, 10.0), ("L01_V002", 200, 20.0), ("L01_V003", 300, 30.0)]


class AgentRunTests(unittest.TestCase):
    def test_confident_round_one_skips_the_tool_loop(self):
        deepseek = FakeDeepSeek()
        gemini = FakeGemini({("L01_V002", 200): 9})
        search = FakeSearch(HITS)
        result = make_agent(deepseek, gemini, search).run(task_from_text("q-kis", "thuyền đỏ"))

        self.assertIsNone(result.error)
        self.assertEqual(deepseek.chats, [])
        self.assertEqual((result.ranking[0].video_id, result.ranking[0].frame_idx), ("L01_V002", 200))
        self.assertEqual([entry["tool"] for entry in result.trace], ["plan", "search", "verify"])
        self.assertEqual({q["kind"] for q in search.batches[0]}, {"visual", "audio"})

    def test_a_frame_the_server_cannot_cut_stays_unverified_alone(self):
        search = FakeSearch(HITS)
        original = search.image

        def image(video_id, frame_idx, width=640):
            if video_id == "L01_V001":
                raise RuntimeError("HTTP 500 ffmpeg failed")
            return original(video_id, frame_idx, width)

        search.image = image
        gemini = FakeGemini({("L01_V002", 200): 9})
        result = make_agent(FakeDeepSeek(), gemini, search).run(task_from_text("q-kis", "x"))

        verdicts = {c.video_id: c.verdict for c in result.ranking}
        self.assertIsNone(verdicts["L01_V001"])
        self.assertEqual(verdicts["L01_V002"].score, 9)

    def test_weak_round_one_hands_over_to_deepseek_tools(self):
        deepseek = FakeDeepSeek(turns=[
            tool_call("search", queries=[
                {"kind": "visual", "text": "boat interior", "video_ids": ["L01_V003", "bad id"]},
                {"kind": "visual", "text": "a red boat", "paraphrases": ["boat"]},  # round 1 ran it
                {"kind": "nonsense"},
            ]),
            tool_call("finish", ranking=["C3", "C999"], notes="chắc chắn"),
        ])
        gemini = FakeGemini({("L01_V003", 300): 8})
        search = FakeSearch(HITS)
        result = make_agent(deepseek, gemini, search, early_stop_score=9).run(
            task_from_text("q-kis", "thuyền đỏ"))

        scoped = search.batches[1]
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0]["video_ids"], ["L01_V003"])
        self.assertEqual(result.ranking[0].cid, "C3")
        self.assertEqual(result.notes, "chắc chắn")
        self.assertIn("Rejected invalid queries", deepseek.chats[1][-1]["content"])
        # the second DeepSeek turn sees the first turn's assistant message unchanged
        self.assertEqual(deepseek.chats[1][2]["tool_calls"][0]["function"]["name"], "search")

    def test_neighbours_and_follow_up_questions_keep_the_match_score(self):
        deepseek = FakeDeepSeek(turns=[
            tool_call("neighbours", candidate_id="C1", radius=2),
            tool_call("inspect", candidate_ids=["C1"], question="Chữ trên biển là gì?"),
            tool_call("finish", ranking=["C1"]),
        ])
        gemini = FakeGemini({("L01_V001", 100): 8}, answers={("L01_V001", 100): "SẮC CỔ"})
        search = FakeSearch(HITS)
        result = make_agent(deepseek, gemini, search).run(task_from_text("q-kis", "biển chữ"))

        self.assertIsNone(result.error)
        top = result.ranking[0]
        self.assertEqual((top.cid, top.verdict.score, top.verdict.answer), ("C1", 8, "SẮC CỔ"))
        neighbour_step = next(entry for entry in result.trace if entry["tool"] == "neighbours")
        self.assertEqual(neighbour_step["summary"], "Neighbour candidates: C4, C5, C1, C6, C7")
        # neighbours at +-10 and +-20 frames joined the pool as their own candidates
        self.assertEqual(
            {c.frame_idx for c in result.ranking if c.video_id == "L01_V001"}, {80, 90, 100, 110, 120})

    def test_qa_answer_prefers_a_dedicated_reading_over_verdict_answers(self):
        task = task_from_text("q-qa", "Con số trên cân là bao nhiêu?")
        plan = {**PLAN, "question_vi": "Con số trên cân?", "answer_source": "on_screen_text",
                "lookups": ["Cân này là của chợ nào?"]}
        gemini = FakeGemini({("L01_V001", 100): 9}, answers={("L01_V001", 100): "can màu xanh"})
        result = make_agent(FakeDeepSeek(plan), gemini, FakeSearch(HITS)).run(task)
        self.assertEqual((result.answer, gemini.answer_calls), ("42", 1))
        self.assertIn("cá mập", gemini.answer_prompt)  # web lookups reach the reading as hints

        gemini = FakeGemini({("L01_V001", 100): 9}, answers={("L01_V001", 100): "5"}, reading="")
        result = make_agent(FakeDeepSeek(plan), gemini, FakeSearch(HITS)).run(task)
        self.assertEqual(result.answer, "5")
        self.assertIn("unconfirmed", result.answer_evidence)

    def test_spent_budget_gets_one_finish_only_turn(self):
        task = task_from_text("q-qa", "Tên của con đèo là gì?")
        plan = {**PLAN, "question_vi": "Tên con đèo?", "answer_source": "speech"}
        deepseek = FakeDeepSeek(plan, turns=[
            tool_call("web_search", question="đèo nào sạt lở?"),
            tool_call("finish", ranking=["C2"], answer="Đèo Tà Pứa"),
        ])
        gemini = FakeGemini({("L01_V002", 200): 7})
        result = make_agent(deepseek, gemini, FakeSearch(HITS), max_steps=1).run(task)

        self.assertEqual(deepseek.tool_names[-1], ["finish"])
        self.assertEqual((result.answer, gemini.answer_calls), ("Đèo Tà Pứa", 0))
        self.assertEqual(result.ranking[0].cid, "C2")
        self.assertEqual([e["tool"] for e in result.trace].count("web_search"), 1)

    def test_trake_refines_each_event_forward_in_time(self):
        task = task_from_text("q-trake", "E1: a\nE2: b")
        plan = {**PLAN, "events_en": ["start", "end"]}
        temporal = [{"rank": 1, "video_id": "L01_V005", "score": 0.8, "sequences": [
            {"score": 0.8, "events": [{"frame_idx": 100, "time": 10.0, "score": 0.5},
                                      {"frame_idx": 400, "time": 40.0, "score": 0.5}]}]}]
        result = make_agent(FakeDeepSeek(plan), FakeGemini(pick="F2"),
                            FakeSearch(HITS, temporal), max_steps=0).run(task)

        self.assertEqual(len(result.sequences), 1)
        sequence = result.sequences[0]
        # "F2" is the centre of the 5 fake neighbours -> the original frame
        self.assertEqual(sequence.frames, [100, 400])
        self.assertTrue(sequence.refined)
        times = sequence.times
        self.assertEqual(times, sorted(times))

    def test_trake_paths_are_reordered_by_per_event_checks(self):
        task = task_from_text("q-trake", "E1: a\nE2: b")
        plan = {**PLAN, "events_en": ["start", "end"]}

        def path(rank, video, first):
            return {"rank": rank, "video_id": video, "score": 1 / rank, "sequences": [{
                "score": 1 / rank, "events": [{"frame_idx": first, "time": first / 10, "score": 0.5},
                                              {"frame_idx": first + 300, "time": first / 10 + 30, "score": 0.5}]}]}

        temporal = [path(1, "L01_V005", 100), path(2, "L01_V006", 100), path(3, "L01_V007", 100)]
        gemini = FakeGemini(event_scores={"L01_V005": 2, "L01_V006": 9})
        result = make_agent(FakeDeepSeek(plan), gemini, FakeSearch(HITS, temporal),
                            max_steps=0, trake_check=2, trake_refine=1).run(task)

        self.assertEqual([s.video_id for s in result.sequences], ["L01_V006", "L01_V005", "L01_V007"])
        self.assertEqual([s.event_scores for s in result.sequences], [[9, 9], [2, 2], None])
        self.assertEqual([s.refined for s in result.sequences], [True, False, False])

    def test_planning_failure_is_reported_not_raised(self):
        with self.assertLogs("agent", "ERROR"):
            result = make_agent(FakeDeepSeek(LLMError("down")), FakeGemini(), FakeSearch(HITS)).run(
                task_from_text("q-kis", "x"))
        self.assertIn("planning failed", result.error)
        self.assertEqual(result.ranking, [])


if __name__ == "__main__":
    unittest.main()
