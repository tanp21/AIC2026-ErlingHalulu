import unittest

from agent.plan import BATCH_LIMIT, initial_queries, parse_plan, transcript_text
from agent.pool import CandidatePool, Verdict
from agent.tasks import task_from_text

KIS = task_from_text("query-p1-1-kis", "mô tả")
TRAKE = task_from_text("query-p1-4-trake", "E1: a\nE2: b\nE3: c")


def plan_data(**overrides):
    data = {
        "summary_vi": "tóm tắt",
        "scenes": [
            {"description_vi": "cảnh đầu", "visual_en": ["a map with dams", "map", "dam map"]},
            {"description_vi": "cảnh sau", "visual_en": ["aerial view of a dam", "dam in rain"]},
        ],
        "target_scene": 1,
        "transcript_vi": ["con đập 211 mét", "thủy lợi"],
        "ocr": ["Nà Ní"],
        "lookups": ["what is X?"],
        "question_vi": "ignored for kis",
        "answer_source": "speech",
    }
    data.update(overrides)
    return data


class ParsePlanTests(unittest.TestCase):
    def test_coerces_and_caps_untrusted_fields(self):
        plan = parse_plan(plan_data(
            transcript_vi="một cụm", ocr=[1, "", "  Nà  Ní ", "NÀ NÍ", "kg"], target_scene=9,
        ), KIS)
        self.assertEqual(plan.transcript_vi, ["một cụm"])
        self.assertEqual(plan.ocr, ["Nà Ní"])
        self.assertEqual(plan.target_scene, 1)
        self.assertIsNone(plan.question_vi)
        self.assertIsNone(plan.answer_source)

    def test_trake_events_stand_in_for_missing_scenes(self):
        plan = parse_plan({"scenes": [], "events_en": ["e1", "e2", "e3"]}, TRAKE)
        self.assertEqual([scene.visual_en for scene in plan.scenes], [["e1"], ["e2"], ["e3"]])

    def test_rejects_a_plan_without_any_visual_scene(self):
        with self.assertRaises(ValueError):
            parse_plan({"scenes": [{"visual_en": []}]}, KIS)
        with self.assertRaises(ValueError):
            parse_plan(["not", "an", "object"], KIS)


class InitialQueriesTests(unittest.TestCase):
    def test_target_scene_first_then_temporal_speech_text_and_other_scenes(self):
        queries = initial_queries(parse_plan(plan_data(), KIS), KIS)
        ids = [query["id"] for query in queries]
        self.assertEqual(ids[:2], ["r1.s1", "r1.t"])
        self.assertEqual((queries[0]["text"], queries[0]["paraphrases"]), ("aerial view of a dam", ["dam in rain"]))
        self.assertEqual(queries[1]["events"], ["a map with dams", "aerial view of a dam"])
        self.assertLess(ids.index("r1.o0"), ids.index("r1.s0"))
        self.assertEqual(queries[-1]["paraphrases"], ["map", "dam map"])
        spoken = [q for q in queries if q["kind"] == "audio"]
        self.assertEqual(spoken[0]["text"], "con đập 211 mét")
        self.assertEqual([q["method"] for q in spoken], ["keyword", "semantic", "keyword"])
        self.assertLessEqual(len(queries), BATCH_LIMIT)
        self.assertEqual(len(ids), len(set(ids)))

    def test_trake_uses_events_and_attaches_speech_to_temporal(self):
        plan = parse_plan(plan_data(scenes=[{"visual_en": ["kitchen"]}],
                                    events_en=["e1", "e2", "e3"]), TRAKE)
        temporal = next(q for q in initial_queries(plan, TRAKE) if q["kind"] == "temporal")
        self.assertEqual(temporal["events"], ["e1", "e2", "e3"])
        self.assertEqual(temporal["audio_text"], "con đập 211 mét")

    def test_one_query_per_scene_carrying_at_most_four_paraphrases(self):
        scenes = [{"visual_en": [f"s{i}v{j}" for j in range(6)]} for i in range(4)]
        queries = initial_queries(parse_plan(plan_data(scenes=scenes, target_scene=0), KIS), KIS)
        visual = [q for q in queries if q["kind"] == "visual"]
        self.assertEqual([q["id"] for q in visual], ["r1.s0", "r1.s1", "r1.s2", "r1.s3"])
        self.assertEqual(visual[0]["paraphrases"], ["s0v1", "s0v2", "s0v3", "s0v4"])
        self.assertLessEqual(len(queries), BATCH_LIMIT)
        single = parse_plan(plan_data(scenes=[{"visual_en": ["only caption"]}], target_scene=0), KIS)
        self.assertNotIn("paraphrases", initial_queries(single, KIS)[0])

    def test_transcript_numbers_stay_digits_unless_the_index_spells_them(self):
        self.assertEqual(transcript_text("năm 2024"), "năm 2024")
        self.assertEqual(transcript_text("năm 2024", spoken_numbers=True),
                         "năm hai ngàn không trăm hai mươi tư")
        queries = initial_queries(parse_plan(plan_data(), KIS), KIS, spoken_numbers=True)
        self.assertEqual(next(q for q in queries if q["kind"] == "audio")["text"],
                         "con đập hai trăm mười một mét")


def hit(video, frame, time, rank, **extra):
    return {"video_id": video, "frame_idx": frame, "time": time, "rank": rank, "score": 1.0, **extra}


class CandidatePoolTests(unittest.TestCase):
    def test_merges_nearby_keyframes_and_attaches_spans(self):
        pool = CandidatePool()
        created = pool.add_results({
            "v0": {"kind": "visual", "hits": [hit("L01_V001", 100, 10.0, 1), hit("L01_V002", 5, 1.0, 2)]},
            "v1": {"kind": "visual", "hits": [hit("L01_V001", 110, 12.0, 1)]},
            "a0": {"kind": "audio", "hits": [
                hit("L01_V001", 90, 20.0, 1, start=5.0, end=35.0, text="lời thoại"),
                hit("L01_V009", 7, 3.0, 2, start=0.0, end=6.0, text="khác"),
            ]},
        })
        self.assertEqual(len(created), 3)
        first = pool.get("C1")
        self.assertEqual((first.video_id, first.frame_idx), ("L01_V001", 100))
        self.assertEqual(set(first.evidence), {"v0", "v1", "a0"})
        self.assertEqual(pool.ordered()[0].cid, "C1")

    def test_exact_ocr_counts_more_than_fuzzy(self):
        pool = CandidatePool()
        pool.add_results({"o": {"kind": "ocr", "hits": [
            hit("L01_V001", 10, 1.0, 1, start=0.0, end=2.0, exact=True),
            hit("L01_V002", 10, 1.0, 1, start=0.0, end=2.0, exact=False),
        ]}})
        kinds = {c.video_id: next(iter(c.evidence.values())).kind for c in pool.candidates.values()}
        self.assertEqual(kinds, {"L01_V001": "ocr_exact", "L01_V002": "ocr"})
        self.assertEqual(pool.ordered()[0].video_id, "L01_V001")

    def test_verdicts_reorder_candidates(self):
        pool = CandidatePool()
        pool.add_results({"v": {"kind": "visual", "hits": [
            hit("L01_V001", 1, 0.0, 1), hit("L01_V002", 1, 0.0, 2), hit("L01_V003", 1, 0.0, 3),
        ]}})
        pool.get("C1").verdict = Verdict(1, "sai màu áo")
        pool.get("C3").verdict = Verdict(8, "khớp")
        self.assertEqual([c.cid for c in pool.ordered()], ["C3", "C2", "C1"])
        self.assertEqual([c.cid for c in pool.unverified(5)], ["C2"])

    def test_temporal_paths_become_candidates_and_sequences(self):
        pool = CandidatePool()
        pool.add_results({"t": {"kind": "temporal", "hits": [{
            "rank": 1, "video_id": "L01_V001", "score": 0.9,
            "sequences": [
                {"score": 0.9, "events": [{"frame_idx": 10, "time": 1.0, "score": 0.5},
                                          {"frame_idx": 90, "time": 9.0, "score": 0.4}]},
                {"score": 0.1, "events": [{"frame_idx": 500, "time": 50.0, "score": 0.1},
                                          {"frame_idx": 600, "time": 60.0, "score": 0.1}]},
            ],
        }]}})
        self.assertEqual(len(pool.sequences), 1)
        sequence = pool.sequences[0]
        self.assertEqual((sequence.frames, sequence.candidate_ids), ([10, 90], ["C1", "C2"]))
        self.assertEqual(set(pool.get("C2").evidence), {"t.e2"})


if __name__ == "__main__":
    unittest.main()
