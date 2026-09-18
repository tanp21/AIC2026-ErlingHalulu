import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent.config import AgentSettings, read_dotenv
from agent.tasks import GoldRow, load_answers, load_tasks, task_from_text
from agent.vi_numbers import spell_integer, spell_numbers

ROOT = Path(__file__).resolve().parents[2]


class SpellNumbersTests(unittest.TestCase):
    def test_spells_integers_the_way_the_asr_writes_them(self):
        cases = {
            0: "không", 5: "năm", 10: "mười", 15: "mười lăm", 21: "hai mươi mốt",
            24: "hai mươi tư", 105: "một trăm linh năm", 211: "hai trăm mười một",
            1500: "một ngàn năm trăm", 2024: "hai ngàn không trăm hai mươi tư",
            3_000_000: "ba triệu",
        }
        for value, words in cases.items():
            with self.subTest(value=value):
                self.assertEqual(spell_integer(value), words)

    def test_spells_numbers_inside_text(self):
        self.assertEqual(spell_numbers("cá nhám nặng 211kg"), "cá nhám nặng hai trăm mười một kg")
        self.assertEqual(spell_numbers("tỷ lệ tử vong 88%"), "tỷ lệ tử vong tám mươi tám phần trăm")
        self.assertEqual(spell_numbers("biển báo 2.15"), "biển báo hai phẩy mười lăm")
        self.assertEqual(spell_numbers("giá 1.000 đồng"), "giá một ngàn đồng")
        self.assertEqual(spell_numbers("không có số"), "không có số")


class TaskTests(unittest.TestCase):
    def test_kind_and_trake_event_count_come_from_the_query(self):
        task = task_from_text("query-p1-16-trake", "Intro.\nE1: a\nE2 b\nE3: c")
        self.assertEqual((task.kind, task.event_count), ("trake", 3))
        scenes = task_from_text("query-p2-21-trake", "Cảnh 1: a\nCảnh 2: b\nCảnh 3: c\nCảnh 4: d")
        self.assertEqual(scenes.event_count, 4)
        self.assertEqual(task_from_text("query-p1-3-qa", "x").kind, "qa")
        self.assertIsNone(task_from_text("query-p1-1-kis", "E1: a").event_count)

    def test_loads_zip_in_natural_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for name in ("query-p1-10-kis.txt", "query-p1-2-qa.txt", "query-p1-1-kis.txt"):
                    archive.writestr(name, "﻿mô tả")
            tasks = load_tasks(path)
        self.assertEqual([t.query_id for t in tasks], ["query-p1-1-kis", "query-p1-2-qa", "query-p1-10-kis"])
        self.assertEqual(tasks[0].text, "mô tả")

    def test_loads_both_answer_key_formats(self):
        single = load_answers(ROOT / "queries" / "round1-2025.csv")
        self.assertEqual(single["query-p1-1-kis"][0].frames, (25724,))
        self.assertEqual(single["query-p1-4-trake"][0].frames, (4724, 5122, 5435, 5871))
        self.assertEqual(single["query-p1-15-qa"][0].answer, "Giang Ly")

        zipped = load_answers(ROOT / "queries" / "Round12026_answer.zip")
        self.assertEqual([row.frames[0] for row in zipped["query-p1-2-kis"]], [3123, 3225])
        self.assertEqual(zipped["query-p1-17-qa"][0].answer, "Đèo Tà Pứa")
        self.assertEqual(zipped["query-p1-16-trake"][0].frames, (9784, 10132, 10191))

    def test_loads_a_round_from_the_team_visual_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.json"
            path.write_text(json.dumps({"cases": [
                {"id": "Round 2 - 2026/query-p2-3-kis:1", "targets": [["L01_V001", 5], ["L01_V009", 7]]},
                {"id": "Round 2 - 2026/query-p2-8-trake:2", "targets": [["L01_V002", 90]]},
                {"id": "Round 2 - 2026/query-p2-8-trake:1", "targets": [["L01_V007", 1], ["L01_V002", 10]]},
                {"id": "Round 1 - 2026/query-p1-1-kis:1", "targets": [["L01_V003", 1]]},
            ]}), encoding="utf-8")
            answers = load_answers(f"{path}::Round 2 - 2026")
        self.assertEqual(sorted(answers), ["query-p2-3-kis", "query-p2-8-trake"])
        self.assertEqual(answers["query-p2-3-kis"], [GoldRow("L01_V001", (5,)), GoldRow("L01_V009", (7,))])
        # events in order, from the one video that carries all of them
        self.assertEqual(answers["query-p2-8-trake"], [GoldRow("L01_V002", (10, 90))])


class SettingsTests(unittest.TestCase):
    def test_environment_overrides_dotenv_and_keys_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text('# c\nDEEPSEEK_API_KEY="from-file"\nGEMINI_API_KEY=g\nbroken\n', encoding="utf-8")
            self.assertEqual(read_dotenv(dotenv)["DEEPSEEK_API_KEY"], "from-file")
            settings = AgentSettings.from_env({"DEEPSEEK_API_KEY": "env", "AIC_AGENT_URL": "http://x/"}, dotenv)
            self.assertEqual((settings.deepseek_api_key, settings.gemini_api_key), ("env", "g"))
            self.assertEqual(settings.server_url, "http://x")
            self.assertEqual(settings.deepseek_model, "deepseek-flash")
            with self.assertRaises(RuntimeError):
                AgentSettings.from_env({}, Path(directory) / "missing.env")


if __name__ == "__main__":
    unittest.main()
