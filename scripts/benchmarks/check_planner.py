"""Planner regressions: the cases a prompt rule was written for, checked against fresh plans.

Every rule in `agent.prompts.ANALYZE_SYSTEM` came from a query the planner once got wrong. This
replans those queries and checks the property each fix was about, so a rewrite of the prompt
(shorter, more general, fewer examples) can be shown to keep them. Plans vary between calls, so
each case runs `--runs` times and reports how many passed.

    uv run python -m scripts.benchmarks.check_planner --runs 3
    uv run python -m scripts.benchmarks.check_planner --thinking   # plan with DeepSeek thinking
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import re
import time

from agent.config import AgentSettings
from agent.llm import DeepSeekClient
from agent.plan import analysis_messages, initial_queries, parse_plan
from agent.prompts import ANALYZE_SYSTEM
from agent.tasks import task_from_text


def captions(plan):
    return [text for scene in plan.scenes for text in scene.visual_en] + list(plan.readings_en)


def spoken(plan):
    return " ".join(plan.transcript_vi).lower()


def no_relationship(plan):
    """"hai mẹ con": a relationship no picture shows is the bare count, never a guessed gender or age."""
    first = " ".join(scene.visual_en[0] for scene in plan.scenes).lower()
    return not re.search(r"\b(mother|son|daughter|child|kid|parent|family|young man|older woman|old woman)\b", first)


def scale_display(plan):
    """A question about the number on a scale keeps the scale's readable display in the caption."""
    text = " ".join(captions(plan) + plan.caption_en).lower()
    return "scale" in text and bool(re.search(r"display|screen|reading|number|digit", text))


def unnamed_x(plan):
    """"Hỏi X là con gì": the unknown is described, never named in visual or transcript queries."""
    text = " ".join(captions(plan) + plan.caption_en).lower() + " " + spoken(plan)
    guessed = re.search(r"\b(clam|crab|shrimp|prawn|oyster|snail|mussel|nghêu|cua|tôm|sò|ốc|hến)\b", text)
    return not guessed and bool(plan.lookups)


def one_slide(plan):
    """Items listed as on one slide are one lecture scene, each also searched alone."""
    return (len(plan.scenes) == 1 and "lecture" in plan.scenes[0].visual_en[0].lower()
            and len(plan.readings_en) >= 1)


def no_prompt_words(plan):
    """Transcript phrases come from the query, not from the instructions' own examples.

    An earlier prompt illustrated lecture topics with "khác biệt" and "làm việc nhóm", and the
    planner searched exactly those words. A phrase that also appears in the prompt counts as copied.
    """
    prompt = ANALYZE_SYSTEM.lower()
    return not any(phrase.lower() in prompt for phrase in plan.transcript_vi)


def cooking(plan):
    """A cooking query plans a narrated clip: dish or ingredient words to hear."""
    return plan.topic == "cooking" and bool(plan.transcript_vi)


def trake_events(plan):
    return len(plan.events_en) == 3


CASES = [
    ("hai mẹ con", "kis", "Đoạn phim bắt đầu với cảnh hai mẹ con cầm điện thoại trò chuyện, sau đó chuyển "
     "sang cảnh người con trả lời phỏng vấn, phía sau có một phụ nữ da đen đang bước qua.", [no_relationship]),
    ("cân cá (p1-3)", "qa", "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loại bị "
     "một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?", [scale_display]),
    ("4 con X (p2-30)", "qa", "Một cô gái đeo tạp dề màu trắng, bên cạnh là một lọ hoa riềng tía. Sau đó cô gái "
     "này đặt lên 1 dĩa trắng 4 con X, được biết X là nguyên liệu cho món ăn trong tập này. Sau đó người này lại "
     "cầm lên 2 con X. Sau đó người này đối thoại với một người đối diện để xem hôm nay nấu món gì. Hỏi X là con gì?",
     [unnamed_x]),
    ("slide kéo co", "kis", "Đây là một đoạn trong bài giảng. Trên slide bao gồm:\n- Một nhóm nhân vật người 3D "
     "màu trắng vây quanh một nhân vật màu đỏ ở chính giữa.\n- Hai nhân vật hoạt hình nam đang trong tư thế thi "
     "đấu kéo co, đối đầu nhau với sợi dây thừng.", [one_slide, no_prompt_words]),
    ("bún gà (p1-18)", "kis", "Cảnh quay cho thấy hành động trình bày món ăn sau khi hoàn thành giai đoạn chế "
     "biến. Bún được cho đầu tiên vào một chén rỗng, sau đó đầu bếp lần lượt cho từng vá nước dùng cùng các "
     "nguyên liệu như thịt gà, cà rốt, sả, nấm mèo vào chén bún, và kết thúc bằng việc thả một cọng ngò lên "
     "trên cùng.", [cooking]),
    ("múa lân (p1-16)", "trake", "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh lá cờ "
     "trắng viền đỏ.\nE1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.\nE2 Khoảnh khắc "
     "đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.\nE3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng "
     "múa lân.", [trake_events]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--thinking", action="store_true", help="plan with DeepSeek thinking on")
    parser.add_argument("--verbose", action="store_true", help="print each plan's queries")
    args = parser.parse_args()
    settings = AgentSettings.from_env()
    client = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url)

    def one(case_run):
        (name, kind, text, checks), run = case_run
        task = task_from_text(name, text, kind)
        started = time.monotonic()
        plan = parse_plan(client.chat_json(analysis_messages(task), thinking=args.thinking), task)
        return name, run, plan, task, [check(plan) for check in checks], time.monotonic() - started

    jobs = [(case, run) for case in CASES for run in range(args.runs)]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(one, jobs))
    passed = total = 0
    for name, run, plan, task, outcomes, seconds in results:
        passed += all(outcomes)
        total += 1
        marks = " ".join(f"{check.__name__}={'ok' if ok else 'FAIL'}" for check, ok in
                         zip(next(c[3] for c in CASES if c[0] == name), outcomes))
        print(f"{name:18} run {run + 1}  {seconds:5.1f}s  {marks}")
        if args.verbose or not all(outcomes):
            for query in initial_queries(plan, task):
                print(f"    {query['id']:8} {query['kind']:8} {(query.get('text') or str(query.get('events')))[:140]}")
            if plan.lookups:
                print(f"    lookups: {plan.lookups}")
    print(f"\n{passed}/{total} plans passed every check")


if __name__ == "__main__":
    main()
