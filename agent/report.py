"""A static HTML page for the human check before submitting.

Thumbnails load straight from the search server, so the browser asks for the
team's Basic Auth once; nothing is copied locally.
"""
from __future__ import annotations

from html import escape

SHOWN = 20

STYLE = """
:root { --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e3e2dc;
        --good:#1f7a3a; --mid:#9a6a00; --bad:#a3322a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#ecebe6; --muted:#a09f98; --card:#21211f; --line:#34332f;
          --good:#6cc98a; --mid:#e0b04a; --bad:#ef8a80; }
}
body { margin:0; padding:16px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, sans-serif; }
section { max-width:1200px; margin:0 auto 32px; }
h2 { font-size:17px; margin:0 0 4px; }
.query { white-space:pre-wrap; color:var(--muted); margin:0 0 8px; }
.meta { font-size:13px; margin:0 0 10px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(200px, 1fr)); gap:10px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.card img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:var(--line); }
.card div { padding:6px 8px; font-size:12px; }
.score { font-weight:600; }
.good { color:var(--good); } .mid { color:var(--mid); } .bad { color:var(--bad); }
.answer { font-size:15px; font-weight:600; }
"""


def _verdict_class(verdict):
    if verdict is None:
        return ""
    return "good" if verdict["score"] >= 6 else "mid" if verdict["score"] > 3 else "bad"


def render(results: list[dict], server_url: str) -> str:
    sections = []
    for result in results:
        cards = []
        for rank, candidate in enumerate(result.get("ranking", [])[:SHOWN], 1):
            verdict = candidate.get("verdict")
            label = (f'<span class="score {_verdict_class(verdict)}">{verdict["score"]:.0f}/10</span> '
                     f'{escape(verdict["reason"])}' if verdict else "chưa kiểm tra")
            thumb = f"{server_url}/thumb/{candidate['video_id']}/{candidate['frame_idx']}?w=320"
            cards.append(
                f'<div class="card"><a href="{server_url}/frame/{candidate["video_id"]}/'
                f'{candidate["frame_idx"]}" target="_blank"><img loading="lazy" src="{escape(thumb)}" '
                f'alt=""></a><div>#{rank} {candidate["video_id"]} · {candidate["frame_idx"]} · '
                f'{candidate["time"]:.1f}s<br>{label}</div></div>'
            )
        extra = []
        if result.get("answer") is not None:
            extra.append(f'<p class="answer">Đáp án: {escape(result["answer"])}</p>')
        for sequence in result.get("sequences", [])[:3]:
            extra.append(f'<p class="meta">TRAKE {sequence["video_id"]}: {sequence["frames"]}'
                         f'{" (đã tinh chỉnh)" if sequence.get("refined") else ""}</p>')
        if result.get("notes"):
            extra.append(f'<p class="meta">Ghi chú agent: {escape(result["notes"])}</p>')
        if result.get("error"):
            extra.append(f'<p class="meta bad">Lỗi: {escape(result["error"])}</p>')
        sections.append(
            f'<section><h2>{escape(result["query_id"])}</h2>'
            f'<p class="query">{escape(result["query"])}</p>'
            f'<p class="meta">{len(result.get("trace", []))} bước · {result.get("elapsed", 0)}s</p>'
            f'{"".join(extra)}<div class="grid">{"".join(cards)}</div></section>'
        )
    return (
        '<!doctype html><html lang="vi"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Agent results</title><style>{STYLE}</style></head>"
        f"<body>{''.join(sections)}</body></html>"
    )
