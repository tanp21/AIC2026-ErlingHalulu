#!/usr/bin/env python3
"""Render the pilot captions as a side-by-side Markdown report for manual review.

Nothing here scores a caption. The production config is chosen by reading the output,
so the job of this script is to put every variant of one shot next to every other and
next to the v1 caption, with the full configuration spelled out beside each.

    python -m scripts.captioning.pilot_report \
        artifacts/captions/pilot-v2/*/captions.parquet -o artifacts/captions/pilot-v2/report.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import re
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts" / "captions" / "1024-default-8frames" / "captions.parquet"
# Broadcaster furniture the prompt asks the model to omit; used only to measure
# compliance here, never to edit a caption.
BRANDING = re.compile(r"HTV|Tu[oô]?i\s*Tre?|TUOI\s*TRE|60\s*Gi[aâ]y|VTV|THVL", re.I)
CONFIG_FIELDS = (
    ("resolution", lambda r: f"{r['frame_width']}px wide" if r["frame_width"] else "native"),
    ("frames used", lambda r: r["frames_used"]),
    ("KV dtype", lambda r: r["kv_cache_dtype"]),
    ("max_tokens", lambda r: r["max_tokens"]),
    ("prompt / completion tokens", lambda r: f"{r['prompt_tokens']} / {r['output_tokens']}"),
    ("finish reason", lambda r: r["finish_reason"]),
    ("elapsed", lambda r: f"{r['elapsed_sec']:.1f} s"),
)


def load_rows(paths):
    rows = []
    for path in paths:
        table = pq.read_table(path)
        label = Path(path).parent.name
        for row in table.to_pylist():
            rows.append({**row, "config": label})
    return rows


def load_baseline():
    if not BASELINE.exists():
        return {}
    return {
        (row["video_id"], row["shot_index"]): row
        for row in pq.read_table(BASELINE).to_pylist()
    }


def render_config_table(row, generation):
    lines = ["| field | value |", "|---|---|"]
    for name, accessor in CONFIG_FIELDS:
        lines.append(f"| {name} | {accessor(row)} |")
    lines.append(f"| generation | {generation} |")
    return "\n".join(lines)


def render(rows, generation, baseline, include_raw=True, title=None):
    by_unit = defaultdict(list)
    for row in rows:
        by_unit[(row["video_id"], row["shot_index"], row["window_index"])].append(row)

    out = [
        title or "# CapRL-Video-4B captions",
        "",
        "Prompt and 8-frame sampling are the \"1024-default-8frames\" recipe; generation",
        "parameters differ per configuration and are printed beside each caption, and",
        "`max_tokens` is 2048 rather than the baseline's 1024 so captions are not cut",
        "off. One section per shot window; a shot split by the 20 s / 2 s-overlap",
        "windowing gets one section per window.",
        "",
    ]
    if not include_raw:
        out += [
            "Raw model output is omitted here — the caption above is the model text",
            "with only bracketed timestamps stripped. The full report (`../report.md`)",
            "and `../pilot.json` carry it verbatim.",
            "",
        ]
    if len(by_unit) > 3:
        out += ["## Windows", ""]
        for key in sorted(by_unit):
            head = by_unit[key][0]
            out.append(
                f"- [window {key[2] + 1} of {head['window_count']}]"
                f"(#{anchor(head)}) — "
                f"{head['window_start_sec']:.2f}–{head['window_end_sec']:.2f} s"
            )
        out.append("")
    for key in sorted(by_unit):
        video_id, shot_index, window_index = key
        # Ascending resolution, so the columns read small -> large; `native` is stored
        # as width 0 but is the widest, so it sorts last.
        variants = sorted(by_unit[key], key=lambda r: (
            r["frame_width"] or 1 << 20, r["kv_cache_dtype"], r["max_tokens"]))
        head = variants[0]
        out += [
            "---",
            "",
            f"## {video_id} shot {shot_index} "
            f"(window {window_index + 1}/{head['window_count']})",
            "",
            f"Shot {head['start_sec']:.2f}-{head['end_sec']:.2f} s "
            f"({head['duration_sec']:.2f} s). "
            f"Window {head['window_start_sec']:.2f}-{head['window_end_sec']:.2f} s "
            f"({head['window_duration_sec']:.2f} s). "
            f"Frames {head['start_frame']}-{head['end_frame']}.",
            "",
        ]
        for row in variants:
            out += [
                f"### {row['config']}",
                "",
                render_config_table(row, generation),
                "",
            ]
            # The baseline prompt asks for one continuous narrative, on-screen text
            # read inline as it goes -- there is no separate Summary/On-screen-text
            # section to print unless a row actually carries one (an older run that
            # did ask for them).
            if row["summary"]:
                out += [f"**Summary.** {row['summary']}", ""]
            out += [strip_lead(row["caption"], row["summary"]), ""]
            if row["onscreen_text"]:
                out += [f"**On-screen text (separately read).** {row['onscreen_text']}", ""]
            if include_raw:
                out += [
                    "<details><summary>raw output</summary>",
                    "",
                    "```",
                    row["raw_output"],
                    "```",
                    "",
                    "</details>",
                    "",
                ]
        previous = baseline.get((video_id, shot_index))
        if previous and window_index == 0:
            out += [
                "### v1 baseline (1024-default-8frames, transformers, 8 frames)",
                "",
                previous["caption"],
                "",
            ]
    return "\n".join(out) + "\n"


def render_summary(rows, rates):
    """One config now (the "1024-default-8frames" recipe): throughput, truncation,
    branding compliance, and the per-shot index into the split files."""
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)

    out = [
        "# CapRL-Video-4B pilot — summary",
        "",
        f"{len({(r['video_id'], r['shot_index']) for r in rows})} shots "
        f"(manual-review targets plus hard cases from the three 2026 rounds), expanded to "
        f"{len({(r['video_id'], r['shot_index'], r['window_index']) for r in rows})} caption "
        "units by the 20 s / 2 s-overlap windowing.",
        "Prompt and 8-frame sampling follow the \"1024-default-8frames\" quality",
        "baseline; generation parameters (temperature / top_p / top_k / repetition",
        "penalty) and the seed are printed in each configuration's captions.",
        "Measured on one RTX 5090, vLLM 0.29.0.",
        "",
        "## Throughput", "",
        "| config | mean prompt tok | mean output tok | hit the cap | mean s/unit | units/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(by_config):
        rows_c = by_config[name]
        truncated = sum(1 for r in rows_c if r["finish_reason"] != "stop")
        out.append(
            f"| {name} "
            f"| {round(sum(r['prompt_tokens'] for r in rows_c) / len(rows_c)):,} "
            f"| {round(sum(r['output_tokens'] for r in rows_c) / len(rows_c)):,} "
            f"| {truncated}/{len(rows_c)} "
            f"| {sum(r['elapsed_sec'] for r in rows_c) / len(rows_c):.1f} "
            f"| {rates.get(name, float('nan')):.2f} |"
        )

    out += [
        "", "## Prompt compliance: broadcast branding", "",
        "The baseline prompt asks the model not to name the broadcaster or its overlays.",
        "It is not reliably obeyed -- checked against the raw caption text, since this",
        "recipe has no separate on-screen-text field.",
        "",
        "| config | shots with branding leaked into the caption |",
        "|---|---:|",
    ]
    for name in sorted(by_config):
        rows_c = by_config[name]
        out.append(
            f"| {name} "
            f"| {sum(1 for r in rows_c if BRANDING.search(r['caption'] or ''))}/{len(rows_c)} |"
        )
    out += [
        "",
        "`HTV`/`HTV9` accounts for nearly all of it. A deterministic parse-time filter",
        "over the known broadcaster strings (the same approach `caption_shots_gemini.py`",
        "already takes with its `BROADCAST_OVERLAY` regex) can clean this up from the",
        "existing parquet with no further GPU time, once the corpus is captioned.",
        "",
        "## Shots", "",
        "| shot | duration | windows | file |", "|---|---:|---:|---|",
    ]
    seen = {}
    for row in rows:
        key = (row["video_id"], row["shot_index"])
        seen.setdefault(key, row)
    for (video_id, shot_index), row in sorted(seen.items()):
        out.append(
            f"| `{video_id}` #{shot_index} | {row['duration_sec']:.2f} s "
            f"| {row['window_count']} | [{shot_slug(video_id, shot_index)}.md]"
            f"(by-shot/{shot_slug(video_id, shot_index)}.md) |"
        )
    return "\n".join(out) + "\n"


def shot_slug(video_id, shot_index):
    return f"{video_id}_shot{shot_index:04d}"


def anchor(row):
    """GitHub-style heading anchor for the per-window section."""
    text = (f"{row['video_id']} shot {row['shot_index']} "
            f"(window {row['window_index'] + 1}/{row['window_count']})")
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s]+", "-", slug.strip())


def strip_lead(description, summary):
    """`caption` is summary + description joined; drop the echo so columns differ."""
    if summary and description.startswith(summary):
        return description[len(summary):].strip()
    return description


def write_split(rows, generation, baseline, directory):
    """One file per shot: the full report is 3.9 MB, which no editor enjoys.

    Raw output is dropped here because it duplicates the caption; the long lecture
    shots have 20-37 windows and would otherwise run past a megabyte each.
    """
    by_shot = defaultdict(list)
    for row in rows:
        by_shot[(row["video_id"], row["shot_index"])].append(row)
    directory.mkdir(parents=True, exist_ok=True)
    for (video_id, shot_index), shot_rows in by_shot.items():
        path = directory / f"{shot_slug(video_id, shot_index)}.md"
        path.write_text(render(
            shot_rows, generation, baseline, include_raw=False,
            title=f"# {video_id} shot {shot_index}",
        ))
    return len(by_shot)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=None,
                        help="also write one Markdown file per shot")
    parser.add_argument("--summary", type=Path, default=None,
                        help="also write the config comparison table")
    parser.add_argument("--rates", default="",
                        help="label=units_per_second pairs, comma separated")
    parser.add_argument("--generation", default="temperature 0.7, top_p 0.8, "
                                                "top_k 20, seed 42")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_rows(args.parquet)
    baseline = load_baseline()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, args.generation, baseline))
    units = {(r["video_id"], r["shot_index"], r["window_index"]) for r in rows}
    configs = sorted({r["config"] for r in rows})
    print(f"{len(rows)} captions over {len(units)} units, configs {configs} -> {args.output}")

    if args.split_dir:
        count = write_split(rows, args.generation, baseline, args.split_dir)
        print(f"{count} per-shot files -> {args.split_dir}")
    if args.summary:
        rates = {}
        for pair in args.rates.split(","):
            if "=" in pair:
                label, value = pair.split("=", 1)
                rates[label.strip()] = float(value)
        args.summary.write_text(render_summary(rows, rates))
        print(f"summary -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
