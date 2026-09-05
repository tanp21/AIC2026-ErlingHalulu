#!/usr/bin/env python3
"""Score OCR configurations against benchmarks/ocr/ocr_ground_truth.json.

The acceptance gate is the three hand-labelled frames the OCR work was
commissioned around: a flood gauge with 11px digits, a two-column embossed
couplet, and a perspective-tilted recipe card. A configuration that cannot read
those is not a candidate no matter how fast it is, which is why
``scripts.ocr.scoring.rank_configs`` sorts on the gate before anything else.

    uv run python -m scripts.benchmarks.bench_ocr --only-acceptance
    uv run python -m scripts.benchmarks.bench_ocr --sweep --write-config

Needs a GPU and the OCR extras, so it is a manual tool, not a unit test; the
scoring functions it calls are covered by tests/test_ocr_benchmark.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

from scripts.ocr.engines import OcrConfig, OcrPipeline
from scripts.ocr.scoring import ConfigReport, rank_configs, score_frame

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))
GROUND_TRUTH = ROOT / "benchmarks" / "ocr" / "ocr_ground_truth.json"
CONFIG_OUT = ROOT / "benchmarks" / "ocr" / "ocr_config.json"
REPORT_OUT = ROOT / "benchmarks" / "ocr" / "ocr_benchmark.json"


def load_frames(path: Path, only_acceptance: bool) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data["frames"]
    if only_acceptance:
        frames = [f for f in frames if f.get("acceptance")]
    return frames


def candidate_configs(sweep: bool) -> dict[str, object]:
    """The bake-off grid.

    ``baseline`` is what the tuning converged on. The rest exist to keep the
    choice honest: each isolates one decision, so re-running the sweep shows
    whether that decision still pays for itself.

    Values are either an :class:`OcrConfig` (routed through the tuned
    ``OcrPipeline``) or a bare ``(image, ignore_regions=None) -> str``
    callable from :mod:`scripts.ocr.alt_engines` -- engines outside the tuned
    pipeline added for the 2026-09 shot-audit bake-off. ``evaluate`` branches
    on which one it got; ``score_frame`` doesn't care either way.
    """
    baseline = OcrConfig()
    if not sweep:
        return {"baseline": baseline}
    grid = {
        # v4 detection + small-text rescan + CLAHE + VietOCR seq2seq.
        "baseline": baseline,
        # Does the small-text rescan earn its ~3x cost?
        "no-rescan": replace(baseline, small_rescan=False),
        # Is CLAHE earning its measured +1 on the pillar?
        "no-clahe": replace(baseline, rec_clahe_clip=0.0),
        # PP-OCRv5 detectors: newer, and they read the gauge digit unaided, but
        # their extra boxes cost reading order.
        "v5-mobile-det": replace(baseline, det_model="v5-mobile"),
        "v5-server-det": replace(baseline, det_model="v5-server"),
        "v6-medium-det": replace(baseline, det_model="v6-medium"),
        # The 2026-09 shot audit's finding: v5-mobile-det generalises better than
        # v4 on the holdout set, and Vintern's reader role recovers most of what
        # it still misses (dense/small-text lines, stylised logos) -- without
        # postcorrect's Qwen2.5-VL-7B tax. Not yet compared against
        # postcorrect+reader on the full sweep at the time this was added.
        "v5-mobile-det+reader": replace(baseline, det_model="v5-mobile", reader=True),
        # Is the slower transformer recogniser worth 3x the time?
        "transformer-rec": replace(baseline, rec_model="vgg_transformer"),
        # Does extra tone-mark headroom in the crop help or hurt?
        "padded-crop": replace(baseline, rect_pad_y=0.10),
        # Is the confidence floor discarding real text?
        "no-conf-floor": replace(baseline, rec_min_conf=0.0),
        # Vision-LLM post-correction on top of the winner. Far too slow for a
        # whole-corpus sweep, so it is a quality ceiling reference and a
        # candidate for a low-confidence shortlist, not a default.
        "postcorrect": replace(baseline, postcorrect=True),
        # Post-correction plus an independent Vietnamese reader over cropped
        # blocks, merged additively. The only configuration that recovers text
        # the recogniser cannot read at all.
        "postcorrect+reader": replace(baseline, postcorrect=True, reader=True),
    }
    from scripts.ocr.alt_engines import ENGINES

    grid.update(ENGINES)
    return grid


def evaluate(config, frames: list[dict], *, verbose: bool) -> ConfigReport:
    read = OcrPipeline(config).read if isinstance(config, OcrConfig) else config
    report = ConfigReport(config_id="")
    for entry in frames:
        path = DATA / entry["path"]
        if not path.exists():
            raise SystemExit(f"missing eval frame: {path}")
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not read eval frame: {path}")
        start = time.perf_counter()
        text = read(image, ignore_regions=entry.get("ignore_regions"))
        elapsed = time.perf_counter() - start
        score = score_frame(entry, text, elapsed)
        report.scores.append(score)
        if verbose:
            status = "PASS" if score.passed else "FAIL"
            print(f"    [{status}] {entry['id']:16s} "
                  f"required {score.required_found}/{score.required_total} "
                  f"phrases {score.phrase_found}/{score.phrase_total} "
                  f"cer {score.cer:.3f} wr {score.word_recall:.3f} {elapsed:.2f}s")
            if score.required_missing:
                print(f"           missing: {list(score.required_missing)}")
            if score.phrase_missing:
                print(f"           phrases: {list(score.phrase_missing)}")
            print(f"           text: {text[:300]}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=GROUND_TRUTH)
    parser.add_argument("--only-acceptance", action="store_true",
                        help="score only the three hand-labelled acceptance frames")
    parser.add_argument("--sweep", action="store_true",
                        help="evaluate the full candidate grid instead of the baseline")
    parser.add_argument("--configs", default=None,
                        help="comma-separated subset of the sweep grid, e.g. "
                             "'baseline,postcorrect' (implies --sweep)")
    parser.add_argument("--min-fps", type=float, default=5.0)
    parser.add_argument("--device", default=None, help="override rec_device, e.g. cpu")
    parser.add_argument("--write-config", action="store_true",
                        help="write the winner to benchmarks/ocr/ocr_config.json")
    parser.add_argument("--postcorrect", action="store_true",
                        help="enable vision-LLM post-correction on the baseline")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    frames = load_frames(args.eval_set, args.only_acceptance)
    print(f"[bench-ocr] {len(frames)} frames from {args.eval_set}")

    grid = candidate_configs(args.sweep or bool(args.configs))
    if args.configs:
        wanted = [name.strip() for name in args.configs.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in grid]
        if unknown:
            raise SystemExit(f"unknown config(s) {unknown}; grid is {sorted(grid)}")
        grid = {name: grid[name] for name in wanted}

    reports = []
    for name, config in grid.items():
        if isinstance(config, OcrConfig):
            if args.device:
                config = replace(config, rec_device=args.device)
            if args.postcorrect:
                config = replace(config, postcorrect=True)
        print(f"\n[bench-ocr] === {name} ===")
        report = evaluate(config, frames, verbose=not args.quiet)
        report.config_id = name
        # Only an OcrConfig serialises into ocr_config.json; a bare engine
        # adapter from scripts.ocr.alt_engines has nothing to write.
        report.config = config if isinstance(config, OcrConfig) else None
        reports.append(report)
        summary = report.summary()
        print(f"    -> gate={'PASS' if summary['acceptance_passed'] else 'FAIL'} "
              f"required={summary['required_recall']:.3f} folded={summary['folded_recall']:.3f} "
              f"phrase={summary['phrase_recall']:.3f} "
              f"cer={summary['cer']:.3f} fps={summary['frames_per_second']:.2f}")

    ranked = rank_configs(reports, min_fps=args.min_fps)
    header = (f"\n{'rank':<5}{'config':<20}{'gate':<7}{'req':<8}{'folded':<8}"
              f"{'phrase':<9}{'cer':<8}{'fps':<7}")
    print(header)
    for position, report in enumerate(ranked, 1):
        s = report.summary()
        print(f"{position:<5}{report.config_id:<20}"
              f"{'PASS' if s['acceptance_passed'] else 'FAIL':<7}"
              f"{s['required_recall']:<8.3f}{s['folded_recall']:<8.3f}"
              f"{s['phrase_recall']:<9.3f}"
              f"{s['cer']:<8.3f}{s['frames_per_second']:<7.2f}")

    REPORT_OUT.write_text(
        json.dumps([r.summary() for r in ranked], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[bench-ocr] report -> {REPORT_OUT}")

    winner = ranked[0]
    if args.write_config:
        if not winner.acceptance_passed:
            raise SystemExit(
                f"refusing to write config: {winner.config_id} fails the acceptance gate "
                f"({winner.summary()['failures']})"
            )
        if winner.config is None:
            raise SystemExit(
                f"refusing to write config: {winner.config_id!r} is a bare engine adapter "
                "(scripts.ocr.alt_engines), not an OcrConfig, so it cannot be serialised "
                "to ocr_config.json"
            )
        CONFIG_OUT.write_text(
            json.dumps(
                {"config_id": winner.config_id,
                 "fingerprint": winner.config.fingerprint(),
                 "summary": winner.summary(),
                 "config": asdict(winner.config)},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[bench-ocr] winner {winner.config_id!r} -> {CONFIG_OUT}")

    raise SystemExit(0 if winner.acceptance_passed else 1)


if __name__ == "__main__":
    main()
