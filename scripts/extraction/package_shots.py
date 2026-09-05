"""Zip TransNetV2 shot JSON into per-batch archives for Kaggle upload.

Matches the batch split already used for the video zips in
`scripts.extraction.build_kaggle_manifest` — batch1 = L21..L30 (2026), batch2 = K01..K20
(2025) — but as a single zip each, since shot JSON is small enough that it never needed
the K01-07/08-14/15-20 three-way split the (much heavier) video zips require.

    uv run python -m scripts.extraction.package_shots --out dist/kaggle-shots

Each JSON lands in the archive at `shots/<group>/<video_id>.json`, so unzipping
reproduces the `data/<year>/shots/` layout `shots.py` reads directly.
"""
import argparse
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))

BATCHES = (
    {"name": "batch1_2026_shots", "year": "2026", "prefix": "L"},
    {"name": "batch2_2025_shots", "year": "2025", "prefix": "K"},
)


def log(message):
    print(message, flush=True)


def package_batch(batch, data_dir, out_dir):
    shots_dir = data_dir / batch["year"] / "shots"
    sources = sorted(shots_dir.glob(f"{batch['prefix']}*/*.json"))
    if not sources:
        log(f"skip {batch['name']}: no shot JSON under {shots_dir}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{batch['name']}.zip"
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sources:
            arcname = f"shots/{path.parent.name}/{path.name}"
            archive.write(path, arcname)

    uncompressed = sum(path.stat().st_size for path in sources)
    compressed = target.stat().st_size
    log(f"{target}: {len(sources)} files, {uncompressed / 1e6:.1f} MB -> {compressed / 1e6:.1f} MB")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "kaggle-shots")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    written = [package_batch(batch, args.data_dir, args.out) for batch in BATCHES]
    if not any(written):
        raise SystemExit(f"no shot JSON found under {args.data_dir}/<year>/shots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
