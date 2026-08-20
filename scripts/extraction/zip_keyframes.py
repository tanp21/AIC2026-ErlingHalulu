import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def zip_group(job):
    group, keyframes, out = job
    arc = out / f"{group.name}.zip"
    n = 0
    with zipfile.ZipFile(arc, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(group.rglob("*.png")):
            z.write(p, p.relative_to(keyframes))
            n += 1
    return group.name, n, arc.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes", type=Path, default=Path("data/2026/keyframes"))
    parser.add_argument("--out", type=Path, default=Path("data/2026/keyframes_zips"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    groups = sorted(path for path in args.keyframes.iterdir() if path.is_dir())
    if not groups:
        raise SystemExit(f"no keyframe groups found under {args.keyframes}")
    with ProcessPoolExecutor(max_workers=len(groups)) as ex:
        jobs = ((group, args.keyframes, args.out) for group in groups)
        for name, n, size in ex.map(zip_group, jobs):
            print(f"{name}.zip: {n} files, {size/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
