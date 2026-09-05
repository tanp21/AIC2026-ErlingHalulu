#!/usr/bin/env python3
"""Benchmark index types: latency + recall@k vs Flat (ground truth)."""
import gc
import time

import numpy as np

from backend import core

QUERIES = [
    "a red car driving on the street",
    "a group of people in a meeting",
    "aerial view of a city at night",
    "a dog playing in the park",
    "food on a table",
    "a person riding a bicycle",
    "traffic on a highway",
    "a football match",
    "a cooking show in a kitchen",
    "an anchor reading the news",
    "a beach with waves",
    "a concert stage with lights",
    "a map or chart being shown",
    "a person giving an interview",
    "a shop with many products",
]


def bench(k=50):
    Q = np.vstack([core.encode_text(q) for q in QUERIES]).astype("float32")

    flat = core.get_index("flat")
    t0 = time.perf_counter()
    _, If = flat.search(Q, k)
    flat_ms = (time.perf_counter() - t0) * 1000 / len(QUERIES)
    print(f"\n{len(QUERIES)} queries, k={k}\n")
    print(f"{'index':<18}{'latency_ms':>12}{'recall@k':>10}")
    print(f"{'flat (exact)':<18}{flat_ms:>12.1f}{1.0:>10.3f}")
    del flat
    gc.collect()

    for name, params in (("ivf", (16, 64, 128)), ("hnsw", (32, 128, 256))):
        idx = core.get_index(name)
        for p in params:
            if name == "ivf":
                idx.nprobe = p
            else:
                idx.hnsw.efSearch = p
            t0 = time.perf_counter()
            _, I = idx.search(Q, k)
            ms = (time.perf_counter() - t0) * 1000 / len(QUERIES)
            recall = np.mean([len(set(I[i].tolist()) & set(If[i].tolist())) / k
                              for i in range(len(QUERIES))])
            label = f"{name}(nprobe={p})" if name == "ivf" else f"{name}(ef={p})"
            print(f"{label:<18}{ms:>12.1f}{recall:>10.3f}")
        del idx
        gc.collect()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-k", type=int, default=50)
    args = p.parse_args()
    bench(args.k)
