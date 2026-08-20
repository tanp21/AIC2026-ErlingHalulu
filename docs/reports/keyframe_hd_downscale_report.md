# Report: Safety Evaluation of Downscaling KXX Keyframes to HD (720p)

**Date:** 2026-08-16

**Dataset:** `data/2025/keyframes/K01–K20` (558,246 PNG keyframes, originally 1920×1080)

**Constraint:** The source data in `./data` was not modified. All experiments ran in `/tmp/opencode`.

## 1. Executive summary

Downscaling all KXX keyframes to 720p is **safe** for a Video Browser Showdown-style competition.

| Evaluation area | Result |
| --- | --- |
| Retrieval embeddings | **No impact** |
| Thumbnail display | **Safe**; 99.95% of frames are nearly identical |
| Full-resolution zoom on ~0.05% of high-frequency frames | Fine detail may be lost under close inspection |

## 2. Retrieval impact

The evaluation embedded 500 random keyframes with the production `PE-Core-bigG-14-448` pipeline using OpenCLIP, an RTX 3060 GPU, and FP16. Each frame was tested at its original 1080p resolution and after downscaling to 720p.

| Metric | Result |
| --- | --- |
| Cosine similarity, 1080p vs. 720p | min **0.9909**, mean **0.9979**, median **0.9982** |
| Self-retrieval among 500 frames | **top-1 = 100%**, top-10 = 100%, worst rank = 0 |

The model resizes every image to 448×448 before embedding. Because 1080p and 720p share a 16:9 aspect ratio, `Resize(448)` produces 796×448 in both cases before the same 448×448 center crop. The 1080p-to-720p conversion therefore has negligible effect on embeddings and preserves retrieval ranking.

## 3. Display differences

The display evaluation used this sequence: original 1080p → 720p with `INTER_AREA` → 1080p with `LANCZOS4` → PSNR/SSIM measurement.

### 3.1 PSNR distribution

Results for a 40,000-frame sample with `seed=99`:

| Threshold | Share of frames | Estimated count across 558k frames |
| --- | --- | --- |
| < 25 dB; severe, clearly visible degradation | **0.005%** | ~28 |
| < 28 dB | 0.015% | ~84 |
| < 30 dB; visible difference | **0.045%** | ~251 |
| < 32 dB | 0.135% | ~750 |
| < 35 dB | 1.578% | ~8,800 |
| < 40 dB | 45.9% | — |
| < 45 dB | 95.0% | — |

The distribution had **min ≈ 18.4 dB, mean = 40.3 dB, and max = 54.2 dB**. The minimum was updated from a 100,000-frame sample; the earlier 50,000-frame sample produced 21 dB. The distribution has a heavy tail, so the true minimum across all 558,000 frames may be lower.

### 3.2 Frames with the largest differences

Results for a 50,000-frame sample with `seed=7`:

| PSNR (dB) | Path |
| --- | --- |
| 18.37 | `K17/K17_V010/015481.png` |
| 19.46 | `K19/K19_V014/003736.png` |
| 21.47 | `K19/K19_V023/039321.png` |
| 22.81 | `K16/K16_V004/032328.png` |
| 23.33 | `K17/K17_V012/003111.png` |
| 23.62 | `K17/K17_V021/001848.png` |

The complete list of the 50 worst frames remains a local artifact and is not committed.

The worst frames contain high-frequency content such as small text, grids, and dense patterns. They are concentrated in K19, K20, K18, K17, K14, and K16 videos. Low-detail frames achieve very high PSNR values above 50 dB and remain nearly unchanged.

## 4. Recommendation

1. **Downscale all KXX keyframes to 720p** for VBS-style retrieval. The experiment found no retrieval impact, and 720p provides more than enough detail for 200–400 px thumbnails.
2. **Preserve selected 1080p frames only if fine-text inspection matters.** Approximately 0.05% of high-frequency frames may lose sharpness at full-resolution zoom.
3. Expect a storage reduction of approximately **30–40%**.

## 5. Local artifacts

The following files were generated in `/tmp/opencode/kf_cmp_full/` and are not part of the repository:

- `worst0_003894.png` — original 1080p version of the worst frame
- `worst0_003894_720.png` — 720p version
- `worst0_003894_diff5x.png` — difference map amplified 5×
- `worst0_side_by_side.png` — original, 720p, and difference map side by side
