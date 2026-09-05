# Temporal beam and candidate-dedup benchmark

Audio was unavailable; all variants use visual fallback.

| Variant | MRR | Recall@10 | p95 ms | Per-query rank |
|---|---:|---:|---:|---|
| baseline | 0.3810 | 0.5714 | 758.7 | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-2 | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-4 | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-8 | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-16 | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| dedup-1s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| dedup-2s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| dedup-3s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-2+dedup-1s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-2+dedup-2s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-2+dedup-3s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-4+dedup-1s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-4+dedup-2s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-4+dedup-3s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-8+dedup-1s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-8+dedup-2s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-8+dedup-3s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-16+dedup-1s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-16+dedup-2s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |
| beam-16+dedup-3s | 0.3810 | 0.5714 | — | query-p1-4:—, query-p1-16:—, query-p1-18:3, query-p2-10:1, query-p2-22:—, query-p3-1:3, query-p3-24:1 |

Chosen: `baseline`.
