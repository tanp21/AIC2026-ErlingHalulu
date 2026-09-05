# Temporal search benchmark

| K | Scoring | MRR | Recall@10 | p95 ms |
|---:|---|---:|---:|---:|
| 500 | minmax_mean | 0.2857 | 0.2857 | 457.5 |
| 500 | percentile_mean | 0.2143 | 0.2857 | 460.9 |
| 500 | raw_mean | 0.2857 | 0.2857 | 488.3 |
| 500 | raw_min | 0.2143 | 0.2857 | 449.9 |
| 500 | shifted_harmonic | 0.2857 | 0.2857 | 445.6 |
| 1000 | minmax_mean | 0.3143 | 0.4286 | 469.9 |
| 1000 | percentile_mean | 0.2619 | 0.4286 | 461.0 |
| 1000 | raw_mean | 0.3333 | 0.4286 | 483.5 |
| 1000 | raw_min | 0.2273 | 0.2857 | 503.7 |
| 1000 | shifted_harmonic | 0.3214 | 0.4286 | 497.5 |
| 2000 | minmax_mean | 0.3262 | 0.4286 | 691.2 |
| 2000 | percentile_mean | 0.2643 | 0.4286 | 664.3 |
| 2000 | raw_mean | 0.3810 | 0.5714 | 758.7 |
| 2000 | raw_min | 0.2276 | 0.2857 | 693.9 |
| 2000 | shifted_harmonic | 0.3690 | 0.5714 | 704.5 |
| 5000 | minmax_mean | 0.3344 | 0.4286 | 865.0 |
| 5000 | percentile_mean | 0.2643 | 0.4286 | 925.6 |
| 5000 | raw_mean | 0.3810 | 0.5714 | 852.6 |
| 5000 | raw_min | 0.2276 | 0.2857 | 879.3 |
| 5000 | shifted_harmonic | 0.3690 | 0.5714 | 909.1 |

Chosen: K=2000, `raw_mean`.
