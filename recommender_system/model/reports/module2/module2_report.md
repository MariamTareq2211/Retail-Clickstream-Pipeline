# Module 2 - Profiles, model selection and tuning

Generated: 2026-10-29T00:42:18.516897+00:00
Configuration fingerprint: `e7e9ed3c238fc12d33a2de98ff71600287bd7299d2be72479b4d9a425df2091d`

## Benchmark protocol

- Users sampled: 15,000
- Users scored: 12,449
- Validation windows: 3
- Selection cut-off: K = 10
- Seen items excluded from candidates and labels: True

## Model comparison (mean over validation windows)

| model | Recall@K mean | Recall@K std | HitRate@K mean | Recall overall | HitRate overall |
|---|---:|---:|---:|---:|---:|
| Model 3 - Time-Decayed Weighted Centroid | 0.0276 | 0.0024 | 0.0535 | 0.0283 | 0.0590 |
| Model 2 - Interaction-Weighted Centroid | 0.0260 | 0.0019 | 0.0511 | 0.0268 | 0.0565 |
| Model 1 - Unweighted Centroid | 0.0235 | 0.0010 | 0.0456 | 0.0244 | 0.0504 |

**Winner:** Model 3 - Time-Decayed Weighted Centroid

Selection rule: max mean Recall@K, then mean HitRate@K, then min Recall std.

## Winner by validation window

| window | users | HitRate@K | Recall@K | Precision@K | MRR@K |
|---:|---:|---:|---:|---:|---:|
| 0 | 4,536 | 0.0571 | 0.0298 | 0.0075 | 0.0227 |
| 1 | 4,971 | 0.0565 | 0.0288 | 0.0072 | 0.0215 |
| 2 | 4,961 | 0.0468 | 0.0242 | 0.0059 | 0.0185 |

## Optuna tuning

- Study: `module2_centroid` (12 completed trials)
- Best trial: 5
- Best recall@10: 0.031335
- Best parameters: `{'max_features': 4000, 'ngram_range': [1, 2], 'min_df': 3, 'price_weight': 0.050872, 'decay_rate_per_hour': 0.0825648, 'count_transform': 'raw', 'event_weights': {'1': 0.218561, '2': 1.76313, '4': 1.0}}`

## Final profile matrix

- Users: 709,231
- Features: 4,001
- Mean items per user: 4.78
- Mean profile non-zeros per user: 8.88
- Empty profiles: 0
- Catalogue matrix non-zeros: 471,170

## Decay

- λ = 0.082565 per hour
- Half-life: 8.40 hours
- Weight after 24 h: 0.1379

## Timings

- load: 3.14s
- benchmark: 321.91s
- tune: 434.82s
- rebuild: 11.69s
