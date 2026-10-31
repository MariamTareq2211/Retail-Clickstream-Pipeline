# Module 3 - Ranking and hidden-test evaluation

Generated: 2026-10-29T00:53:37.475882+00:00
Configuration fingerprint: `e7e9ed3c238fc12d33a2de98ff71600287bd7299d2be72479b4d9a425df2091d`
Selected model: **Model 3 - Time-Decayed Weighted Centroid**
Module 2 fingerprint: `e7e9ed3c238fc12d33a2de98ff71600287bd7299d2be72479b4d9a425df2091d`

## Evaluation protocol

- Evaluation split: `test`
- Events: 1,237,224
- Users in split: 197,717
- Users with a profile: 70,738
- Users sampled: 70,738
- Users scored after seen-item filtering: 60,373
- Users without an unseen target: 0
- Cold-start users (no training profile): 126,979 (64.22%)
- Cold-start events: 686,513 (55.49%)
- Evaluation items absent from training: 5,996
- Evaluation events absent from training: 13,755 (1.11%)
- Seen items excluded from candidates and labels: True

## Ranking metrics

| K | users | HitRate@K | Recall@K | Precision@K | MRR@K | Coverage@K |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 60,373 | 0.0366 | 0.0176 | 0.0087 | 0.0197 | 0.3336 |
| 10 | 60,373 | 0.0569 | 0.0282 | 0.0074 | 0.0223 | 0.4758 |
| 20 | 60,373 | 0.0867 | 0.0454 | 0.0063 | 0.0244 | 0.6182 |

## Diversity and novelty

- Users analysed: 5,000
- Intra-list diversity (1 - mean cosine): 0.0370
- Catalogue coverage: 0.2066
- Distinct categories per list: 1.03 (ratio 0.0515)
- Distinct brands per list: 1.55 (ratio 0.0777)
- Novelty: 16.677 bits

## Sample recommendations

| user_key | rank | product_id | cosine |
|---:|---:|---:|---:|
| -9223190590261031936 | 1 | 6000220 | 0.9327 |
| -9223190590261031936 | 2 | 6000140 | 0.9326 |
| -9223190590261031936 | 3 | 6000259 | 0.9325 |
| -9223190590261031936 | 4 | 6000138 | 0.9325 |
| -9223190590261031936 | 5 | 6000137 | 0.9324 |
| -9223190590261031936 | 6 | 6000267 | 0.9324 |
| -9223190590261031936 | 7 | 6000197 | 0.9324 |
| -9223190590261031936 | 8 | 6000258 | 0.9324 |
| -9223190590261031936 | 9 | 6000146 | 0.9323 |
| -9223190590261031936 | 10 | 6000183 | 0.9323 |

## Timings

- load: 1.53s
- select: 11.65s
- rank: 461.41s
- evaluate: 19.92s
- diversity: 4.30s
