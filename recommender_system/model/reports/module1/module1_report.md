# Module 1 - Preprocessing report

Generated: 2026-10-29T00:06:41.518426+00:00
Configuration fingerprint: `30b2b1c6ce2db62c75538a06bdc903ed46f612d509fb2aa2ec80d04d4cffbf64`

## Splits

| split | events | share | users | items | start | end |
|---|---:|---:|---:|---:|---|---|
| train | 5,773,678 | 70.00% | 709,231 | 106,481 | 2019-10-01T00:00:00+00:00 | 2019-10-05T18:10:46+00:00 |
| val | 1,237,217 | 15.00% | 191,141 | 67,026 | 2019-10-05T18:10:47+00:00 | 2019-10-06T18:15:31+00:00 |
| test | 1,237,224 | 15.00% | 197,717 | 70,295 | 2019-10-06T18:15:32+00:00 | 2019-10-07T23:59:59+00:00 |

Cutoffs: train < `2019-10-05T18:10:47+00:00` <= val < `2019-10-06T18:15:32+00:00` <= test.

## Cold start relative to train

| split | users unseen | items unseen | events on unseen items |
|---|---:|---:|---:|
| val | 117,029 (61.23%) | 4,317 (6.44%) | 0.61% |
| test | 126,979 (64.22%) | 5,996 (8.53%) | 1.11% |

## Text features (TF-IDF, fitted on train items only)

- Vocabulary size: 2,228
- Training documents: 106,481 (88,794 with usable text)
- Catalogue rows without in-vocabulary tokens: 20,467 (17.45%)
- Most frequent tokens: cat1_electronics, cat1_appliances, cat1_apparel, cat2_kitchen, cat1_computers, cat2_clocks, cat2_shoes, cat1_furniture, cat1_construction, cat1_kids, cat2_audio, cat2_tools

## Price features (fitted on train-linked products only)

- Imputation value (median): 77.2200
- log1p applied: True
- RobustScaler centering: disabled
- RobustScaler center / scale: disabled / 1.9041
- Missing prices in training slice: 0

## Fused item matrix

- Shape: 117,286 products x 2,229 features
- Non-zero entries: 343,551 (density 0.001314)
- Rows without signal: 0 (0.00%)
- Mean price share of row norm: 0.4950
- Rows L2-normalised: True
