# Content-Based Recommendation System — Offline Pipeline

Modular implementation of the architecture in `../Plan.md`. All three stages are
implemented and exercised on the real dataset:

1. **Module 1** — validation, deduplication, chronological split and sparse
   TF-IDF + price feature fusion.
2. **Module 2** — user-profile construction, a three-model stability benchmark
   and Optuna tuning of the winning model.
3. **Module 3** — cosine candidate scoring, business-rule filtering, top-K
   ranking, hidden-test evaluation and diversity/novelty diagnostics.

## Status

| Stage | Contents | State |
|---|---|---|
| Module 1 | load → validate → dedup → temporal split → TF-IDF + price → sparse fusion | **complete** |
| Module 2 | profiles, 3-model benchmark, Optuna tuning, artifact export | **complete** |
| Module 3 | cosine top-K, HitRate/Recall/Precision/MRR, diversity/novelty | **complete** |

## Prerequisites

```powershell
# 1. Dependencies, installed strictly inside the project virtual environment
venv\Scripts\python.exe -m pip install -r model\requirements.txt

# 2. Raw data (only when the warehouse changed). Requires interactive Azure login.
venv\Scripts\python.exe diagnosis\fabric_to_my_machine.py --check-only
venv\Scripts\python.exe diagnosis\fabric_to_my_machine.py
```

## Running the pipeline

Run the stages in order from the project root:

```powershell
venv\Scripts\python.exe -m model.pipelines.module1_preprocess
venv\Scripts\python.exe -m model.pipelines.module2_train
venv\Scripts\python.exe -m model.pipelines.module3_predict
```

Every stage accepts `--config path\to\config.yaml`. Exit code `0` means success,
`2` means a configuration, artifact-contract or data failure. Logs and markdown
summaries are written to `model/reports/module{1,2,3}/`.

## Layout

```
model/
├── config/     config.yaml (all knobs) + config_schema.py (typed, validated)
├── core/       paths, logging, seeding, atomic persistence
├── data/       loader, validators, deduplication, temporal_split
├── features/   text_features (TF-IDF), price_features, item_matrix (fusion)
├── profiles/   interaction_weights, time_decay, user_profiles
├── models/     model registry, centroid builders, stability benchmark
├── ranking/    cosine top-K scorer, seen-item / ground-truth filters
├── evaluation/ metrics (HitRate, Recall, Precision, MRR), evaluator, diversity
├── tuning/     Optuna search space, per-trial matrix factory, study
├── pipelines/  module1_preprocess.py, module2_train.py, module3_predict.py
├── artifacts/  generated: matrices, splits, profiles, recommendations, manifests
├── reports/    generated: markdown summaries, logs
├── tests/      synthetic smoke tests for Modules 1-3
└── requirements.txt
```

## Artifact contract

### Module 1 → Module 2

| Artifact | Content |
|---|---|
| `artifacts/module1/item_matrix.npz` | CSR `float64`, 117,286 × 2,229 (2,228 TF-IDF + 1 price), L2-normalised |
| `artifacts/module1/item_index.parquet` | `product_id`, `item_row` — row mapping for the matrix |
| `artifacts/module1/splits/{train,val,test}.parquet` | `user_key`, `product_id`, `event_type_key`, `event_ts`, `interaction_count`, `split` |
| `artifacts/module1/tfidf_vectorizer.joblib` | Fitted on **train items only** |
| `artifacts/module1/price_transformer.joblib` | Train median + `log1p` + non-centred `RobustScaler` |
| `artifacts/module1/split_manifest.json` | Dedup stats, split audit, cold-start report |
| `artifacts/module1/feature_manifest.json` | Vocabulary, zero-text rows, fusion diagnostics |
| `artifacts/module1/module1_manifest.json` | Config fingerprint, timings, SHA-256 of every artifact |

### Module 2 → Module 3

| Artifact | Content |
|---|---|
| `artifacts/module2/item_matrix.npz` | Catalogue matrix rebuilt with the **tuned** TF-IDF/fusion parameters |
| `artifacts/module2/profiles.npz` | L2-normalised user profiles, aligned with `user_index` |
| `artifacts/module2/seen_items.npz` | Binary user × item mask of training interactions (business rule) |
| `artifacts/module2/item_index.parquet` | `product_id`, `item_row` for the exported matrix |
| `artifacts/module2/user_index.parquet` | `user_key`, `user_row` for the exported profiles |
| `artifacts/module2/best_params.json` | Selected model, tuned parameters and replay metadata |
| `artifacts/module2/benchmark.json` | Per-model metrics, validation-window stability, selection rationale |
| `artifacts/module2/model_spec.json` | Model behaviour flags, decay diagnostics, feature/user counts |
| `artifacts/module2/tuning_trials.csv` | Full Optuna trial table (present when tuning ran) |
| `artifacts/module2/module2_manifest.json` | Config fingerprints, timings, SHA-256 of every artifact |

> **Contract:** Module 3 must consume **Module 2's** `item_matrix.npz`. Tuning may
> change `max_features`, `ngram_range`, `min_df` or `price_weight`, so the Module 1
> matrix is not guaranteed to share the profile feature space.

### Module 3 outputs

| Artifact | Content |
|---|---|
| `artifacts/module3/recommendations.parquet` | `user_key`, `rank`, `item_row`, `product_id`, `score` (top-K per exported user) |
| `artifacts/module3/evaluation.json` | Protocol, user/cold-start diagnostics and metrics per cut-off |
| `artifacts/module3/diversity.json` | Intra-list diversity, coverage, category/brand diversity, novelty |
| `artifacts/module3/module3_manifest.json` | Config fingerprints, timings, metrics and SHA-256 of outputs |

## Leakage guarantees

1. Deduplication and the split use the **raw timestamps**; no future row can enter train.
2. The TF-IDF vocabulary and IDF weights are fitted on items with **training**
   interactions; test-only tokens stay out of vocabulary.
3. Price imputation median and `RobustScaler` statistics are fitted on
   **train-linked products** only.
4. User profiles aggregate **training interactions only**; validation and test
   rows never influence a profile.
5. The seen-item mask is built from training pairs and is applied to both the
   candidate list and the relevance labels, so a repeated purchase can never be
   counted as a miss.
6. Train/val/test are asserted to be temporally disjoint before anything is saved.
7. Configuration fingerprints and artifact checksums make every matrix traceable
   to the exact configuration that produced it.

## Verified on the real dataset

The interaction log spans **2019-10-01 → 2019-10-07** (7 days, 8,248,730 events).

| Split | Events | Share | Users | Items |
|---|---:|---:|---:|---:|
| train | 5,773,678 | 70.00% | 709,231 | 106,481 |
| val | 1,237,217 | 15.00% | 191,141 | 67,026 |
| test | 1,237,224 | 15.00% | 197,717 | 70,295 |

### Module 1

- TF-IDF: 2,228 features; 20,467 catalogue rows (17.45%) have no in-vocabulary token.
- Item matrix: 117,286 × 2,229, 343,551 non-zeros, **0 all-zero rows**.
- **Deduplication:** with `event_ts` present, only **611** rows are exact
  duplicates. The 3.19M rows that looked duplicated in the 3-column export are
  genuine repeat views at different moments; collapsing them across time (the
  `collapsed` grain) reproduces the 5,056,471-row figure but destroys the decay
  signal. Default is therefore `timestamped`.
- **Price block:** centring is disabled so a product at the median price keeps a
  non-zero feature. That makes the price block strictly positive and much larger
  than its centred version, so `fusion.price_weight` was lowered from 1.0 to
  **0.35** (price share of the row norm 0.843 → 0.495) to keep text dominant.

### Module 2

Benchmark on **15,000** sampled validation users (12,449 scorable after the
seen-item filter), mean ± std over three chronological windows at K = 10:

| Model | Recall@10 mean | Recall@10 std | HitRate@10 mean | Overall Recall@10 |
|---|---:|---:|---:|---:|
| **Model 3 — Time-Decayed Weighted** | **0.027591** | 0.002429 | **0.053464** | **0.028278** |
| Model 2 — Interaction-Weighted | 0.026026 | 0.001919 | 0.051051 | 0.026808 |
| Model 1 — Unweighted | 0.023478 | 0.000983 | 0.045578 | 0.024415 |

**Winner: Model 3 (time-decayed).** Optuna ran 12 trials on 5,000 sampled users
and reached `Recall@10 = 0.031335` with:

```text
max_features=4000, ngram_range=(1, 2), min_df=3, price_weight=0.050872
decay_rate_per_hour=0.082565 (8.40 h half-life), count_transform=raw
event_weights = {view: 0.2186, cart: 1.7631, purchase: 1.0}
```

The final matrix is 117,286 × 4,001 (471,170 non-zeros, 0 zero rows) and the
profiles are 709,231 × 4,001. The benchmark selects by **mean Recall@K over
three chronological validation windows**, breaking ties by mean HitRate@K, then
by lower Recall standard deviation. The purchase weight is anchored at 1.0
during tuning because cosine similarity is invariant to a global rescaling of
the user vector; only the view:purchase and cart:purchase ratios are identifiable.
Decay is expressed **per hour** because the log spans only one week.

### Module 3

Hidden-test evaluation of the tuned Model 3 profiles at K = 5/10/20:

| K | users | HitRate@K | Recall@K | Precision@K | MRR@K | Coverage@K |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 60,373 | 0.0366 | 0.0176 | 0.0087 | 0.0197 | 0.3336 |
| 10 | 60,373 | 0.0569 | 0.0282 | 0.0074 | 0.0223 | 0.4758 |
| 20 | 60,373 | 0.0867 | 0.0454 | 0.0063 | 0.0244 | 0.6182 |

- Test split: 197,717 users; **64.22% are cold-start** (no training profile) and
  account for 55.49% of test events. 70,738 users had a profile; 60,373 remained
  after removing users whose every target was already seen in training.
- `Recall@10 = 0.0282` matches the validation result (`0.0283`), so the model did
  not degrade on the later temporal window.
- Only 1.11% of test events involve an item absent from training, so item cold
  start is negligible compared with user cold start.
- Diversity on a 5,000-user sample: intra-list diversity 0.0370, catalogue
  coverage 0.2066, 1.03 distinct categories and 1.55 distinct brands per list,
  novelty 16.68 bits.
- A 500-user audit confirmed **zero** recommendations repeat a training item and
  every list is sorted by descending cosine similarity.

## Configuration

`config/config.yaml` is the only place numbers live. Key knobs:

- `split.*`, `dedup.*` — chronology and repeat policy.
- `text.*`, `price.*`, `fusion.price_weight` — feature engineering.
- `profiles.weights`, `profiles.decay.*` — profile formula.
- `benchmark.*` — model-selection sample, cut-off and stability windows.
- `tuning.*` — Optuna budget, objective and search space.
- `evaluation.*` — metric cut-offs and scoring batch size.
- `module3.*` — evaluation split, sampling and export/ diversity depth.

Unknown keys and inconsistent values are rejected at load time by
`config_schema.py`.

## Testing

```powershell
venv\Scripts\python.exe -m model.tests.smoke_module1
venv\Scripts\python.exe -m model.tests.smoke_module2
venv\Scripts\python.exe -m model.tests.smoke_module3
```

Each test builds a synthetic catalogue in a temporary folder. Module 1 checks
split integrity, dedup behaviour, train-only vocabulary and matrix
normalisation; Module 2 adds the three-model benchmark, tuning replay and
artifact shapes; Module 3 adds metric ranges, the no-seen-recommendation rule,
recommendation export and diversity diagnostics.
