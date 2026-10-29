# Demo Layer - FastAPI + Streamlit

Production-style serving and presentation layer for the Module 1-3
content-based recommender. The API loads the trained artefacts once and serves
sub-second top-K recommendations; the Streamlit dashboard provides an
interactive console over that API.

## Architecture

```
model/artifacts/module2/          model/artifacts/module3/
  item_matrix.npz                   evaluation.json
  profiles.npz                      recommendations.parquet
  seen_items.npz                    diversity.json
  item_index.parquet
  user_index.parquet
  model_spec.json
          |                                  |
          +---------------+------------------+
                          |
                    app/service.py
              (load once, mask + rank)
                          |
                     app/api.py  (FastAPI :8000)
                          |
                  HTTP / JSON (requests)
                          |
                    app/dashboard.py
                   (Streamlit :8501)
```

The dashboard never reads model artefacts directly. It calls the API, so the two
processes can be started, restarted and deployed independently.

## Prerequisites

- Python 3.11 virtual environment at `venv/`.
- Completed offline pipelines, so `model/artifacts/module2/` and
  `model/artifacts/module3/` exist.

```powershell
venv\Scripts\python.exe -m pip install -r model\requirements.txt
venv\Scripts\python.exe -m pip install -r app\requirements.txt
```

## Run

Use two terminals from the project root.

**Terminal 1 - API**

```powershell
venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive OpenAPI page.

**Terminal 2 - dashboard**

```powershell
venv\Scripts\streamlit run app/dashboard.py
```

Open `http://localhost:8501`.

The dashboard shows a clear start-up error with the exact API command if the
service is offline.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service discovery document |
| `GET` | `/health` | Readiness, model, matrix dimensions and load time |
| `GET` | `/users?limit=100` | Curated user keys for the dropdown |
| `GET` | `/metadata` | Filter vocabularies, model info and evaluation metrics |
| `GET` | `/profile/{user_key}` | Training-history summary |
| `GET` | `/recommend` | Ranked, filtered, unseen recommendations |

### `/recommend` parameters

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `user_key` | string | required | Signed 64-bit key, always sent as a string |
| `top_k` | integer | `10` | Bounded by `DEMO_MIN_TOP_K`/`DEMO_MAX_TOP_K` (5-20 by default) |
| `category_level1` | string | none | Department, for example `electronics` |
| `category_level2` | string | none | Category, for example `smartphone` |
| `category_level3` | string | none | Subcategory |
| `brand` | string | none | Exact brand value |
| `min_price` | float | none | Inclusive lower bound |
| `max_price` | float | none | Inclusive upper bound |

Example:

```powershell
curl "http://127.0.0.1:8000/recommend?user_key=-9223190590261032003&top_k=10&min_price=50&max_price=500"
```

Each response includes the profile summary, the applied filters, the number of
eligible candidates and the engine latency:

```json
{
  "user_key": "-9223190590261032003",
  "profile": {"total_interactions": 51, "top_categories": [], "top_brands": []},
  "recommendations": [
    {
      "rank": 1,
      "product_id": 1005180,
      "category_path": "electronics > smartphone",
      "brand": "samsung",
      "price": 1158.05,
      "score": 0.862828,
      "match_percent": 86.3
    }
  ],
  "returned": 10,
  "requested_top_k": 10,
  "candidates_considered": 117240,
  "ranking_ms": 31.2,
  "model": "Model 3 - Time-Decayed Weighted Centroid"
}
```

## Serving design

- **Single load.** `app/service.py` reads the CSR matrices, identifier maps,
  product metadata and training history once inside the FastAPI lifespan.
  Requests never touch the disk.
- **Sparse cosine.** Profiles and catalogue rows are L2-normalised, so the score
  vector is `profile @ item_matrix.T`. One row is scored at a time; the result is
  a dense 117,286-element vector.
- **Masks before ranking.** Seen items, category, brand and price filters are
  combined into a boolean candidate mask before the top-K cut. A user can never
  receive a product they already interacted with in training.
- **Deterministic ties.** Candidates are sorted by descending cosine with a
  stable sort over the catalogue row order.
- **Bounded caches.** The dense score vector for the most recent
  `DEMO_SCORE_CACHE_SIZE` users is cached, so switching filters on the same
  profile avoids recomputing the dot product.
- **String keys.** `user_key` values are signed `int64` near `-9.2e18`, which
  exceeds JavaScript's safe integer range. The API serialises them as strings.
- **Explicit failures.** Unknown keys return `404` with guidance to `/users`;
  malformed keys, invalid filters and out-of-range depths return `400`.

## Dashboard features

- Curated sample-user dropdown plus a custom `user_key` form.
- Profile snapshot: interactions, distinct products, departments, brands, event
  mix and top-category/top-brand bars.
- Responsive recommendation cards with product ID, category path, brand, price,
  cosine score and a percentage-match progress bar.
- Table view with a Streamlit progress column and CSV export.
- Sidebar filters: top-K slider (5-20), cascading department/category selects,
  brand select and price range. Every change re-queries the API.
- Dark and light palettes, custom cards and metric tiles.

## Configuration

All values are environment variables read by `app/config.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEMO_API_HOST` | `127.0.0.1` | API bind interface |
| `DEMO_API_PORT` | `8000` | API port |
| `DEMO_API_URL` | `http://<host>:<port>` | Base URL used by the dashboard |
| `DEMO_DEFAULT_TOP_K` | `10` | Default recommendation depth |
| `DEMO_MIN_TOP_K` | `5` | Smallest accepted depth |
| `DEMO_MAX_TOP_K` | `20` | Largest accepted depth |
| `DEMO_SAMPLE_USERS` | `100` | Curated users returned by `/users` |
| `DEMO_SAMPLE_MIN_INTERACTIONS` | `5` | Minimum history for a curated user |
| `DEMO_SAMPLE_SEED` | `42` | Deterministic sample selection |
| `DEMO_BRAND_OPTIONS` | `200` | Brands exposed to the dashboard |
| `DEMO_PROFILE_TOP_N` | `5` | Facets returned per profile section |
| `DEMO_SCORE_CACHE_SIZE` | `32` | Per-user score vectors kept in memory |
| `DEMO_REQUEST_TIMEOUT` | `8.0` | Dashboard HTTP timeout in seconds |

To run on different ports:

```powershell
$env:DEMO_API_PORT="9000"
venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 9000

$env:DEMO_API_URL="http://127.0.0.1:9000"
venv\Scripts\streamlit run app/dashboard.py
```

## Verification

```powershell
venv\Scripts\python.exe -m app.tests.smoke_api
venv\Scripts\python.exe -m app.tests.benchmark_api
```

`smoke_api` verifies every endpoint, ranking order, score bounds, seen-item
exclusion, each filter, error handling and p95 latency. `benchmark_api` reports
p50/p95/p99 for five request shapes and fails if p95 is not sub-second.

Verified on the local artefacts: 16/16 smoke checks pass, p95 latency around
30 ms for warm requests.

## Project layout

```
app/
  __init__.py
  api.py                 FastAPI application and routes
  service.py             Artefact loading, profile summaries, ranking
  schemas.py             Pydantic response models
  config.py              Environment-driven settings
  api_client.py          HTTP client used by the dashboard
  dashboard.py           Streamlit page and interaction flow
  components.py          HTML cards, profile panels and empty states
  styles.py              Dark/light design tokens and CSS
  requirements.txt       Pinned UI/API dependencies
  .streamlit/config.toml Streamlit base theme
  tests/
    smoke_api.py         Endpoint and contract checks
    benchmark_api.py     Latency SLO benchmark
  README.md
```

## Notes and limitations

- A score is a cosine similarity between the user's time-decayed weighted
  centroid and the product vector. It is a ranking signal, not a calibrated
  purchase probability.
- The profile matrix covers users with training interactions. Users who appear
  only in the test split are cold-start and deliberately return `404`.
- The dashboard exposes the 200 most common brands; the API accepts any brand
  present in `/metadata` (use the full metadata response for rare brands).
- With a very narrow category and price filter, fewer than `top_k` products may
  remain. The UI states this explicitly and never pads the list with seen items.
