"""FastAPI service for the content-based recommendation demo.

The process loads the trained artefacts once during application start-up and
then serves recommendations from an in-memory CSR profile row and the CSR
catalogue matrix. Serving does not touch the disk per request.

Run from the project root::

    venv\\Scripts\\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000

Interactive documentation is available at ``http://127.0.0.1:8000/docs``.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Allow ``python app/api.py`` in addition to the documented module invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Path as PathParameter,
    Query,
    Request,
)
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import DemoSettings
from app.schemas import (
    FacetModel,
    HealthResponse,
    ProfileResponse,
    RecommendResponse,
    RecommendationFilters,
    RecommendationModel,
    SampleUserModel,
    UsersResponse,
)
from app.service import (
    DemoError,
    DemoState,
    ProfileSummary,
    Recommendation,
    RecommendationResult,
)

_LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the artefacts before accepting traffic and release them on exit."""
    settings = DemoSettings()
    _LOGGER.info("Starting recommendation API with settings: %s", settings)
    application.state.settings = settings
    application.state.demo = await run_in_threadpool(DemoState.load, settings)
    try:
        yield
    finally:
        application.state.demo = None
        _LOGGER.info("Recommendation API stopped")


app = FastAPI(
    title="SIC Content-Based Recommender",
    summary="Serving API for the time-decayed weighted-centroid model.",
    description=(
        "Recommendation demo backed by the trained Module 2 artefacts and the "
        "Module 3 hidden-test evaluation. All user keys are returned as strings "
        "because they exceed JavaScript's safe integer range."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Expose end-to-end request time for lightweight observability."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(DemoError)
async def demo_error_handler(_: Request, error: DemoError) -> JSONResponse:
    """Translate engine errors into consistent JSON error responses."""
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": str(error)},
    )


def get_demo(request: Request) -> DemoState:
    """Return the process-wide demo state or fail with 503."""
    state = getattr(request.app.state, "demo", None)
    if state is None:  # pragma: no cover - only during startup/shutdown races
        raise HTTPException(
            status_code=503, detail="Recommendation engine is not ready."
        )
    return state


def _facet_models(facets) -> list[FacetModel]:
    """Convert engine facets into response models."""
    return [FacetModel(value=facet.value, count=facet.count) for facet in facets]


def _profile_model(profile: ProfileSummary) -> ProfileResponse:
    """Convert a profile summary into its response model."""
    return ProfileResponse(
        user_key=profile.user_key,
        total_interactions=profile.total_interactions,
        distinct_products=profile.distinct_products,
        categories_explored=profile.categories_explored,
        brands_explored=profile.brands_explored,
        event_counts=profile.event_counts,
        top_categories=_facet_models(profile.top_categories),
        top_departments=_facet_models(profile.top_departments),
        top_brands=_facet_models(profile.top_brands),
    )


def _recommendation_model(item: Recommendation) -> RecommendationModel:
    """Convert one ranked item into its response model."""
    return RecommendationModel(
        rank=item.rank,
        product_id=item.product_id,
        category_level1=item.category_level1,
        category_level2=item.category_level2,
        category_level3=item.category_level3,
        category_path=item.category_path,
        brand=item.brand,
        price=item.price,
        score=item.score,
        match_percent=item.match_percent,
    )


def _recommend_response(
    result: RecommendationResult, requested_top_k: int, model_label: str
) -> RecommendResponse:
    """Assemble the complete recommendation payload."""
    return RecommendResponse(
        user_key=result.user_key,
        profile=_profile_model(result.profile),
        recommendations=[
            _recommendation_model(item) for item in result.recommendations
        ],
        returned=len(result.recommendations),
        requested_top_k=requested_top_k,
        candidates_considered=result.candidates_considered,
        filters=RecommendationFilters(**result.filters),
        ranking_ms=result.ranking_ms,
        model=model_label,
    )


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    """Return a small discovery document."""
    return {
        "service": "SIC Content-Based Recommender",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/metadata",
        "users": "/users",
        "recommend": "/recommend?user_key=<key>&top_k=10",
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(state: DemoState = Depends(get_demo)) -> HealthResponse:
    """Report process readiness and artefact dimensions."""
    return HealthResponse(**state.health())


@app.get("/users", response_model=UsersResponse, tags=["demo"])
def users(
    limit: int | None = Query(
        None, ge=1, le=500, description="Maximum number of curated users."
    ),
    state: DemoState = Depends(get_demo),
) -> UsersResponse:
    """Return curated user keys for the dashboard dropdown."""
    selected = state.users(limit)
    return UsersResponse(
        total=len(state.sample_users),
        count=len(selected),
        users=[
            SampleUserModel(
                user_key=user.user_key,
                interactions=user.interactions,
                in_evaluation_sample=user.in_evaluation_sample,
            )
            for user in selected
        ],
    )


@app.get("/metadata", tags=["demo"])
def metadata(state: DemoState = Depends(get_demo)) -> dict:
    """Return filter vocabularies, model information and offline metrics."""
    return state.metadata()


@app.get(
    "/profile/{user_key}",
    response_model=ProfileResponse,
    tags=["demo"],
)
def profile(
    user_key: str = PathParameter(
        ..., description="Signed 64-bit user key as a string."
    ),
    state: DemoState = Depends(get_demo),
) -> ProfileResponse:
    """Return a user's training-history summary."""
    return _profile_model(state.profile(user_key))


@app.get("/recommend", response_model=RecommendResponse, tags=["demo"])
def recommend(
    user_key: str = Query(
        ...,
        description="Signed 64-bit user key. Pass it as a string.",
        examples=["-9223190590261032003"],
    ),
    top_k: int | None = Query(
        None, ge=1, le=100, description="Number of recommendations to return."
    ),
    category_level1: str | None = Query(
        None, description="Department filter, e.g. 'electronics'."
    ),
    category_level2: str | None = Query(
        None, description="Category filter, e.g. 'smartphone'."
    ),
    category_level3: str | None = Query(
        None, description="Subcategory filter."
    ),
    brand: str | None = Query(None, description="Exact brand filter."),
    min_price: float | None = Query(
        None, ge=0, description="Inclusive minimum price."
    ),
    max_price: float | None = Query(
        None, ge=0, description="Inclusive maximum price."
    ),
    state: DemoState = Depends(get_demo),
) -> RecommendResponse:
    """Rank unseen items for one user after applying all filters."""
    depth = state.settings.default_top_k if top_k is None else top_k
    result = state.recommend(
        user_key=user_key,
        top_k=depth,
        category_level1=category_level1,
        category_level2=category_level2,
        category_level3=category_level3,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
    )
    label = state.model_spec.get("model", {}).get("label", "recommender")
    return _recommend_response(result, depth, str(label))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    settings = DemoSettings()
    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
