"""Pydantic response models for the demo API.

The models are intentionally explicit: ``user_key`` is serialised as a string
because the dataset keys are signed 64-bit integers and therefore exceed the
JavaScript safe-integer range. Product identifiers stay numeric because they
are small enough for JSON clients.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FacetModel(BaseModel):
    """A label and its frequency."""

    value: str = Field(..., description="Facet label.")
    count: int = Field(..., ge=0, description="Supporting interactions or items.")


class SampleUserModel(BaseModel):
    """A curated user exposed to the dashboard."""

    user_key: str = Field(..., description="Signed 64-bit identifier as a string.")
    interactions: int = Field(..., ge=0, description="Training interactions.")
    in_evaluation_sample: bool = Field(
        ..., description="Whether Module 3 exported this user's ranking."
    )


class ProfileModel(BaseModel):
    """Summary of a user's training history."""

    user_key: str
    total_interactions: int = Field(..., ge=0)
    distinct_products: int = Field(..., ge=0)
    categories_explored: int = Field(..., ge=0)
    brands_explored: int = Field(..., ge=0)
    event_counts: dict[str, int] = Field(
        ..., description="Interaction counts keyed by event type."
    )
    top_categories: list[FacetModel]
    top_departments: list[FacetModel]
    top_brands: list[FacetModel]


class RecommendationModel(BaseModel):
    """One ranked product."""

    rank: int = Field(..., ge=1)
    product_id: int
    category_level1: str
    category_level2: str
    category_level3: str
    category_path: str = Field(..., description="Non-unknown category hierarchy.")
    brand: str
    price: float = Field(..., ge=0)
    score: float = Field(..., ge=0, le=1, description="Cosine similarity.")
    match_percent: float = Field(..., ge=0, le=100)


class RecommendationFilters(BaseModel):
    """Filters applied to the candidate set."""

    category_level1: str | None = None
    category_level2: str | None = None
    category_level3: str | None = None
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None


class RecommendResponse(BaseModel):
    """Response of ``GET /recommend``."""

    user_key: str
    profile: ProfileModel
    recommendations: list[RecommendationModel]
    returned: int = Field(..., ge=0)
    requested_top_k: int = Field(..., ge=1)
    candidates_considered: int = Field(
        ..., ge=0, description="Items still eligible after all masks."
    )
    filters: RecommendationFilters
    ranking_ms: float = Field(..., ge=0)
    model: str


class ProfileResponse(ProfileModel):
    """Response of ``GET /profile/{user_key}``."""


class UsersResponse(BaseModel):
    """Response of ``GET /users``."""

    total: int = Field(..., ge=0, description="Curated users available.")
    count: int = Field(..., ge=0, description="Users in this response.")
    users: list[SampleUserModel]


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: str
    model: dict[str, str | None]
    catalogue_items: int
    profile_users: int
    active_profiles: int
    feature_dimension: int
    sample_users: int
    load_seconds: float
    started_at: str
    artifacts: dict[str, str]
