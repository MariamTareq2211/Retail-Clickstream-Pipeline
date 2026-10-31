"""Environment-driven settings shared by the API and the dashboard.

The demo layer keeps every tunable value here instead of scattering literals
across the routes and UI widgets. Values are read from environment variables at
construction time so the same code runs unchanged in local, container and test
environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a clear failure message."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from error


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a clear failure message."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from error


@dataclass(frozen=True)
class DemoSettings:
    """Runtime settings for both demo processes.

    Attributes:
        api_host: Interface the FastAPI server binds to.
        api_port: Port the FastAPI server binds to.
        api_url: Base URL used by the Streamlit dashboard.
        default_top_k: Default number of recommendations.
        min_top_k: Smallest accepted recommendation depth.
        max_top_k: Largest accepted recommendation depth.
        sample_users: Number of curated users returned by ``/users``.
        sample_min_interactions: Minimum training history for a sample user.
        sample_seed: Deterministic seed for sample-user selection.
        brand_options: Number of brands exposed to the dashboard filter.
        profile_top_n: Facets returned per profile section.
        score_cache_size: Per-user dense score vectors kept in memory.
        request_timeout_seconds: Dashboard HTTP timeout.
    """

    api_host: str = field(
        default_factory=lambda: os.getenv("DEMO_API_HOST", "127.0.0.1")
    )
    api_port: int = field(default_factory=lambda: _env_int("DEMO_API_PORT", 8000))
    api_url: str = field(default_factory=lambda: os.getenv("DEMO_API_URL", ""))
    default_top_k: int = field(
        default_factory=lambda: _env_int("DEMO_DEFAULT_TOP_K", 10)
    )
    min_top_k: int = field(default_factory=lambda: _env_int("DEMO_MIN_TOP_K", 5))
    max_top_k: int = field(default_factory=lambda: _env_int("DEMO_MAX_TOP_K", 20))
    sample_users: int = field(
        default_factory=lambda: _env_int("DEMO_SAMPLE_USERS", 100)
    )
    sample_min_interactions: int = field(
        default_factory=lambda: _env_int("DEMO_SAMPLE_MIN_INTERACTIONS", 5)
    )
    sample_seed: int = field(
        default_factory=lambda: _env_int("DEMO_SAMPLE_SEED", 42)
    )
    brand_options: int = field(
        default_factory=lambda: _env_int("DEMO_BRAND_OPTIONS", 200)
    )
    profile_top_n: int = field(
        default_factory=lambda: _env_int("DEMO_PROFILE_TOP_N", 5)
    )
    score_cache_size: int = field(
        default_factory=lambda: _env_int("DEMO_SCORE_CACHE_SIZE", 32)
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: _env_float("DEMO_REQUEST_TIMEOUT", 8.0)
    )

    def __post_init__(self) -> None:
        if not 1 <= self.min_top_k <= self.default_top_k <= self.max_top_k:
            raise ValueError(
                "Recommendation bounds must satisfy "
                "1 <= DEMO_MIN_TOP_K <= DEMO_DEFAULT_TOP_K <= DEMO_MAX_TOP_K."
            )
        if self.max_top_k > 100:
            raise ValueError("DEMO_MAX_TOP_K must not exceed 100.")
        if self.sample_users < 1:
            raise ValueError("DEMO_SAMPLE_USERS must be positive.")
        if self.sample_min_interactions < 1:
            raise ValueError("DEMO_SAMPLE_MIN_INTERACTIONS must be positive.")
        if self.profile_top_n < 1:
            raise ValueError("DEMO_PROFILE_TOP_N must be positive.")
        if self.score_cache_size < 0:
            raise ValueError("DEMO_SCORE_CACHE_SIZE must be zero or positive.")
        if self.request_timeout_seconds <= 0:
            raise ValueError("DEMO_REQUEST_TIMEOUT must be positive.")
        if not self.api_url:
            object.__setattr__(
                self, "api_url", f"http://{self.api_host}:{self.api_port}"
            )
