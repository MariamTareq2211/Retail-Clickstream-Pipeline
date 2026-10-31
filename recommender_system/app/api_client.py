"""Minimal typed HTTP client used by the Streamlit dashboard.

Keeping the transport in one place means the dashboard never builds URLs or
interprets HTTP errors itself, and the same client can be reused by tests or
notebooks.
"""

from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    """Raised when the demo API is unreachable or returns an error."""

    def __init__(
        self, message: str, status_code: int | None = None, detail: Any = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class DemoApiClient:
    """Small wrapper around the FastAPI demo endpoints."""

    def __init__(self, base_url: str, timeout: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "sic-demo-dashboard/1.0"}
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Perform a GET request and normalise errors."""
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as error:
            raise ApiError(
                f"Cannot reach the recommendation API at {self.base_url}."
            ) from error

        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            if isinstance(detail, list):
                detail = "; ".join(str(item) for item in detail)
            raise ApiError(
                str(detail) or f"API returned HTTP {response.status_code}.",
                status_code=response.status_code,
                detail=detail,
            )

        try:
            return response.json()
        except ValueError as error:
            raise ApiError("The API returned a non-JSON response.") from error

    def health(self) -> dict[str, Any]:
        """Return the API health payload."""
        return self._get("/health")

    def metadata(self) -> dict[str, Any]:
        """Return filter vocabularies, model details and offline metrics."""
        return self._get("/metadata")

    def users(self, limit: int) -> dict[str, Any]:
        """Return curated users for the dashboard dropdown."""
        return self._get("/users", params={"limit": int(limit)})

    def profile(self, user_key: str) -> dict[str, Any]:
        """Return one user's profile summary."""
        return self._get(f"/profile/{user_key}")

    def recommend(
        self,
        user_key: str,
        top_k: int,
        *,
        category_level1: str | None = None,
        category_level2: str | None = None,
        category_level3: str | None = None,
        brand: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> dict[str, Any]:
        """Return ranked recommendations for one user."""
        params: dict[str, Any] = {"user_key": str(user_key), "top_k": int(top_k)}
        for name, value in (
            ("category_level1", category_level1),
            ("category_level2", category_level2),
            ("category_level3", category_level3),
            ("brand", brand),
            ("min_price", min_price),
            ("max_price", max_price),
        ):
            if value is not None:
                params[name] = value
        return self._get("/recommend", params=params)
