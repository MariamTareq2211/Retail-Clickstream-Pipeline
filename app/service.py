"""Serving engine for the recommendation demo.

The engine loads the trained Module 2 artefacts once at start-up and keeps the
small amount of state required for interactive requests:

* the L2-normalised catalogue matrix (CSR) and user profile matrix (CSR),
* the training seen-item mask and identifier maps,
* product metadata aligned to the catalogue row order,
* a compact, sorted representation of the training interactions used for
  profile summaries,
* a bounded cache of dense per-user score vectors.

Because both matrices are L2-normalised, scoring a user is a sparse dot product
and a top-K request only sorts the candidate scores. Category, brand, price and
seen-item masks are applied before ranking, so the API never recommends an item
the user already interacted with.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from app.config import DemoSettings
from model.config.config_schema import PipelineConfig, load_config
from model.core.paths import resolve_path
from model.core.persistence import load_json, load_parquet, load_sparse

_LOGGER = logging.getLogger(__name__)

# Event codes are defined by the extraction contract (1 = view, 2 = cart,
# 4 = purchase). Unknown codes are still counted under a synthetic label.
_EVENT_LABELS = {1: "view", 2: "cart", 4: "purchase"}

_PRODUCT_COLUMNS = (
    "category_level1",
    "category_level2",
    "category_level3",
    "brand",
    "base_price",
)


class DemoError(RuntimeError):
    """Base class for errors that map to a user-facing HTTP response."""

    status_code = 400


class DemoLoadError(DemoError):
    """Raised when a required artefact is missing or inconsistent."""

    status_code = 500


class InvalidUserKeyError(DemoError):
    """Raised when a user key is not a signed 64-bit integer."""

    status_code = 400


class UnknownUserError(DemoError):
    """Raised when a valid integer key has no profile in the artefacts."""

    status_code = 404


class EmptyProfileError(DemoError):
    """Raised when a profile contains no interaction signal."""

    status_code = 422


class InvalidFilterError(DemoError):
    """Raised when a category, brand or price filter cannot be applied."""

    status_code = 400


@dataclass(frozen=True)
class Facet:
    """A label and the number of interactions or products behind it."""

    value: str
    count: int


@dataclass(frozen=True)
class SampleUser:
    """A curated user exposed to the dashboard dropdown."""

    user_key: str
    interactions: int
    in_evaluation_sample: bool


@dataclass(frozen=True)
class ProfileSummary:
    """Compact description of a user's training history."""

    user_key: str
    total_interactions: int
    distinct_products: int
    categories_explored: int
    brands_explored: int
    event_counts: dict[str, int]
    top_categories: list[Facet]
    top_departments: list[Facet]
    top_brands: list[Facet]


@dataclass(frozen=True)
class Recommendation:
    """One ranked catalogue item."""

    rank: int
    product_id: int
    category_level1: str
    category_level2: str
    category_level3: str
    category_path: str
    brand: str
    price: float
    score: float
    match_percent: float


@dataclass(frozen=True)
class RecommendationResult:
    """Response payload for a scored user."""

    user_key: str
    profile: ProfileSummary
    recommendations: list[Recommendation]
    candidates_considered: int
    filters: dict[str, Any]
    ranking_ms: float


def _require_file(path, description: str):
    """Return ``path`` or raise a descriptive load error."""
    if not path.exists():
        raise DemoLoadError(
            f"Missing {description}: {path}. Run the offline pipelines first."
        )
    return path


def _factorize(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode string values as dense codes and return counts per code."""
    codes, uniques = pd.factorize(values.astype(str), sort=False)
    if (codes < 0).any():
        raise DemoLoadError("Product metadata contains missing filter values.")
    codes = codes.astype(np.int32)
    unique_values = np.asarray(uniques, dtype=object).astype(str)
    counts = np.bincount(codes, minlength=len(unique_values)).astype(np.int64)
    return codes, unique_values, counts


def _top_facets(
    codes: np.ndarray,
    values: np.ndarray,
    item_rows: np.ndarray,
    limit: int,
) -> list[Facet]:
    """Return the most frequent facets for a slice of interaction item rows."""
    if item_rows.size == 0:
        return []
    counts = np.bincount(codes[item_rows], minlength=len(values))
    present = np.flatnonzero(counts)
    if present.size == 0:
        return []
    ordered = sorted(
        present.tolist(),
        key=lambda index: (
            str(values[index]) == "Unknown",
            -int(counts[index]),
            str(values[index]),
        ),
    )[:limit]
    return [Facet(value=str(values[index]), count=int(counts[index])) for index in ordered]


def _facet_payload(counts: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    """Build a count-sorted facet list for the metadata endpoint.

    The ``Unknown`` sentinel is placed last so it never becomes the default
    option in the dashboard filters.
    """
    order = sorted(
        range(len(values)),
        key=lambda index: (
            str(values[index]) == "Unknown",
            -int(counts[index]),
            str(values[index]),
        ),
    )
    return [
        {"value": str(values[index]), "count": int(counts[index])}
        for index in order
        if counts[index] > 0
    ]


class DemoState:
    """Loaded artefacts and request-time recommendation logic."""

    def __init__(
        self,
        settings: DemoSettings,
        config: PipelineConfig,
        *,
        item_matrix: sp.csr_matrix,
        profiles: sp.csr_matrix,
        seen_items: sp.csr_matrix,
        item_index: pd.DataFrame,
        user_index: pd.DataFrame,
        meta: pd.DataFrame,
        model_spec: dict[str, Any],
        evaluation: dict[str, Any] | None,
        diversity: dict[str, Any] | None,
        train_users: np.ndarray,
        train_items: np.ndarray,
        train_events: np.ndarray,
        exported_users: np.ndarray,
        load_seconds: float,
        module2_dir: str,
        module3_dir: str,
    ) -> None:
        self.settings = settings
        self.config = config
        self.item_matrix = item_matrix
        self.profiles = profiles
        self.seen_items = seen_items
        self.item_index = item_index
        self.user_index = user_index
        self.meta = meta
        self.model_spec = model_spec
        self.evaluation = evaluation or {}
        self.diversity = diversity
        self.load_seconds = load_seconds
        self.module2_dir = module2_dir
        self.module3_dir = module3_dir
        self.started_at = datetime.now(timezone.utc)

        # --- Identifier lookups -------------------------------------------------
        self._item_product_ids = item_index["product_id"].to_numpy(dtype=np.int64)
        self._item_rows = item_index["item_row"].to_numpy(dtype=np.int64)

        user_keys = user_index["user_key"].to_numpy(dtype=np.int64)
        user_rows = user_index["user_row"].to_numpy(dtype=np.int64)
        order = np.argsort(user_keys, kind="stable")
        self._user_keys = user_keys[order]
        self._user_rows = user_rows[order]

        # --- Product metadata aligned to catalogue rows -------------------------
        self.product_ids = meta["product_id"].to_numpy(dtype=np.int64)
        self.prices = meta["base_price"].to_numpy(dtype=np.float64)
        self.cat1_codes, self.cat1_values, self.cat1_counts = _factorize(
            meta["category_level1"]
        )
        self.cat2_codes, self.cat2_values, self.cat2_counts = _factorize(
            meta["category_level2"]
        )
        self.cat3_codes, self.cat3_values, self.cat3_counts = _factorize(
            meta["category_level3"]
        )
        self.brand_codes, self.brand_values, self.brand_counts = _factorize(
            meta["brand"]
        )
        self.cat1_lookup = {
            str(value): index for index, value in enumerate(self.cat1_values)
        }
        self.cat2_lookup = {
            str(value): index for index, value in enumerate(self.cat2_values)
        }
        self.cat3_lookup = {
            str(value): index for index, value in enumerate(self.cat3_values)
        }
        self.brand_lookup = {
            str(value): index for index, value in enumerate(self.brand_values)
        }

        self.item_signal = item_matrix.getnnz(axis=1) > 0
        self.profile_signal = profiles.getnnz(axis=1) > 0

        # --- Training interactions, sorted by user for O(log n) slices ----------
        self._train_users = train_users
        self._train_items = train_items
        self._train_events = train_events

        # --- Bounded caches -----------------------------------------------------
        self._cached_scores = lru_cache(maxsize=settings.score_cache_size)(
            self._compute_scores
        )

        # --- Curated dashboard users -------------------------------------------
        self.sample_users = self._select_sample_users(exported_users)

        _LOGGER.info(
            "Demo state ready in %.2fs | catalogue=%s | profiles=%s | "
            "active profiles=%s | sample users=%s",
            load_seconds,
            f"{item_matrix.shape[0]:,}",
            f"{profiles.shape[0]:,}",
            f"{int(self.profile_signal.sum()):,}",
            len(self.sample_users),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls, settings: DemoSettings, config: PipelineConfig | None = None
    ) -> "DemoState":
        """Load every required artefact and validate the serving contract.

        Args:
            settings: Runtime demo settings.
            config: Optional pre-loaded pipeline configuration.

        Returns:
            A fully initialised state object.

        Raises:
            DemoLoadError: If a file is missing or an artefact shape is invalid.
        """
        started = time.perf_counter()
        pipeline = config or load_config()
        module2_dir = resolve_path(pipeline.paths.artifacts_module2)
        module3_dir = resolve_path(pipeline.paths.artifacts_module3)
        products_path = resolve_path(pipeline.paths.products)
        train_path = resolve_path(pipeline.paths.artifacts) / "splits" / "train.parquet"

        required = [
            module2_dir / "item_matrix.npz",
            module2_dir / "profiles.npz",
            module2_dir / "seen_items.npz",
            module2_dir / "item_index.parquet",
            module2_dir / "user_index.parquet",
            module2_dir / "model_spec.json",
            module3_dir / "evaluation.json",
            products_path,
            train_path,
        ]
        for path in required:
            _require_file(path, "demo artefact")

        _LOGGER.info("Loading demo artefacts from %s", module2_dir)
        item_matrix = sp.csr_matrix(load_sparse(module2_dir / "item_matrix.npz"))
        profiles = sp.csr_matrix(load_sparse(module2_dir / "profiles.npz"))
        seen_items = sp.csr_matrix(load_sparse(module2_dir / "seen_items.npz"))
        item_index = load_parquet(module2_dir / "item_index.parquet")
        user_index = load_parquet(module2_dir / "user_index.parquet")
        model_spec = load_json(module2_dir / "model_spec.json")
        evaluation = load_json(module3_dir / "evaluation.json")

        diversity_path = module3_dir / "diversity.json"
        diversity = load_json(diversity_path) if diversity_path.exists() else None

        recommendations_path = module3_dir / "recommendations.parquet"
        if recommendations_path.exists():
            exported_users = (
                pd.read_parquet(recommendations_path, columns=["user_key"])["user_key"]
                .drop_duplicates()
                .to_numpy(dtype=np.int64)
            )
        else:
            exported_users = np.empty(0, dtype=np.int64)

        # --- Shape and identifier validation --------------------------------
        if item_matrix.shape[0] != len(item_index):
            raise DemoLoadError("item_matrix rows and item_index length disagree.")
        if profiles.shape[0] != len(user_index):
            raise DemoLoadError("profiles rows and user_index length disagree.")
        if profiles.shape[1] != item_matrix.shape[1]:
            raise DemoLoadError(
                "profiles and item_matrix have different feature dimensions."
            )
        if seen_items.shape != (profiles.shape[0], item_matrix.shape[0]):
            raise DemoLoadError(
                "seen_items does not match (profiles, catalogue) dimensions."
            )
        if item_index["product_id"].duplicated().any():
            raise DemoLoadError("item_index contains duplicate product_id values.")
        if user_index["user_key"].duplicated().any():
            raise DemoLoadError("user_index contains duplicate user_key values.")

        item_index = item_index.sort_values("item_row", kind="stable").reset_index(
            drop=True
        )
        if not np.array_equal(
            item_index["item_row"].to_numpy(dtype=np.int64),
            np.arange(len(item_index), dtype=np.int64),
        ):
            raise DemoLoadError("item_index.item_row is not a dense 0..N-1 range.")
        user_index = user_index.sort_values("user_row", kind="stable").reset_index(
            drop=True
        )
        if not np.array_equal(
            user_index["user_row"].to_numpy(dtype=np.int64),
            np.arange(len(user_index), dtype=np.int64),
        ):
            raise DemoLoadError("user_index.user_row is not a dense 0..N-1 range.")

        # --- Product metadata aligned to catalogue rows ---------------------
        products = load_parquet(products_path)
        missing_columns = [c for c in ("product_id", *_PRODUCT_COLUMNS) if c not in products]
        if missing_columns:
            raise DemoLoadError(
                "Product catalogue is missing columns: " + ", ".join(missing_columns)
            )
        meta = item_index[["product_id", "item_row"]].merge(
            products[["product_id", *_PRODUCT_COLUMNS]],
            on="product_id",
            how="left",
            validate="one_to_one",
        )
        meta = meta.sort_values("item_row", kind="stable").reset_index(drop=True)
        if meta[list(_PRODUCT_COLUMNS)].isna().any().any():
            raise DemoLoadError("Some catalogue products have missing metadata.")
        for column in ("category_level1", "category_level2", "category_level3", "brand"):
            meta[column] = meta[column].astype(str)
        meta["base_price"] = meta["base_price"].astype(np.float64)
        if not np.isfinite(meta["base_price"]).all():
            raise DemoLoadError("Catalogue prices contain non-finite values.")

        # --- Compact training history for profile summaries ------------------
        _LOGGER.info("Loading training interactions for profile summaries")
        train = pd.read_parquet(
            train_path, columns=["user_key", "product_id", "event_type_key"]
        )
        row_lookup = pd.Series(
            item_index["item_row"].to_numpy(dtype=np.int64),
            index=item_index["product_id"].to_numpy(dtype=np.int64),
        )
        mapped_items = row_lookup.reindex(
            train["product_id"].to_numpy(dtype=np.int64)
        ).to_numpy()
        if np.isnan(mapped_items).any():
            raise DemoLoadError(
                "Training data references products absent from the catalogue index."
            )

        train_users = train["user_key"].to_numpy(dtype=np.int64)
        train_items = mapped_items.astype(np.int32)
        train_events = train["event_type_key"].to_numpy(dtype=np.int8)
        del train, mapped_items

        order = np.argsort(train_users, kind="stable")
        train_users = train_users[order]
        train_items = train_items[order]
        train_events = train_events[order]

        load_seconds = time.perf_counter() - started
        return cls(
            settings=settings,
            config=pipeline,
            item_matrix=item_matrix,
            profiles=profiles,
            seen_items=seen_items,
            item_index=item_index,
            user_index=user_index,
            meta=meta,
            model_spec=model_spec,
            evaluation=evaluation,
            diversity=diversity,
            train_users=train_users,
            train_items=train_items,
            train_events=train_events,
            exported_users=exported_users,
            load_seconds=load_seconds,
            module2_dir=str(module2_dir),
            module3_dir=str(module3_dir),
        )

    # ------------------------------------------------------------------
    # Identifier helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_user_key(user_key: str | int) -> int:
        """Convert a request value to a signed 64-bit integer."""
        text = str(user_key).strip()
        try:
            value = int(text)
        except (TypeError, ValueError) as error:
            raise InvalidUserKeyError(
                f"user_key must be a signed 64-bit integer, got {user_key!r}."
            ) from error
        if not -(2**63) <= value < 2**63:
            raise InvalidUserKeyError("user_key is outside the signed 64-bit range.")
        return value

    def _lookup_user_row(self, key: int) -> int:
        """Map a user key to its profile row, or raise a 404-style error."""
        position = int(np.searchsorted(self._user_keys, key))
        if position >= self._user_keys.size or int(self._user_keys[position]) != key:
            raise UnknownUserError(
                "No profile exists for this user_key. Users without training "
                "interactions are cold-start and cannot be scored; choose a key "
                "from /users."
            )
        return int(self._user_rows[position])

    def _select_sample_users(self, exported_users: np.ndarray) -> list[SampleUser]:
        """Pick a deterministic set of active users for the UI dropdown.

        Users exported by Module 3 are preferred because they are demonstrably
        scorable; if that pool is too small the selection falls back to every
        profiled user with enough training history.
        """
        unique_users, counts = np.unique(self._train_users, return_counts=True)
        eligible = counts >= self.settings.sample_min_interactions
        pool_keys = unique_users[eligible]
        if pool_keys.size == 0:
            pool_keys = unique_users
            counts = counts

        preferred = (
            pool_keys[np.isin(pool_keys, exported_users, assume_unique=True)]
            if exported_users.size
            else np.empty(0, dtype=np.int64)
        )
        candidates = (
            preferred if preferred.size >= self.settings.sample_users else pool_keys
        )
        if candidates.size == 0:
            return []

        size = min(self.settings.sample_users, candidates.size)
        rng = np.random.default_rng(self.settings.sample_seed)
        if candidates.size > size:
            positions = np.sort(rng.choice(candidates.size, size=size, replace=False))
        else:
            positions = np.arange(candidates.size)

        chosen = candidates[positions]
        chosen_counts = counts[np.searchsorted(unique_users, chosen)]
        exported_mask = (
            np.isin(chosen, exported_users, assume_unique=True)
            if exported_users.size
            else np.zeros(chosen.size, dtype=bool)
        )
        display_order = np.lexsort((chosen, -chosen_counts))
        return [
            SampleUser(
                user_key=str(int(chosen[index])),
                interactions=int(chosen_counts[index]),
                in_evaluation_sample=bool(exported_mask[index]),
            )
            for index in display_order
        ]

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------
    def _train_slice(self, key: int) -> tuple[int, int]:
        """Return the half-open slice of training rows belonging to ``key``."""
        start = int(np.searchsorted(self._train_users, key, side="left"))
        end = int(np.searchsorted(self._train_users, key, side="right"))
        return start, end

    def _profile_for_row(self, row: int, key: int) -> ProfileSummary:
        """Build the profile summary for one validated user row."""
        start, end = self._train_slice(key)
        items = self._train_items[start:end]
        events = self._train_events[start:end]
        total = int(end - start)

        seen_start = int(self.seen_items.indptr[row])
        seen_end = int(self.seen_items.indptr[row + 1])
        distinct = int(seen_end - seen_start)

        event_counts = {
            label: int(np.count_nonzero(events == code))
            for code, label in _EVENT_LABELS.items()
        }
        for code in np.unique(events):
            event_code = int(code)
            if event_code not in _EVENT_LABELS:
                event_counts[f"event_{event_code}"] = int(
                    np.count_nonzero(events == event_code)
                )

        top_n = self.settings.profile_top_n
        return ProfileSummary(
            user_key=str(key),
            total_interactions=total,
            distinct_products=distinct,
            categories_explored=int(np.count_nonzero(np.bincount(
                self.cat1_codes[items], minlength=len(self.cat1_values)
            )))
            if items.size
            else 0,
            brands_explored=int(np.count_nonzero(np.bincount(
                self.brand_codes[items], minlength=len(self.brand_values)
            )))
            if items.size
            else 0,
            event_counts=event_counts,
            top_categories=_top_facets(
                self.cat2_codes, self.cat2_values, items, top_n
            ),
            top_departments=_top_facets(
                self.cat1_codes, self.cat1_values, items, top_n
            ),
            top_brands=_top_facets(self.brand_codes, self.brand_values, items, top_n),
        )

    def profile(self, user_key: str | int) -> ProfileSummary:
        """Return a user's profile summary.

        Args:
            user_key: Signed 64-bit user identifier as a string or integer.

        Returns:
            The profile summary.

        Raises:
            InvalidUserKeyError: If the key is not an integer.
            UnknownUserError: If the key has no training profile.
        """
        key = self._parse_user_key(user_key)
        row = self._lookup_user_row(key)
        return self._profile_for_row(row, key)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    def _compute_scores(self, user_row: int) -> np.ndarray:
        """Return the immutable cosine score vector for one profile row."""
        profile = self.profiles.getrow(user_row)
        scores = np.asarray(profile.dot(self.item_matrix.T).todense()).ravel()
        np.clip(scores, 0.0, 1.0, out=scores)
        scores.setflags(write=False)
        return scores

    def _category_index(self, value: str, lookup: dict[str, int], name: str) -> int:
        """Resolve a category value or raise a descriptive filter error."""
        cleaned = str(value).strip()
        if cleaned in lookup:
            return lookup[cleaned]
        raise InvalidFilterError(
            f"Unknown {name} value {value!r}. Use /metadata to list valid values."
        )

    def _build_allowed(
        self,
        user_row: int,
        category_level1: str | None,
        category_level2: str | None,
        category_level3: str | None,
        brand: str | None,
        min_price: float | None,
        max_price: float | None,
    ) -> np.ndarray:
        """Combine seen-item and business-rule filters into one boolean mask."""
        allowed = self.item_signal.copy()

        seen_start = int(self.seen_items.indptr[user_row])
        seen_end = int(self.seen_items.indptr[user_row + 1])
        allowed[self.seen_items.indices[seen_start:seen_end]] = False

        if category_level1 is not None:
            allowed &= self.cat1_codes == self._category_index(
                category_level1, self.cat1_lookup, "category_level1"
            )
        if category_level2 is not None:
            allowed &= self.cat2_codes == self._category_index(
                category_level2, self.cat2_lookup, "category_level2"
            )
        if category_level3 is not None:
            allowed &= self.cat3_codes == self._category_index(
                category_level3, self.cat3_lookup, "category_level3"
            )
        if brand is not None:
            allowed &= self.brand_codes == self._category_index(
                brand, self.brand_lookup, "brand"
            )
        if min_price is not None:
            allowed &= self.prices >= min_price
        if max_price is not None:
            allowed &= self.prices <= max_price
        return allowed

    @staticmethod
    def _validate_price_range(
        min_price: float | None, max_price: float | None
    ) -> None:
        """Reject inverted, negative or non-finite price bounds."""
        for name, value in (("min_price", min_price), ("max_price", max_price)):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise InvalidFilterError(f"{name} must be a non-negative number.")
        if (
            min_price is not None
            and max_price is not None
            and min_price > max_price
        ):
            raise InvalidFilterError("min_price must not exceed max_price.")

    def recommend(
        self,
        user_key: str | int,
        top_k: int,
        *,
        category_level1: str | None = None,
        category_level2: str | None = None,
        category_level3: str | None = None,
        brand: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> RecommendationResult:
        """Score a user and return the top-K unseen items after filters.

        Args:
            user_key: Signed 64-bit user identifier.
            top_k: Number of recommendations to return.
            category_level1: Optional department filter.
            category_level2: Optional category filter.
            category_level3: Optional subcategory filter.
            brand: Optional brand filter.
            min_price: Optional inclusive lower price bound.
            max_price: Optional inclusive upper price bound.

        Returns:
            Recommendations, the profile summary and filter diagnostics.

        Raises:
            InvalidUserKeyError: If the key is not an integer.
            UnknownUserError: If the key has no profile.
            EmptyProfileError: If the profile carries no signal.
            InvalidFilterError: If a filter value is invalid.
        """
        started = time.perf_counter()
        if not self.settings.min_top_k <= int(top_k) <= self.settings.max_top_k:
            raise InvalidFilterError(
                f"top_k must be between {self.settings.min_top_k} and "
                f"{self.settings.max_top_k}."
            )
        self._validate_price_range(min_price, max_price)

        key = self._parse_user_key(user_key)
        user_row = self._lookup_user_row(key)
        if not bool(self.profile_signal[user_row]):
            raise EmptyProfileError(
                "This user has a profile row but no interaction signal."
            )

        profile = self._profile_for_row(user_row, key)
        allowed = self._build_allowed(
            user_row,
            category_level1,
            category_level2,
            category_level3,
            brand,
            min_price,
            max_price,
        )
        candidates = np.flatnonzero(allowed)
        recommendations: list[Recommendation] = []

        if candidates.size:
            scores = self._cached_scores(user_row)[candidates]
            take = min(int(top_k), int(candidates.size))
            # Stable sorting on candidate order (ascending catalogue row)
            # makes ties deterministic across identical requests.
            order = np.argsort(-scores, kind="stable")[:take]
            for rank, local in enumerate(order, start=1):
                score = float(scores[local])
                if score <= 0.0:
                    continue
                item_row = int(candidates[local])
                level1 = str(self.cat1_values[self.cat1_codes[item_row]])
                level2 = str(self.cat2_values[self.cat2_codes[item_row]])
                level3 = str(self.cat3_values[self.cat3_codes[item_row]])
                path = " > ".join(
                    part
                    for part in (level1, level2, level3)
                    if part and part != "Unknown"
                ) or "Unknown"
                recommendations.append(
                    Recommendation(
                        rank=rank,
                        product_id=int(self.product_ids[item_row]),
                        category_level1=level1,
                        category_level2=level2,
                        category_level3=level3,
                        category_path=path,
                        brand=str(self.brand_values[self.brand_codes[item_row]]),
                        price=round(float(self.prices[item_row]), 2),
                        score=round(score, 6),
                        match_percent=round(score * 100.0, 1),
                    )
                )

        filters = {
            "category_level1": category_level1,
            "category_level2": category_level2,
            "category_level3": category_level3,
            "brand": brand,
            "min_price": None if min_price is None else round(float(min_price), 2),
            "max_price": None if max_price is None else round(float(max_price), 2),
        }
        return RecommendationResult(
            user_key=str(key),
            profile=profile,
            recommendations=recommendations,
            candidates_considered=int(candidates.size),
            filters=filters,
            ranking_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    # ------------------------------------------------------------------
    # Public metadata
    # ------------------------------------------------------------------
    def users(self, limit: int | None = None) -> list[SampleUser]:
        """Return curated users, optionally truncated to ``limit``."""
        if limit is None:
            return list(self.sample_users)
        return list(self.sample_users[: max(int(limit), 0)])

    def health(self) -> dict[str, Any]:
        """Return a compact operational health payload."""
        model = self.model_spec.get("model", {})
        return {
            "status": "ok",
            "model": {
                "key": model.get("key"),
                "label": model.get("label"),
            },
            "catalogue_items": int(self.item_matrix.shape[0]),
            "profile_users": int(self.profiles.shape[0]),
            "active_profiles": int(self.profile_signal.sum()),
            "feature_dimension": int(self.item_matrix.shape[1]),
            "sample_users": len(self.sample_users),
            "load_seconds": round(self.load_seconds, 3),
            "started_at": self.started_at.isoformat(),
            "artifacts": {
                "module2": self.module2_dir,
                "module3": self.module3_dir,
            },
        }

    def metadata(self) -> dict[str, Any]:
        """Return filter vocabularies and the offline evaluation summary."""
        model = self.model_spec.get("model", {})
        metrics = self.evaluation.get("metrics", {})
        by_k = metrics.get("by_k", {})

        level2 = [
            {
                "value": str(self.cat2_values[index]),
                "count": int(self.cat2_counts[index]),
            }
            for index in sorted(
                range(len(self.cat2_values)),
                key=lambda index: (
                    str(self.cat2_values[index]) == "Unknown",
                    -int(self.cat2_counts[index]),
                    str(self.cat2_values[index]),
                ),
            )
            if self.cat2_counts[index] > 0
        ]
        parents: dict[str, list[str]] = {}
        for level1, level2_value in (
            self.meta[["category_level1", "category_level2"]]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            parents.setdefault(str(level2_value), []).append(str(level1))
        for entry in level2:
            entry["parents"] = sorted(parents.get(entry["value"], []))

        brands = _facet_payload(self.brand_counts, self.brand_values)
        return {
            "model": {
                "key": model.get("key"),
                "label": model.get("label"),
                "description": model.get("description"),
            },
            "recommendation_depth": {
                "min": self.settings.min_top_k,
                "default": self.settings.default_top_k,
                "max": self.settings.max_top_k,
            },
            "catalogue_items": int(self.item_matrix.shape[0]),
            "profile_users": int(self.profiles.shape[0]),
            "price": {
                "min": round(float(self.prices.min()), 2),
                "max": round(float(self.prices.max()), 2),
            },
            "categories": {
                "level1": _facet_payload(self.cat1_counts, self.cat1_values),
                "level2": level2,
                "level3": _facet_payload(self.cat3_counts, self.cat3_values),
            },
            "brands": brands[: self.settings.brand_options],
            "brand_count": len(brands),
            "evaluation": {
                "split": self.evaluation.get("protocol", {}).get(
                    "evaluation_split"
                ),
                "scored_users": metrics.get("users_scored"),
                "by_k": by_k,
                "diversity": self.diversity,
            },
        }


def build_state(settings: DemoSettings | None = None) -> DemoState:
    """Load the demo state using the process configuration."""
    return DemoState.load(settings or DemoSettings())
