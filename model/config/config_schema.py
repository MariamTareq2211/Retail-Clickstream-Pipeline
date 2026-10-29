"""Typed, validated configuration contract for the recommendation pipeline.

Blueprint reference: Plan.md - Modules 1-3. The preprocessing sections are
consumed by Module 1; ``profiles``, ``benchmark``, ``tuning`` and ``evaluation``
drive Module 2 (Module 3 reuses ``evaluation``).

Why this module exists
----------------------
YAML is convenient to edit but dangerous to consume directly: a typo in a key
name silently falls back to a default and the experiment becomes unreproducible.
This module converts the raw YAML into frozen dataclasses, rejects unknown keys,
type-checks every value and validates the relationships between them (ratios
summing to one, ascending cutoffs, and so on) before any data is loaded.

Typical use::

    from model.config.config_schema import load_config

    config = load_config()
    config.split.train_ratio        # 0.7
    config.text.tfidf.min_df        # 2

Public API
----------
``load_config``      Read YAML, apply overrides, validate, return ``PipelineConfig``.
``config_hash``      Stable SHA-256 of the effective configuration.
``ConfigError``      Raised for any structural or semantic problem.
"""

from __future__ import annotations

import copy
import hashlib
import json
import types
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Section schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathsConfig:
    """Input and output locations, resolved against the project root."""

    products: str = "data/dim_product_clean.parquet"
    events: str = "data/fact_events_clean.parquet"
    artifacts: str = "model/artifacts/module1"
    reports: str = "model/reports/module1"
    artifacts_module2: str = "model/artifacts/module2"
    reports_module2: str = "model/reports/module2"
    # Module 3 consumes Module 2's artifacts and writes predictions/evaluation.
    artifacts_module3: str = "model/artifacts/module3"
    reports_module3: str = "model/reports/module3"


@dataclass(frozen=True)
class LoggingConfig:
    """Logging verbosity and the per-stage log file name."""

    level: str = "INFO"
    file_name: str | None = "module1.log"

    def __post_init__(self) -> None:
        if self.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"logging.level must be a standard level, got {self.level!r}.")


@dataclass(frozen=True)
class DataConfig:
    """Input schema contract, including the mandatory event timestamp."""

    timestamp_column: str = "event_ts"
    required_product_columns: tuple[str, ...] = (
        "product_id",
        "category_code",
        "category_level1",
        "category_level2",
        "category_level3",
        "brand",
        "base_price",
    )
    required_event_columns: tuple[str, ...] = ("user_key", "product_id", "event_type_key")

    def __post_init__(self) -> None:
        if not self.timestamp_column:
            raise ConfigError("data.timestamp_column must not be empty.")
        if self.timestamp_column in self.required_event_columns:
            raise ConfigError(
                "data.timestamp_column must not be duplicated in required_event_columns; "
                "it is validated separately."
            )


@dataclass(frozen=True)
class DedupConfig:
    """Deduplication grain and the repeat-count policy forwarded to Module 2."""

    grain: str = "timestamped"
    count_transform: str = "raw"
    # Upper bound used by the "capped" count transform. Five repeats already
    # signal strong interest; beyond that a session is usually a page refresh.
    count_cap: int = 5

    def __post_init__(self) -> None:
        if self.grain not in {"timestamped", "collapsed"}:
            raise ConfigError(
                f"dedup.grain must be 'timestamped' or 'collapsed', got {self.grain!r}."
            )
        if self.count_transform not in {"raw", "log1p", "capped", "none"}:
            raise ConfigError(
                "dedup.count_transform must be one of "
                f"'raw', 'log1p', 'capped', 'none'; got {self.count_transform!r}."
            )
        if self.count_cap < 1:
            raise ConfigError("dedup.count_cap must be at least 1.")


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train / validation / test proportions."""

    mode: str = "quantile"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    fixed_cutoffs: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"quantile", "fixed_date"}:
            raise ConfigError(f"split.mode must be 'quantile' or 'fixed_date', got {self.mode!r}.")

        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios):
            raise ConfigError("split ratios must be non-negative.")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ConfigError(f"split ratios must sum to 1.0, got {sum(ratios):.6f}.")
        if self.train_ratio <= 0:
            raise ConfigError("split.train_ratio must be greater than zero.")
        if self.val_ratio + self.test_ratio <= 0:
            raise ConfigError("At least one of val_ratio / test_ratio must be positive.")

        if self.mode == "fixed_date":
            if self.fixed_cutoffs is None or len(self.fixed_cutoffs) != 2:
                raise ConfigError("split.fixed_cutoffs must contain exactly two ISO timestamps.")
            first, second = self.fixed_cutoffs
            if _parse_iso(first) >= _parse_iso(second):
                raise ConfigError("split.fixed_cutoffs must be strictly ascending.")


@dataclass(frozen=True)
class TfidfConfig:
    """Vectorizer hyperparameters; fitted on training items only."""

    max_features: int = 5000
    min_df: int | float = 2
    ngram_range: tuple[int, int] = (1, 1)
    sublinear_tf: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_features, int) or self.max_features <= 0:
            raise ConfigError("text.tfidf.max_features must be a positive integer.")
        if isinstance(self.min_df, float) and not 0.0 < self.min_df <= 1.0:
            raise ConfigError("A fractional min_df must lie in (0, 1].")
        if isinstance(self.min_df, int) and self.min_df < 1:
            raise ConfigError("An integer min_df must be at least 1.")
        low, high = self.ngram_range
        if low < 1 or high < low:
            raise ConfigError(f"Invalid ngram_range {self.ngram_range!r}.")


@dataclass(frozen=True)
class TextConfig:
    """How catalogue attributes become namespaced TF-IDF tokens."""

    field_prefixes: Mapping[str, str] = field(
        default_factory=lambda: {
            "category_level1": "cat1",
            "category_level2": "cat2",
            "category_level3": "cat3",
            "brand": "brand",
        }
    )
    drop_unknown: bool = True
    unknown_value: str = "Unknown"
    include_category_path: bool = False
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)

    def __post_init__(self) -> None:
        if not self.field_prefixes:
            raise ConfigError("text.field_prefixes must define at least one field.")
        prefixes = list(self.field_prefixes.values())
        if len(prefixes) != len(set(prefixes)):
            raise ConfigError("text.field_prefixes values must be unique.")
        if any(not prefix for prefix in prefixes):
            raise ConfigError("text.field_prefixes values must not be empty.")


@dataclass(frozen=True)
class PriceConfig:
    """Imputation, skew correction and scaling for ``base_price``."""

    column: str = "base_price"
    impute_strategy: str = "median"
    constant_value: float = 0.0
    log_transform: bool = True
    clip_negative: bool = True
    # Centering is disabled on purpose. With centering, a product priced exactly
    # at the training median gets a price feature of 0; combined with empty
    # metadata it becomes an all-zero row that can never be recommended. Without
    # centering the price block is strictly non-negative, so every product keeps
    # a signal and cosine similarity stays well defined.
    scaler_centering: bool = False

    def __post_init__(self) -> None:
        if self.impute_strategy not in {"median", "constant"}:
            raise ConfigError(
                f"price.impute_strategy must be 'median' or 'constant', got {self.impute_strategy!r}."
            )


@dataclass(frozen=True)
class FusionConfig:
    """How the text block and the price block are combined."""

    price_weight: float = 1.0
    normalize_rows: bool = True

    def __post_init__(self) -> None:
        if self.price_weight < 0:
            raise ConfigError("fusion.price_weight must be non-negative.")


@dataclass(frozen=True)
class DecayConfig:
    """Exponential time decay applied to interaction weights.

    The rate is expressed per hour (``lambda``) because the interaction log spans
    only seven days; a rate of ``ln(2) / 24 ~= 0.0289`` corresponds to a 24-hour
    half-life, which is a realistic recency scale for this dataset.
    """

    enabled: bool = True
    reference: str = "global"
    rate_per_hour: float = 0.0289

    def __post_init__(self) -> None:
        if self.reference not in {"global", "per_user"}:
            raise ConfigError(
                f"profiles.decay.reference must be 'global' or 'per_user', got {self.reference!r}."
            )
        if self.rate_per_hour <= 0:
            raise ConfigError("profiles.decay.rate_per_hour must be positive.")
        if self.rate_per_hour > 10:
            raise ConfigError(
                "profiles.decay.rate_per_hour above 10/h decays interactions to nothing "
                "within minutes; check the unit (it is per hour, not per day)."
            )


@dataclass(frozen=True)
class ProfilesConfig:
    """Event-type weights and decay settings for user-profile construction."""

    weights: Mapping[int, float] = field(
        default_factory=lambda: {1: 1.0, 2: 3.0, 4: 5.0}
    )
    decay: DecayConfig = field(default_factory=DecayConfig)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ConfigError("profiles.weights must define at least one event type.")
        for event_type, weight in self.weights.items():
            if not isinstance(event_type, int) or event_type < 0:
                raise ConfigError(f"profiles.weights keys must be non-negative ints, got {event_type!r}.")
            if weight < 0:
                raise ConfigError(f"profiles.weights[{event_type}] must be non-negative.")
        if max(self.weights.values()) <= 0:
            raise ConfigError("profiles.weights must contain at least one positive weight.")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validation settings for the three-model stability benchmark."""

    k: int = 10
    max_users: int = 30000
    stability_windows: int = 3

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ConfigError("benchmark.k must be at least 1.")
        if self.max_users < 1:
            raise ConfigError("benchmark.max_users must be at least 1.")
        if self.stability_windows < 1:
            raise ConfigError("benchmark.stability_windows must be at least 1.")


@dataclass(frozen=True)
class SearchSpaceConfig:
    """Search ranges and candidate values offered to Optuna."""

    view_weight: tuple[float, float] = (0.02, 1.0)
    cart_weight: tuple[float, float] = (0.05, 3.0)
    max_features: tuple[int, ...] = (1000, 2000, 4000)
    ngram_ranges: tuple[tuple[int, int], ...] = ((1, 1), (1, 2))
    min_df: tuple[int, ...] = (1, 2, 3)
    price_weight: tuple[float, float] = (0.05, 1.5)
    decay_rate: tuple[float, float] = (1e-4, 0.2)
    count_transforms: tuple[str, ...] = ("raw", "log1p", "capped", "none")

    def __post_init__(self) -> None:
        for name in ("view_weight", "cart_weight", "price_weight", "decay_rate"):
            low, high = getattr(self, name)
            if low <= 0 or high <= low:
                raise ConfigError(f"tuning.search_space.{name} must be an ascending positive range.")
        if not self.max_features or min(self.max_features) < 1:
            raise ConfigError("tuning.search_space.max_features must list positive integers.")
        if not self.min_df or min(self.min_df) < 1:
            raise ConfigError("tuning.search_space.min_df must list positive integers.")
        if not self.ngram_ranges:
            raise ConfigError("tuning.search_space.ngram_ranges must not be empty.")
        for low, high in self.ngram_ranges:
            if low < 1 or high < low:
                raise ConfigError(
                    f"Invalid ngram range ({low}, {high}) in tuning.search_space."
                )
        if not self.count_transforms:
            raise ConfigError("tuning.search_space.count_transforms must not be empty.")
        allowed = {"raw", "log1p", "capped", "none"}
        if set(self.count_transforms) - allowed:
            raise ConfigError(f"count_transforms must be a subset of {sorted(allowed)}.")


@dataclass(frozen=True)
class TuningConfig:
    """Optuna study configuration for the winning model."""

    enabled: bool = True
    n_trials: int = 20
    max_users: int = 8000
    timeout_seconds: int = 1800
    metric: str = "recall"
    study_name: str = "module2_centroid"
    seed_offset: int = 1
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ConfigError("tuning.n_trials must be at least 1.")
        if self.max_users < 1:
            raise ConfigError("tuning.max_users must be at least 1.")
        if self.metric not in {"recall", "hit_rate", "mrr"}:
            raise ConfigError("tuning.metric must be 'recall', 'hit_rate' or 'mrr'.")


@dataclass(frozen=True)
class EvaluationConfig:
    """Ranking metrics and eligibility rules shared by Modules 2 and 3."""

    k_values: tuple[int, ...] = (5, 10, 20)
    min_train_interactions: int = 1
    min_test_interactions: int = 1
    # Number of users scored per matrix multiplication block. 128 keeps the
    # dense score block near 120 MB for a 117k-item catalogue.
    scoring_batch_size: int = 128
    # Business rule from Module 3: never recommend an item the user already
    # interacted with in the training window.
    exclude_seen_items: bool = True

    def __post_init__(self) -> None:
        if not self.k_values or any(k < 1 for k in self.k_values):
            raise ConfigError("evaluation.k_values must list positive integers.")
        if self.min_train_interactions < 1:
            raise ConfigError("evaluation.min_train_interactions must be at least 1.")
        if self.min_test_interactions < 1:
            raise ConfigError("evaluation.min_test_interactions must be at least 1.")
        if self.scoring_batch_size < 1:
            raise ConfigError("evaluation.scoring_batch_size must be at least 1.")


@dataclass(frozen=True)
class Module3Config:
    """Prediction and hidden-test evaluation settings (Plan.md Module 3).

    ``max_users = 0`` means every eligible user is scored, which is the
    production-like default. A positive value takes a deterministic sample and
    is useful for quick probes on a laptop.
    """

    evaluation_split: str = "test"
    max_users: int = 0
    # Ranking depth exported to parquet. It must cover the largest metric
    # cut-off so the exported lists and the reported metrics agree.
    export_top_k: int = 20
    # Number of scored users whose recommendation list is written to disk.
    # 0 exports every scored user (about 1.4M rows for this dataset).
    export_users: int = 5000
    compute_diversity: bool = True
    # Intra-list diversity is O(K^2) per user; this caps the runtime.
    diversity_users: int = 5000

    def __post_init__(self) -> None:
        if self.evaluation_split not in {"val", "test"}:
            raise ConfigError(
                "module3.evaluation_split must be 'val' or 'test', got "
                f"{self.evaluation_split!r}."
            )
        if self.max_users < 0:
            raise ConfigError("module3.max_users must be 0 (all users) or positive.")
        if self.export_top_k < 1:
            raise ConfigError("module3.export_top_k must be at least 1.")
        if self.export_users < 0:
            raise ConfigError("module3.export_users must be 0 (all) or positive.")
        if self.diversity_users < 1:
            raise ConfigError("module3.diversity_users must be at least 1.")


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration object shared by every pipeline stage.

    Module 1 reads only the preprocessing sections (``paths`` .. ``fusion``);
    Module 2 additionally reads ``profiles``, ``benchmark``, ``tuning`` and
    ``evaluation``; Module 3 adds the ``module3`` section. Keeping one validated
    root object means a single config file describes the whole experiment and
    the fingerprint covers every stage.
    """

    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    text: TextConfig = field(default_factory=TextConfig)
    price: PriceConfig = field(default_factory=PriceConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    profiles: ProfilesConfig = field(default_factory=ProfilesConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    module3: Module3Config = field(default_factory=Module3Config)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed must be a non-negative integer.")
        if self.price.column not in self.data.required_product_columns:
            raise ConfigError(
                f"price.column {self.price.column!r} is not part of "
                "data.required_product_columns."
            )


# Backwards-compatible alias: Module 1 consumes the same root object but only
# reads the preprocessing sections.
Module1Config = PipelineConfig


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _parse_iso(value: str):
    """Parse an ISO-8601 string into a UTC pandas timestamp (lazy import)."""
    import pandas as pd  # imported lazily so config validation stays lightweight

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:  # pragma: no cover - defensive
        raise ConfigError(f"Invalid ISO timestamp {value!r}: {error}") from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _build_section(cls: type, payload: Mapping[str, Any], location: str):
    """Instantiate a frozen dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{location or 'config'} must be a mapping, got {type(payload).__name__}.")

    hints = get_type_hints(cls)
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) under {location or 'config'}: {unknown}.")

    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in payload:
            continue
        path = f"{location}.{item.name}" if location else item.name
        kwargs[item.name] = _coerce(payload[item.name], hints[item.name], path)
    return cls(**kwargs)


def _coerce(value: Any, hint: Any, location: str) -> Any:
    """Convert a raw YAML value to the type declared by a dataclass annotation."""
    origin = get_origin(hint)
    args = get_args(hint)

    # Nested configuration section.
    if is_dataclass(hint):
        return _build_section(hint, value, location)

    # Optional[...] / X | None (PEP 604 unions).
    if origin in (Union, types.UnionType):
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{location} must not be null.")
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _coerce(value, candidate, location)
            except (ConfigError, TypeError, ValueError):
                continue
        raise ConfigError(f"{location} does not match any allowed type {args}.")

    # Fixed-length or homogeneous tuples.
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{location} must be a list or tuple.")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0], location) for item in value)
        if len(value) != len(args):
            raise ConfigError(f"{location} must contain exactly {len(args)} values.")
        return tuple(_coerce(item, arg, location) for item, arg in zip(value, args))

    # Mappings (e.g. field -> token prefix).
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{location} must be a mapping.")
        key_type, value_type = args if len(args) == 2 else (str, str)
        return {
            _coerce(key, key_type, location): _coerce(item, value_type, location)
            for key, item in value.items()
        }

    # Scalars. Booleans are rejected for numeric fields on purpose: YAML has
    # no int/bool ambiguity, but a stray ``true`` must not become ``1``.
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{location} must be a boolean.")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{location} must be an integer.")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{location} must be a number.")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{location} must be a string.")
        return value

    raise ConfigError(f"{location} uses an unsupported configuration type {hint!r}.")


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` without mutating either."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """Load, merge and validate the pipeline configuration.

    Args:
        path: YAML file to read. Defaults to ``model/config/config.yaml``. When
            the file does not exist the dataclass defaults are used, which keeps
            unit tests and synthetic smoke runs independent of the repository
            layout.
        overrides: Nested mapping merged on top of the YAML content. Intended for
            programmatic experiments (``{"split": {"train_ratio": 0.8}}``).

    Returns:
        A fully validated, immutable ``PipelineConfig``.

    Raises:
        ConfigError: If the YAML is unreadable, contains unknown keys, or holds
            values that violate a cross-field rule.
    """
    raw: dict[str, Any] = {}
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ConfigError(f"Cannot parse {config_path}: {error}") from error
        if loaded is not None and not isinstance(loaded, Mapping):
            raise ConfigError(f"{config_path} must contain a mapping at the top level.")
        raw = dict(loaded or {})
    elif path is not None:
        raise ConfigError(f"Configuration file not found: {config_path}")

    if overrides:
        raw = _deep_merge(raw, overrides)

    return _build_section(PipelineConfig, raw, "")


def _jsonify(value: Any) -> Any:
    """Recursively convert dataclasses and tuples into JSON-safe structures."""
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Return the effective configuration as a plain, JSON-serialisable dict."""
    return _jsonify(config)


def config_hash(config: PipelineConfig) -> str:
    """Return a stable SHA-256 fingerprint of the effective configuration.

    The hash is persisted with the artifacts so a matrix can always be traced
    back to the exact configuration that produced it.
    """
    payload = json.dumps(config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
