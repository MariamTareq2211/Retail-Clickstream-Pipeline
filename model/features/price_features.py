"""Price imputation, skew correction and scaling (Plan.md 1.4, 1.5, 1.7).

Order of operations matters and is deliberate:

1. **Impute** with the median of prices observed on *train-linked* products. The
   median is robust to the 10.3% genuine high-end outliers; using the mean would
   shift every missing price towards expensive electronics.
2. **log1p** to neutralise the 3.64 skew before any distance is computed. Without
   it, price would dominate the cosine similarity and cluster the catalogue into
   "cheap" and "expensive".
3. **RobustScaler** (median / IQR) so the single price dimension is comparable in
   magnitude to a unit-norm TF-IDF block and is not stretched by the long tail.

Every statistic is fitted on the training slice only. Training-linked products
are the fitting population because only they can influence the user profiles
built in Module 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from model.config.config_schema import Module1Config
from model.core.logger import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class PriceFeatureArtifacts:
    """Everything needed to reproduce the price transformation on new data."""

    impute_value: float
    log_transform: bool
    scaler: RobustScaler
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable parameters (the scaler itself is persisted with joblib)."""
        return {
            "impute_value": self.impute_value,
            "log_transform": self.log_transform,
            "scaler_center": float(self.scaler.center_[0]) if self.scaler.center_ is not None else None,
            "scaler_scale": float(self.scaler.scale_[0]) if self.scaler.scale_ is not None else None,
            **self.stats,
        }


def _apply_log(values: np.ndarray, log_transform: bool) -> np.ndarray:
    """Apply ``log1p`` when enabled, otherwise return the input unchanged."""
    return np.log1p(values) if log_transform else values


def fit_price_features(train_prices: pd.Series, config: Module1Config) -> PriceFeatureArtifacts:
    """Fit the imputation value and the price scaler on training products.

    Args:
        train_prices: Raw prices of products that appear in the training split.
        config: Validated Module 1 configuration.

    Returns:
        Fitted ``PriceFeatureArtifacts``.
    """
    values = train_prices.astype("float64")
    observed = values.dropna()

    if config.price.impute_strategy == "median" and not observed.empty:
        impute_value = float(observed.median())
    else:
        impute_value = float(config.price.constant_value)

    filled = values.fillna(impute_value)
    clipped = int((filled < 0).sum()) if config.price.clip_negative else 0
    if config.price.clip_negative:
        filled = filled.clip(lower=0.0)

    transformed = _apply_log(filled.to_numpy(), config.price.log_transform)
    scaler = RobustScaler(with_centering=config.price.scaler_centering).fit(
        transformed.reshape(-1, 1)
    )

    stats: dict[str, Any] = {
        "train_products": int(len(values)),
        "train_missing_prices": int(values.isna().sum()),
        "negative_values_clipped": clipped,
        "scaler_centering": bool(config.price.scaler_centering),
        "raw_min": float(observed.min()) if not observed.empty else None,
        "raw_median": float(observed.median()) if not observed.empty else None,
        "raw_max": float(observed.max()) if not observed.empty else None,
    }

    _LOGGER.info(
        "Price transformer fitted on %s training products | impute=%.4f | "
        "log1p=%s | missing=%s",
        f"{stats['train_products']:,}",
        impute_value,
        config.price.log_transform,
        stats["train_missing_prices"],
    )
    return PriceFeatureArtifacts(
        impute_value=impute_value,
        log_transform=config.price.log_transform,
        scaler=scaler,
        stats=stats,
    )


def transform_prices(prices: pd.Series, artifacts: PriceFeatureArtifacts, config: Module1Config) -> np.ndarray:
    """Apply the fitted price transformation to any price series.

    Args:
        prices: Raw prices to transform (the whole catalogue, or a single column).
        artifacts: Output of :func:`fit_price_features`.
        config: Validated Module 1 configuration.

    Returns:
        A one-dimensional array of scaled prices aligned with ``prices``.
    """
    filled = prices.astype("float64").fillna(artifacts.impute_value)
    if config.price.clip_negative:
        filled = filled.clip(lower=0.0)

    transformed = _apply_log(filled.to_numpy(), artifacts.log_transform)
    scaled = artifacts.scaler.transform(transformed.reshape(-1, 1)).ravel()

    _LOGGER.debug(
        "Scaled price range: %.4f -> %.4f (mean %.4f)",
        float(scaled.min()),
        float(scaled.max()),
        float(scaled.mean()),
    )
    return scaled
