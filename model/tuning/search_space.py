"""Optuna search space and trial-scoped item matrices (Plan.md Module 2.3).

Which parameters are sampled depends on the winning model, so the study never
wastes trials on knobs that cannot change the objective:

* every model tunes the TF-IDF block (``max_features``, ``ngram_range``,
  ``min_df``) and the fusion price weight;
* Models 2-3 additionally tune view and cart weights;
* Model 3 additionally tunes the hourly decay rate;
* Models 2-3 additionally tune the repeat-count transform.

The purchase weight stays fixed at 1.0. Cosine similarity is invariant to a
global rescaling of the user vector, so only the view:purchase and
cart:purchase ratios are identifiable; letting Optuna sample all three would
create a flat direction that slows the search without changing any ranking.

Re-fitting TF-IDF per trial is the expensive part, so the text matrix is cached
by its three hyperparameters. The price block is reused unchanged: its scaler
was fitted once on training products in Module 1 and must not move.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp

from model.config.config_schema import FusionConfig, PipelineConfig, TfidfConfig
from model.core.logger import get_logger
from model.features.item_matrix import fuse_features
from model.features.text_features import fit_transform_text
from model.models.base import ModelSpec

_LOGGER = get_logger(__name__)

# Number of distinct TF-IDF matrices kept in memory. A trial never needs more
# than one at a time and TPE revisits good regions, so a small cache captures
# most of the benefit without risking hundreds of megabytes.
_TEXT_CACHE_SIZE = 6


@dataclass(frozen=True)
class TrialParams:
    """A fully resolved set of hyperparameters for one Optuna trial."""

    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    price_weight: float
    decay_rate_per_hour: float
    count_transform: str
    event_weights: Mapping[int, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation (integer mapping keys become strings)."""
        return {
            "max_features": int(self.max_features),
            "ngram_range": [int(self.ngram_range[0]), int(self.ngram_range[1])],
            "min_df": int(self.min_df),
            "price_weight": round(float(self.price_weight), 6),
            "decay_rate_per_hour": round(float(self.decay_rate_per_hour), 8),
            "count_transform": self.count_transform,
            "event_weights": {
                str(int(key)): round(float(value), 6)
                for key, value in sorted(self.event_weights.items())
            },
        }


def _ngram_label(ngram_range: tuple[int, int]) -> str:
    """Encode an n-gram range as an Optuna-friendly categorical string."""
    return f"{int(ngram_range[0])}-{int(ngram_range[1])}"


def _parse_ngram_label(label: str) -> tuple[int, int]:
    """Decode ``"1-2"`` back into ``(1, 2)``."""
    low, high = label.split("-", maxsplit=1)
    return int(low), int(high)


def suggest_trial_params(
    trial: "optuna.Trial",
    model: ModelSpec,
    config: PipelineConfig,
) -> TrialParams:
    """Sample one point of the search space for the given model.

    Args:
        trial: Active Optuna trial.
        model: Winning model whose profile formula defines the relevant knobs.
        config: Validated pipeline configuration.

    Returns:
        A ``TrialParams`` instance.
    """
    space = config.tuning.search_space

    max_features = int(
        trial.suggest_categorical("max_features", [int(v) for v in space.max_features])
    )
    ngram_range = _parse_ngram_label(
        trial.suggest_categorical(
            "ngram_range", [_ngram_label(value) for value in space.ngram_ranges]
        )
    )
    min_df = int(trial.suggest_categorical("min_df", [int(v) for v in space.min_df]))

    price_weight = float(
        trial.suggest_float(
            "price_weight", float(space.price_weight[0]), float(space.price_weight[1]), log=True
        )
    )

    event_weights: dict[int, float] = dict(config.profiles.weights)
    if model.use_event_weights:
        event_weights[1] = float(
            trial.suggest_float(
                "view_weight", float(space.view_weight[0]), float(space.view_weight[1]), log=True
            )
        )
        event_weights[2] = float(
            trial.suggest_float(
                "cart_weight", float(space.cart_weight[0]), float(space.cart_weight[1]), log=True
            )
        )
        # Anchored at 1.0 so the ratios remain identifiable (see module docstring).
        event_weights[4] = 1.0

    if model.use_decay:
        decay_rate = float(
            trial.suggest_float(
                "decay_rate_per_hour",
                float(space.decay_rate[0]),
                float(space.decay_rate[1]),
                log=True,
            )
        )
    else:
        decay_rate = float(config.profiles.decay.rate_per_hour)

    if model.use_counts:
        count_transform = str(
            trial.suggest_categorical("count_transform", list(space.count_transforms))
        )
    else:
        count_transform = config.dedup.count_transform

    return TrialParams(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        price_weight=price_weight,
        decay_rate_per_hour=decay_rate,
        count_transform=count_transform,
        event_weights=event_weights,
    )


class ItemMatrixFactory:
    """Rebuild the catalogue matrix for trial-specific TF-IDF parameters.

    Args:
        products: Product catalogue (used only through ``metadata_text`` and
            ``train_mask``).
        metadata_text: Documents produced by Module 1's
            :func:`build_metadata_text`; the same text for every trial.
        train_mask: Boolean mask of products with training interactions. The
            vectorizer may only ever see these documents.
        price_vector: Scaled price column produced by the Module 1 transformer.
        config: Validated pipeline configuration.
    """

    def __init__(
        self,
        metadata_text: pd.Series,
        train_mask: pd.Series,
        price_vector: np.ndarray,
        config: PipelineConfig,
    ) -> None:
        self._metadata_text = metadata_text.reset_index(drop=True)
        self._train_mask = train_mask.reset_index(drop=True)
        self._price_vector = np.asarray(price_vector, dtype=np.float64)
        self._config = config
        self._text_cache: "OrderedDict[tuple[int, tuple[int, int], int], sp.csr_matrix]" = OrderedDict()
        self._matrix_cache: "OrderedDict[tuple[Any, ...], sp.csr_matrix]" = OrderedDict()

    def _text_matrix(
        self,
        max_features: int,
        ngram_range: tuple[int, int],
        min_df: int,
    ) -> sp.csr_matrix:
        """Return (and cache) the TF-IDF block for one hyperparameter set."""
        key = (int(max_features), (int(ngram_range[0]), int(ngram_range[1])), int(min_df))
        if key in self._text_cache:
            self._text_cache.move_to_end(key)
            return self._text_cache[key]

        trial_config = replace(
            self._config,
            text=replace(
                self._config.text,
                tfidf=TfidfConfig(
                    max_features=int(max_features),
                    min_df=int(min_df),
                    ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
                    sublinear_tf=self._config.text.tfidf.sublinear_tf,
                ),
            ),
        )
        artifacts = fit_transform_text(self._metadata_text, self._train_mask, trial_config)

        self._text_cache[key] = artifacts.matrix
        if len(self._text_cache) > _TEXT_CACHE_SIZE:
            self._text_cache.popitem(last=False)
        return artifacts.matrix

    def build(self, params: TrialParams) -> sp.csr_matrix:
        """Return the fused, normalised item matrix for one trial.

        Args:
            params: Trial hyperparameters.

        Returns:
            CSR catalogue matrix with one row per product.
        """
        text_matrix = self._text_matrix(params.max_features, params.ngram_range, params.min_df)

        key = (
            params.max_features,
            params.ngram_range,
            params.min_df,
            round(float(params.price_weight), 6),
        )
        if key in self._matrix_cache:
            self._matrix_cache.move_to_end(key)
            return self._matrix_cache[key]

        fusion_config = replace(
            self._config,
            fusion=FusionConfig(
                price_weight=float(params.price_weight),
                normalize_rows=self._config.fusion.normalize_rows,
            ),
        )
        artifacts = fuse_features(text_matrix, self._price_vector, fusion_config)

        self._matrix_cache[key] = artifacts.matrix
        if len(self._matrix_cache) > _TEXT_CACHE_SIZE:
            self._matrix_cache.popitem(last=False)
        return artifacts.matrix
