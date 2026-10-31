"""Profile builders for the three centroid models.

This layer exists so the benchmark, the tuner and the final export all construct
profiles through exactly the same call. If Model 3 is selected, the exported
matrix is therefore guaranteed to be the same object the benchmark measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.config.config_schema import PipelineConfig
from model.core.logger import get_logger
from model.models.base import MODEL_SPECS, ModelSpec
from model.profiles.interaction_weights import ProfileParams
from model.profiles.user_profiles import ProfileArtifacts, build_profiles

_LOGGER = get_logger(__name__)


def build_profiles_for_model(
    events: pd.DataFrame,
    item_matrix: sp.spmatrix,
    item_index: pd.DataFrame,
    model: ModelSpec,
    config: PipelineConfig,
    params: ProfileParams | None = None,
    user_keys: np.ndarray | None = None,
    global_reference_ns: int | None = None,
) -> ProfileArtifacts:
    """Build profiles for one model, defaulting to the configured weights.

    Args:
        events: Training interactions.
        item_matrix: Module 1 catalogue matrix.
        item_index: ``product_id`` -> ``item_row`` mapping.
        model: Model description.
        config: Validated pipeline configuration.
        params: Explicit profile parameters (Optuna trials pass their own).
        user_keys: Optional user row order / sample.
        global_reference_ns: Optional decay reference instant.

    Returns:
        ``ProfileArtifacts`` for the requested model.
    """
    resolved = params if params is not None else ProfileParams.from_config(config)
    return build_profiles(
        events,
        item_matrix,
        item_index,
        model,
        resolved,
        config,
        user_keys=user_keys,
        global_reference_ns=global_reference_ns,
    )


def build_all_profiles(
    events: pd.DataFrame,
    item_matrix: sp.spmatrix,
    item_index: pd.DataFrame,
    config: PipelineConfig,
    user_keys: np.ndarray,
    global_reference_ns: int | None = None,
) -> dict[str, ProfileArtifacts]:
    """Build the three models on the same users for a like-for-like comparison.

    Args:
        events: Training interactions restricted to ``user_keys``.
        item_matrix: Module 1 catalogue matrix.
        item_index: ``product_id`` -> ``item_row`` mapping.
        config: Validated pipeline configuration.
        user_keys: Users to build profiles for.
        global_reference_ns: Decay reference instant shared by all models.

    Returns:
        Mapping ``model key -> ProfileArtifacts``.
    """
    params = ProfileParams.from_config(config)
    profiles: dict[str, ProfileArtifacts] = {}

    for model in MODEL_SPECS:
        profiles[model.key] = build_profiles_for_model(
            events,
            item_matrix,
            item_index,
            model,
            config,
            params=params,
            user_keys=user_keys,
            global_reference_ns=global_reference_ns,
        )

    return profiles
