"""Exponential time decay of interaction weights (Plan.md Module 2.2, Model 3).

The log spans seven days, so the decay rate is expressed **per hour**. A rate of
``ln(2) / 24`` roughly 0.0289 gives a 24-hour half-life: an interaction one day
before the reference instant keeps 50% of its weight, one week before keeps
0.8%. Expressing the rate per day (the intuitive default for long logs) would
make every event within this window effectively weightless.

Two reference policies are supported:

``global``
    Age is measured against one instant (the end of the training window) for
    every user. This is what production does: the model is refreshed at time
    ``T`` and the whole population is aged against ``T``.
``per_user``
    Age is measured against each user's own most recent training interaction.
    This rewards recency *relative to the user*, which is useful when sessions
    are sparse, but it is not reproducible at serving time for a returning user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.core.logger import get_logger

_LOGGER = get_logger(__name__)

_NS_PER_HOUR = 3.6e12


def to_utc_ns(series: pd.Series) -> np.ndarray:
    """Convert a timezone-aware timestamp series to int64 UTC nanoseconds.

    Integer arithmetic avoids repeated ``Timestamp`` comparisons on millions of
    rows and keeps decay deterministic across pandas versions.
    """
    values = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return values.to_numpy(dtype="datetime64[ns]").astype("int64")


def resolve_reference_ns(
    timestamps_ns: np.ndarray,
    user_keys: np.ndarray,
    mode: str,
    global_reference_ns: int,
) -> int | np.ndarray:
    """Return the reference instant(s) used to age each interaction.

    Args:
        timestamps_ns: int64 UTC nanoseconds, one per interaction.
        user_keys: User identifier aligned with ``timestamps_ns``.
        mode: ``"global"`` or ``"per_user"``.
        global_reference_ns: Instant used when ``mode == "global"``.

    Returns:
        A scalar for the global policy, or an array aligned with the
        interactions for the per-user policy.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode == "global":
        return int(global_reference_ns)
    if mode == "per_user":
        frame = pd.DataFrame({"user_key": user_keys, "ts": timestamps_ns})
        return frame.groupby("user_key", sort=False)["ts"].transform("max").to_numpy()
    raise ValueError(f"Unknown decay reference {mode!r}; expected 'global' or 'per_user'.")


def decay_weights(
    timestamps_ns: np.ndarray,
    reference_ns: int | np.ndarray,
    rate_per_hour: float,
) -> np.ndarray:
    """Compute ``exp(-lambda * age_hours)`` for every interaction.

    Args:
        timestamps_ns: int64 UTC nanoseconds, one per interaction.
        reference_ns: Scalar or aligned array of reference instants.
        rate_per_hour: Decay rate lambda per hour (must be positive).

    Returns:
        Weights in ``(0, 1]``; future interactions (age below zero) are clipped
        to a weight of one instead of being amplified.

    Raises:
        ValueError: If the rate is not strictly positive.
    """
    if rate_per_hour <= 0:
        raise ValueError(f"Decay rate must be positive, got {rate_per_hour}.")

    age_hours = (np.asarray(reference_ns, dtype=np.float64) - timestamps_ns.astype(np.float64)) / _NS_PER_HOUR
    age_hours = np.clip(age_hours, 0.0, None)
    return np.exp(-rate_per_hour * age_hours)


def describe_decay(rate_per_hour: float) -> dict[str, float]:
    """Summarise the decay effect for the stage report.

    Args:
        rate_per_hour: Decay rate lambda per hour.

    Returns:
        JSON-ready diagnostics: half-life and the weight after 1 h / 24 h / 168 h.
    """
    half_life_hours = float(np.log(2.0) / rate_per_hour)
    return {
        "rate_per_hour": float(rate_per_hour),
        "half_life_hours": round(half_life_hours, 4),
        "weight_after_1h": round(float(np.exp(-rate_per_hour * 1)), 6),
        "weight_after_24h": round(float(np.exp(-rate_per_hour * 24)), 6),
        "weight_after_168h": round(float(np.exp(-rate_per_hour * 168)), 6),
    }


def log_reference(reference_ns: int | np.ndarray) -> None:
    """Log the reference instant(s) in a human-readable form (debug only)."""
    if np.isscalar(reference_ns):
        _LOGGER.debug(
            "Decay reference (global): %s",
            pd.Timestamp(int(reference_ns), unit="ns", tz="UTC"),
        )
    else:
        values = np.asarray(reference_ns)
        _LOGGER.debug(
            "Decay reference (per user): %s users, latest %s",
            f"{len(values):,}",
            pd.Timestamp(int(values.max()), unit="ns", tz="UTC"),
        )
