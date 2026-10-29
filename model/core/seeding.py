"""Reproducibility helpers.

Only the random number generators used by *this* pipeline need seeding today
(Optuna adds its own sampler seed in Module 2). ``PYTHONHASHSEED`` is set as a
best-effort convenience: it only takes effect in child processes, so a warning
is emitted when the current interpreter was started with a different value.
"""

from __future__ import annotations

import os
import random

import numpy as np

from model.core.logger import get_logger

_LOGGER = get_logger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline uses.

    Args:
        seed: Non-negative integer seed taken from the configuration.

    Notes:
        NumPy's legacy global RNG is seeded (not a ``Generator`` instance) because
        scikit-learn utilities still draw from it in a few code paths. The value
        is reduced modulo 2**32 to satisfy NumPy's accepted range.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    random.seed(seed)
    np.random.seed(seed % (2**32))

    current = os.environ.get("PYTHONHASHSEED")
    if current is None:
        os.environ["PYTHONHASHSEED"] = str(seed)
    elif current != str(seed):
        _LOGGER.warning(
            "PYTHONHASHSEED=%s does not match seed=%s; set it in the shell before "
            "starting Python for full hash-order reproducibility.",
            current,
            seed,
        )

    _LOGGER.debug("Global seed set to %s.", seed)
