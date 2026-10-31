"""Registry of the three progressive centroid models (Plan.md Module 2.2).

Each model is described declaratively. The differences are intentionally small
and cumulative so the benchmark answers a single question per step:

* **Model 1** - unweighted centroid over *distinct* interacted items.
* **Model 2** - adds event-type weights and the repeat-count factor.
* **Model 3** - adds exponential time decay on top of Model 2.

No behavioural code lives here; ``user_profiles.build_profiles`` interprets the
flags. This keeps the comparison auditable: a reviewer can read the three rows
below and know exactly what changed between runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from model.core.logger import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one centroid model."""

    key: str
    label: str
    description: str
    use_event_weights: bool
    use_counts: bool
    use_decay: bool


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="unweighted",
        label="Model 1 - Unweighted Centroid",
        description=(
            "Mean of the distinct training item vectors. Ignores event type, "
            "repeat count and recency."
        ),
        use_event_weights=False,
        use_counts=False,
        use_decay=False,
    ),
    ModelSpec(
        key="weighted",
        label="Model 2 - Interaction-Weighted Centroid",
        description=(
            "Event-type weights (view < cart < purchase) combined with the "
            "repeat-count factor; still time-agnostic."
        ),
        use_event_weights=True,
        use_counts=True,
        use_decay=False,
    ),
    ModelSpec(
        key="decayed",
        label="Model 3 - Time-Decayed Weighted Centroid",
        description=(
            "Model 2 multiplied by exponential decay based on event_ts; recent "
            "interactions dominate the profile."
        ),
        use_event_weights=True,
        use_counts=True,
        use_decay=True,
    ),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in MODEL_SPECS}


def get_model_spec(key: str) -> ModelSpec:
    """Return the model registered under ``key``.

    Args:
        key: One of ``unweighted``, ``weighted`` or ``decayed``.

    Returns:
        The matching ``ModelSpec``.

    Raises:
        KeyError: If the key is unknown, with the valid keys listed.
    """
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {key!r}; expected one of {sorted(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[key]


def describe_models() -> list[dict[str, object]]:
    """Return a JSON-ready description of every model (used in reports)."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "use_event_weights": spec.use_event_weights,
            "use_counts": spec.use_counts,
            "use_decay": spec.use_decay,
        }
        for spec in MODEL_SPECS
    ]
