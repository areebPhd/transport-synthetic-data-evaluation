"""Public API for the SynthEval reference implementation."""

from .aggregation import (
    AggregateMetricResult,
    RunObservation,
    aggregate_run_observations,
)
from .api import EvaluationReport
from .config import EvaluationConfig
from .evaluator import SynthEvaluator
from .schema import SchemaConfig

__all__ = [
    "AggregateMetricResult",
    "EvaluationConfig",
    "EvaluationReport",
    "RunObservation",
    "SchemaConfig",
    "SynthEvaluator",
    "aggregate_run_observations",
]
__version__ = "0.2.1"
