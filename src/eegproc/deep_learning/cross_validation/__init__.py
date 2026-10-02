"""Cross-validation for subject-wise EEG experiments.

Split out of the former single-module ``cross_val`` in v2. Modules separate array helpers, metrics, fold execution, and orchestration.

Requires ``eegproc[deep-learning]``.
"""

from __future__ import annotations

from .callbacks import (
    EvaluationLevelValidationMetrics,
    HeldOutUserOracleMetrics,
    TrialValidationMetrics,
)
from .calibration import subject_calibration_cv
from .dataframe import cross_validate_dataframe
from .loso import fixed_loso_cv, loso_cv
from .nested import nested_lnso_cv

__all__ = [
    "cross_validate_dataframe",
    "loso_cv",
    "fixed_loso_cv",
    "subject_calibration_cv",
    "nested_lnso_cv",
    "TrialValidationMetrics",
    "EvaluationLevelValidationMetrics",
    "HeldOutUserOracleMetrics",
]
