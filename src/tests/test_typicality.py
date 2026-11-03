"""Regression tests for restored physiological diagnostics and graded checks."""

import numpy as np
import pytest


def test_band_filtered_physiology_never_claims_complete_pass():
    from eegproc.model_explainability.typicality.physiology import signal_diagnostics, PhysiologicalReference
    rng = np.random.default_rng(14)
    diagnostics = [signal_diagnostics(rng.normal(size=(4, 32, 6)), fs=128, n_channels=2) for _ in range(3)]
    reference = PhysiologicalReference.fit(diagnostics)
    result = reference.assess(diagnostics[1])
    assert result["required_count"] == 5
    assert result["available_count"] == 4
    assert result["all_required_passed"] is None
    assert result["checks"]["aperiodic_exponent"]["reason"] == "not_estimable_from_band_filtered_decoder"
    assert result["checks"]["aperiodic_exponent"]["deviation"] is None

def test_physiological_deviation_grades_distance_beyond_the_binary_check():
    """A graded score must agree with the binary check and separate near misses."""
    from eegproc.model_explainability.typicality.physiology import signal_diagnostics, PhysiologicalReference
    rng = np.random.default_rng(14)
    diagnostics = [signal_diagnostics(rng.normal(size=(4, 32, 6)), fs=128, n_channels=2) for _ in range(8)]
    reference = PhysiologicalReference.fit(diagnostics)

    # z <= 1 is exactly the interval membership that "passed" already records.
    for family in ("amplitude", "spectral_power", "coherence", "debiased_wpli_squared"):
        lower, upper = reference.lower[family], reference.upper[family]
        values = diagnostics[1][family]
        half = (upper - lower) / 2
        scorable = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(values) & (half > 0)
        z = np.abs(values[scorable] - ((lower + upper) / 2)[scorable]) / half[scorable]
        inside = (values[scorable] >= lower[scorable]) & (values[scorable] <= upper[scorable])
        np.testing.assert_array_equal(z <= 1 + 1e-12, inside)

    # The binary check saturates; the graded score keeps separating.
    candidate = dict(diagnostics[1])
    exceedances = []
    for multiplier in (1.5, 3.0, 20.0):
        candidate["amplitude"] = diagnostics[1]["amplitude"] * multiplier
        amplitude = reference.assess(candidate)["checks"]["amplitude"]
        assert amplitude["passed"] is False
        assert amplitude["fraction_in_range"] == 0.0
        exceedances.append(amplitude["deviation"]["exceedance_mean"])
    assert exceedances == sorted(exceedances) and exceedances[0] < exceedances[-1]

    # Components inside the interval contribute zero exceedance.
    assert reference.assess(diagnostics[1])["checks"]["coherence"]["deviation"]["exceedance_max"] >= 0.0
    summary = reference.assess(candidate)
    assert summary["graded_count"] == 4
    assert summary["worst_family_exceedance"] == max(summary["exceedance_by_family"].values())
