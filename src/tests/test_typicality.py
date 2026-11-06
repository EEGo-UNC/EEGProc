"""Regression tests for restored physiological diagnostics and graded checks."""

import json

import numpy as np
import pytest


def test_compact_physiology_preserves_components_and_undefined_values():
    from eegproc.model_explainability.typicality.physiology import (
        FAMILIES, PhysiologicalReference, diagnostic_components, signal_diagnostics,
    )
    signal = np.random.default_rng(31).normal(size=(4, 128, 42))
    diagnostics = signal_diagnostics(signal, fs=128, n_channels=14)
    reference = PhysiologicalReference.fit([diagnostics, diagnostics])
    diagnostics['debiased_wpli_squared'][0, 0] = -0.12345678901234567
    diagnostics['coherence'][0, 0] = np.nan
    diagnostics['coherence'][0, 1] = np.inf
    metadata = dict(channel_names=[f'C{i}' for i in range(14)],
                    band_names=['theta', 'alpha', 'beta'], signal_unit='uV')
    payload = diagnostic_components(diagnostics, **metadata)
    summary = reference.assess(diagnostics, **metadata)
    assert summary['components'] == payload
    saved = json.loads(json.dumps(payload, allow_nan=False))
    for family in FAMILIES:
        np.testing.assert_array_equal(np.asarray(saved[family], dtype=float),
                                      np.where(np.isfinite(diagnostics[family]), diagnostics[family], np.nan))
    assert saved['debiased_wpli_squared'][0][0] == -0.12345678901234567
    assert saved['coherence'][0][:2] == [None, None]
    assert saved['aperiodic_exponent'] == [None] * 14
    assert saved['pair_indices'][:2] == [[0, 1], [0, 2]]
    assert saved['pair_indices'][-1] == [12, 13]
    assert saved['signal_unit'] == 'uV'
    assert not {'psd', 'frequencies_hz', 'rms'} & saved.keys()


def test_compact_physiology_retains_custom_band_edges():
    from eegproc.model_explainability.typicality.physiology import diagnostic_components, signal_diagnostics

    edges = ((2., 6.), (6., 12.), (12., 25.))
    diagnostics = signal_diagnostics(np.random.default_rng(7).normal(size=(3, 128, 6)),
                                     fs=128, n_channels=2, band_edges=edges)
    components = diagnostic_components(diagnostics)
    assert components['band_edges_hz'] == [list(pair) for pair in edges]
    assert components['pair_indices'] == [[0, 1]]
    assert components['channel_names'] is None
    assert components['signal_unit'] is None


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
