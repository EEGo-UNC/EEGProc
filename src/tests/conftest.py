# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
from .utils import *

FS = 128

@pytest.fixture
def sine_df(request):
    fs = 128
    dur = 8.0  # 8 seconds for 1024 samples
    t = np.arange(0, int(dur * fs)) / fs
    freq = float(request.param)
    x = np.sin(2 * np.pi * freq * t)
    df = pd.DataFrame({"A1": x})
    return df



@pytest.fixture
def df_single_alpha():
    """Single column named A1_alpha containing a pure 10 Hz sine."""
    x = make_sine(10.0)
    return pd.DataFrame({"A1_alpha": x})

@pytest.fixture
def df_theta_noise():
    return pd.DataFrame({"A1_theta": make_white_noise(seed=42)})

@pytest.fixture
def df_mismatch_alpha_as_theta():
    """Column is named A1_theta but contains a 10 Hz sine (should integrate ~0 in theta band)."""
    x = make_sine(10.0)
    return pd.DataFrame({"A1_theta": x})

@pytest.fixture
def df_all_bands():
    """
    One column per band; each column contains a sine that sits
    comfortably inside its band's pass range.
    """
    demo = {
        "delta": 2.0,
        "theta": 6.0,
        "alpha": 10.0,
        "betaL": 15.0,
        "betaH": 25.0,
        "gamma": 35.0,
    }
    data = {}
    for band, f in demo.items():
        data[f"A1_{band}"] = make_sine(f)
    return pd.DataFrame(data)

@pytest.fixture
def df_multi_freq_sines_same_amp():
    return pd.DataFrame({ # All channels should have same amplitude
        "A1": make_sine(2.0),
        "A2": make_sine(10.0),
        "A3": make_sine(35.0),
    })

@pytest.fixture
def df_simple_sinusoid():
    return pd.DataFrame({"A1": make_sine(10.0)})
