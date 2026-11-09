from importlib.metadata import PackageNotFoundError, version as _package_version

from .preprocessing import (
    apply_detrend,
    numeric_interp,
    sosfiltfilt_safe,
    apply_notch_once,
    bandpass_filter,
    FREQUENCY_BANDS,
)
from .artifact_removal import remove_artifacts_with_iclabel, remove_eye_artifacts
from .featurization import (
    hjorth_params,
    psd_bandpowers,
    wavelet_band_energy,
    wavelet_entropy,
    shannons_entropy,
    imf_band_energy,
    imf_entropy,
    dwt_subband_ranges,
    choose_dwt_level,
    feature_grouped_by_metadata,
)

try:
    __version__ = _package_version("eegproc")
except PackageNotFoundError:  # a source tree that has not been installed
    __version__ = "unknown"

__all__ = [
    #PREPROCESSING
    "apply_detrend",
    "numeric_interp",
    "sosfiltfilt_safe",
    "apply_notch_once",
    "bandpass_filter",
    "FREQUENCY_BANDS",

    #ARTIFACT REMOVAL
    "remove_artifacts_with_iclabel",
    "remove_eye_artifacts",

    #FEATURIZATION
    "hjorth_params",
    "psd_bandpowers",
    "shannons_entropy",
    "wavelet_band_energy",
    "wavelet_entropy",
    "imf_band_energy",
    "imf_entropy",
    "dwt_subband_ranges",
    "choose_dwt_level",
    "feature_grouped_by_metadata"
]
