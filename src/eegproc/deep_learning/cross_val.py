"""Deprecated alias for :mod:`eegproc.deep_learning.cross_validation`.

The single 7,555-line module was split into a package in v2. This shim keeps
``from eegproc.deep_learning.cross_val import loso_cv`` working; it re-exports the
public surface only. Private helpers moved and are not re-exported.
"""

from __future__ import annotations

import warnings

from .cross_validation import *  # noqa: F401,F403
from .cross_validation import __all__  # noqa: F401

warnings.warn(
    "eegproc.deep_learning.cross_val is deprecated and will be removed in "
    "eegproc 3.0; import from eegproc.deep_learning.cross_validation instead.",
    DeprecationWarning,
    stacklevel=2,
)
