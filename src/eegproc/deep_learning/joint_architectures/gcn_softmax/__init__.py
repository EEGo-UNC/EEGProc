"""GCN-only softmax baseline for EEGProc."""

from .model import build_gcn_softmax_classifier

__all__ = ["build_gcn_softmax_classifier"]
