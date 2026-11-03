"""CPU-only physiological diagnostics and grading of archived EEG trials."""

from .physiology import PhysiologicalReference, signal_diagnostics

__all__ = ["PhysiologicalReference", "signal_diagnostics"]
