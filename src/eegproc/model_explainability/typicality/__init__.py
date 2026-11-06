"""CPU-only physiological diagnostics and grading of archived EEG trials."""

from .physiology import PhysiologicalReference, diagnostic_components, signal_diagnostics

__all__ = ["PhysiologicalReference", "diagnostic_components", "signal_diagnostics"]
