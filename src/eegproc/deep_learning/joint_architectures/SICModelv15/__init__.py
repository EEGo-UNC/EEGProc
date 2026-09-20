"""SIC v15 with convex joint EEG reconstruction."""

from .sic_model import (
    ConvexReconstructionFusion,
    SICModel,
    build_sic_model,
)

__all__ = [
    "ConvexReconstructionFusion",
    "SICModel",
    "build_sic_model",
]
