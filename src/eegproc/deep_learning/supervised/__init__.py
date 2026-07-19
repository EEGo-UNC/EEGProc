"""Supervised deep-learning components for EEGProc."""

from .cmhfe_dan import (
	CMHFEConfig,
	CMHFEFeatureExtractor,
	CMHFEModel,
	CMHFEDANNModel,
	CNN1DFeatureExtractor,
	DomainClassifier,
	EmotionHead,
	GradientReversalLayer,
	TransformerEncoder,
	binary_labels_to_one_hot,
	binarize_ratings,
	build_cmhfe_dann_model,
	build_cmhfe_model,
)

__all__ = [
	"CMHFEConfig",
	"CMHFEFeatureExtractor",
	"CMHFEModel",
	"CMHFEDANNModel",
	"CNN1DFeatureExtractor",
	"DomainClassifier",
	"EmotionHead",
	"GradientReversalLayer",
	"TransformerEncoder",
	"binary_labels_to_one_hot",
	"binarize_ratings",
	"build_cmhfe_dann_model",
	"build_cmhfe_model",
]
