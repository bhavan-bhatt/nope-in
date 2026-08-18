"""Evaluation metrics and explainability for NOPE-IN."""

from nope_in.evaluation.explainability import (
    compute_shap_values,
    plot_event_attribution,
    plot_global_feature_importance,
    plot_regime_conditional_importance,
)
from nope_in.evaluation.metrics import (
    build_eval_metadata,
    compute_iv_metrics,
    compute_pricing_metrics,
    plot_pricing_comparison,
)

__all__ = [
    "build_eval_metadata",
    "compute_iv_metrics",
    "compute_pricing_metrics",
    "compute_shap_values",
    "plot_event_attribution",
    "plot_global_feature_importance",
    "plot_pricing_comparison",
    "plot_regime_conditional_importance",
]
