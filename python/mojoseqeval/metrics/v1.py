"""Strict tagging-scheme API corresponding to ``seqeval.metrics.v1``."""

from __future__ import annotations

from ..scheme import Token, auto_detect
from .sequence_labeling import (
    _report,
    _scores_from_statistics,
    _statistics,
    check_consistent_length,
)


def unique_labels(y_true, y_pred, scheme, suffix=False):
    return _statistics(y_true, y_pred, suffix=suffix, scheme=scheme)[0]


def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    beta=1.0,
    sample_weight=None,
    zero_division="warn",
    scheme=None,
    suffix=False,
    **kwargs,
):
    del sample_weight, kwargs
    stats = _statistics(y_true, y_pred, suffix=suffix, scheme=scheme)
    return _scores_from_statistics(
        stats,
        average=average,
        warn_for=warn_for,
        beta=beta,
        zero_division=zero_division,
    )


def classification_report(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    digits=2,
    output_dict=False,
    zero_division="warn",
    suffix=False,
    scheme=None,
):
    del sample_weight
    check_consistent_length(y_true, y_pred)
    if scheme is None or not issubclass(scheme, Token):
        scheme = auto_detect(y_true, suffix)
    stats = _statistics(y_true, y_pred, suffix=suffix, scheme=scheme)
    return _report(stats, digits, output_dict, zero_division)
