"""Evaluation metrics for regime forecasting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


@dataclass
class EvalResult:
    macro_f1: float
    balanced_accuracy: float
    confusion_matrix: np.ndarray
    brier_score: float
    ece: float
    switch_frequency: float
    mean_entropy: float


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_classes = y_prob.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (confidence-based)."""
    y_pred = np.argmax(y_prob, axis=1)
    confidence = y_prob.max(axis=1)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc - conf)
    return float(ece)


def _switch_frequency(y_pred: np.ndarray) -> float:
    if len(y_pred) < 2:
        return 0.0
    return float((y_pred[1:] != y_pred[:-1]).mean())


def _mean_entropy(y_prob: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(y_prob * np.log(y_prob + eps), axis=1).mean())


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> EvalResult:
    """Compute all regime forecasting metrics."""
    return EvalResult(
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        brier_score=_brier_score(y_true, y_prob),
        ece=_ece(y_true, y_prob),
        switch_frequency=_switch_frequency(y_pred),
        mean_entropy=_mean_entropy(y_prob),
    )


def evaluate_noisy(
    X: np.ndarray,
    model,
    y_true: np.ndarray,
    sigma: float = 0.1,
    seed: int = 42,
) -> EvalResult:
    """Evaluate model on Gaussian-noised inputs."""
    rng = np.random.default_rng(seed)
    X_noisy = X + rng.normal(0, sigma, size=X.shape)
    y_pred = model.predict(X_noisy)
    y_prob = model.predict_proba(X_noisy)
    return evaluate(y_true, y_pred, y_prob)
