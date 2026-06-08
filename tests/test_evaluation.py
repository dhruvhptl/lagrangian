# tests/test_evaluation.py
import numpy as np
import pytest
from src.evaluation.metrics import evaluate, evaluate_noisy, EvalResult


@pytest.fixture
def perfect_preds():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = y_true.copy()
    y_prob = np.eye(4)[y_true]
    return y_true, y_pred, y_prob


@pytest.fixture
def random_preds():
    rng = np.random.default_rng(42)
    y_true = np.array([0, 1, 2, 3] * 25)
    y_pred = rng.integers(0, 4, len(y_true))
    y_prob = rng.dirichlet(np.ones(4), size=len(y_true))
    return y_true, y_pred, y_prob


def test_evaluate_returns_evalresult(perfect_preds):
    y_true, y_pred, y_prob = perfect_preds
    result = evaluate(y_true, y_pred, y_prob)
    assert isinstance(result, EvalResult)


def test_evaluate_perfect_macro_f1(perfect_preds):
    y_true, y_pred, y_prob = perfect_preds
    result = evaluate(y_true, y_pred, y_prob)
    assert result.macro_f1 == pytest.approx(1.0)


def test_evaluate_perfect_brier_score(perfect_preds):
    y_true, y_pred, y_prob = perfect_preds
    result = evaluate(y_true, y_pred, y_prob)
    assert result.brier_score == pytest.approx(0.0)


def test_evaluate_all_fields_present(random_preds):
    y_true, y_pred, y_prob = random_preds
    result = evaluate(y_true, y_pred, y_prob)
    assert hasattr(result, "macro_f1")
    assert hasattr(result, "balanced_accuracy")
    assert hasattr(result, "confusion_matrix")
    assert hasattr(result, "brier_score")
    assert hasattr(result, "ece")
    assert hasattr(result, "switch_frequency")
    assert hasattr(result, "mean_entropy")


def test_evaluate_switch_frequency_range(random_preds):
    y_true, y_pred, y_prob = random_preds
    result = evaluate(y_true, y_pred, y_prob)
    assert 0.0 <= result.switch_frequency <= 1.0


def test_evaluate_entropy_nonnegative(random_preds):
    y_true, y_pred, y_prob = random_preds
    result = evaluate(y_true, y_pred, y_prob)
    assert result.mean_entropy >= 0.0


def test_evaluate_noisy_returns_evalresult(random_preds):
    y_true, _, y_prob = random_preds

    class DummyModel:
        def predict(self, X):
            return np.argmax(X[:, :4], axis=1)
        def predict_proba(self, X):
            p = np.abs(X[:, :4]) + 1e-8
            return p / p.sum(axis=1, keepdims=True)

    X_clean = y_prob.copy()
    noisy_result = evaluate_noisy(X_clean, DummyModel(), y_true, sigma=0.5)
    assert isinstance(noisy_result, EvalResult)
