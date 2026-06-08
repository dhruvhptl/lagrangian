import numpy as np
import pytest
from src.models.baseline_xgb import RegimeXGB, XGBConfig


@pytest.fixture
def xgb_cfg():
    return XGBConfig(n_estimators=10, max_depth=3, seed=42, n_jobs=1)


@pytest.fixture
def toy_flat_data():
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 200, 50, 20
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_train = rng.integers(0, 4, n_train)
    X_val = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    y_val = rng.integers(0, 4, n_val)
    return X_train, y_train, X_val, y_val


def test_regime_xgb_predict_shape(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_regime_xgb_predict_proba_shape(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_regime_xgb_proba_sums_to_one(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_regime_xgb_predict_in_range(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert set(preds).issubset({0, 1, 2, 3})


def test_regime_xgb_feature_importances(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    fi = model.feature_importances()
    assert len(fi) == X_train.shape[1]


import torch
from src.models.baseline_lstm import RegimeLSTM, RegimeGRU, RNNConfig


@pytest.fixture
def rnn_cfg():
    return RNNConfig(
        input_dim=37,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        seed=42,
    )


@pytest.fixture
def toy_seq_data():
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    seq_len, n_feat = 40, 37
    X_train = rng.standard_normal((n_train, seq_len, n_feat)).astype(np.float32)
    y_train = rng.integers(0, 4, n_train)
    X_val = rng.standard_normal((n_val, seq_len, n_feat)).astype(np.float32)
    y_val = rng.integers(0, 4, n_val)
    return X_train, y_train, X_val, y_val


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_forward_output_shape(rnn_cfg, ModelClass):
    model = ModelClass(rnn_cfg)
    x = torch.randn(8, 40, 37)
    out = model(x)
    assert out.shape == (8, 4), f"Expected (8, 4), got {out.shape}"


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_shape(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_proba_shape(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_proba_sums_to_one(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_in_range(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})
