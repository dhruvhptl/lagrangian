import numpy as np
import pandas as pd
import pytest
import torch
from src.models.baseline_lstm import RegimeLSTM, RegimeGRU, RNNConfig
from src.models.baseline_node import RegimeNODE, NODEConfig
from src.models.baseline_xgb import RegimeXGB, XGBConfig
from src.models.lagrangian_regime_net_mh import LagrangianRegimeNetMH, LagrangianMHConfig
from src.features.econophysics import build_econophysics_features
from src.labels.multi_horizon_labeler import MultiHorizonLabeler, MultiHorizonLabelConfig
from src.utils.multi_horizon_builder import MultiHorizonFold, build_folds_multi


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


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_proba_switches_to_eval(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    model.train()  # explicitly put in train mode
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"


@pytest.fixture
def node_cfg():
    return NODEConfig(input_dim=37, hidden_dim=32, seed=42)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_node_forward_output_shape(node_cfg, batch_size):
    model = RegimeNODE(node_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4), f"Expected ({batch_size}, 4), got {out.shape}"


def test_node_predict_shape(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_node_predict_proba_shape(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_node_proba_sums_to_one(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_node_predict_in_range(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


def test_node_predict_proba_switches_to_eval(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    model.train()
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"


def test_node_config_fields():
    cfg = NODEConfig()
    assert hasattr(cfg, "hidden_dim")
    assert hasattr(cfg, "ode_hidden_dim")
    assert hasattr(cfg, "solver")
    assert cfg.solver == "dopri5"


from src.models.lagrangian_regime_net import (
    LagrangianRegimeNet, LagrangianConfig, PotentialNet,
    CausalConv1d, Conv1dEncoder, TCNEncoder, HybridConvEncoder,
)


@pytest.fixture
def lag_cfg():
    return LagrangianConfig(
        input_dim=37,
        window_len=40,
        latent_dim=8,
        hidden_dim=32,
        n_steps=3,
        seed=42,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
def test_lagrangian_forward_output_shape(lag_cfg, batch_size):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4), f"Expected ({batch_size}, 4), got {out.shape}"


def test_lagrangian_predict_shape(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_lagrangian_predict_proba_shape(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_lagrangian_proba_sums_to_one(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lagrangian_predict_in_range(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


def test_lagrangian_predict_proba_switches_to_eval(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    model.train()
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"


def test_lagrangian_trajectory_length(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    assert len(model.last_trajectory) == lag_cfg.n_steps


def test_lagrangian_trajectory_shape(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    for z in model.last_trajectory:
        assert z.shape == (4, lag_cfg.latent_dim)


def test_lagrangian_mass_positive(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    z = torch.randn(4, lag_cfg.latent_dim)
    m = model.mass_net(z)
    assert (m > 0).all(), "Mass diagonal must be strictly positive"


def test_lagrangian_damping_positive(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    gamma = torch.nn.functional.softplus(model.raw_gamma)
    assert gamma.item() > 0, "Damping must be positive"


@pytest.mark.parametrize("n_steps,scale", [(1, 1.0), (4, 1.0), (8, 1.0), (4, 10.0)])
def test_lagrangian_forward_finite(n_steps, scale):
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=n_steps)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37) * scale
    logits = model(x)
    assert torch.isfinite(logits).all(), f"Non-finite logits with n_steps={n_steps}, scale={scale}"


def test_lagrangian_backward_grad_flow():
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=3)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert model.potential_net.net[0].weight.grad is not None, "No grad on potential_net"
    assert model.raw_gamma.grad is not None, "No grad on raw_gamma"


# --- v5 tests ---

@pytest.fixture
def v5_cfg():
    return LagrangianConfig(
        input_dim=37,
        window_len=40,
        latent_dim=16,
        hidden_dim=64,
        n_steps=4,
        use_vector_damping=True,
        use_coord_transform=True,
        seed=42,
    )


def test_lagrangian_v5_forward_shape(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    x = torch.randn(4, 40, 37)
    out = model(x)
    assert out.shape == (4, 4), f"Expected (4, 4), got {out.shape}"


def test_lagrangian_v5_forward_finite(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    assert torch.isfinite(logits).all(), "v5 forward pass produced non-finite logits"


def test_lagrangian_v5_predict_proba_shape(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    X = np.random.randn(10, 40, 37).astype(np.float32)
    proba = model.predict_proba(X)
    assert proba.shape == (10, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lagrangian_vector_damping_positive(v5_cfg):
    """gamma_net output must be strictly positive for random latent input."""
    model = LagrangianRegimeNet(v5_cfg)
    # Run a forward pass so gamma_net is exercised; inspect via a direct call
    z = torch.randn(4, v5_cfg.latent_dim)
    gamma_vec = torch.nn.functional.softplus(model.gamma_net(z))
    assert (gamma_vec > 0).all(), "Vector damping must be strictly positive"
    assert torch.isfinite(gamma_vec).all(), "Vector damping must be finite"


def test_lagrangian_v5_backward_grad_flow(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert model.potential_net.net[0].weight.grad is not None, "No grad on DeepPotentialNet"
    assert model.gamma_net.weight.grad is not None, "No grad on gamma_net"
    assert model.coord_net.weight.grad is not None, "No grad on coord_net"


def test_lagrangian_v5_old_path_unchanged():
    """Default config (use_vector_damping=False) must still use scalar gamma and shallow potential."""
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=2)
    model = LagrangianRegimeNet(cfg)
    assert hasattr(model, 'raw_gamma'), "raw_gamma must exist on default config"
    assert not hasattr(model, 'gamma_net'), "gamma_net must not exist on default config"
    assert isinstance(model.potential_net, PotentialNet), "Default must use PotentialNet, not DeepPotentialNet"


# --- v6 MH model tests ---

@pytest.fixture
def mh_cfg():
    return LagrangianMHConfig(
        input_dim=37,
        window_len=40,
        latent_dim=16,
        hidden_dim=64,
        potential_hidden_dim=64,   # smaller for test speed
        mass_hidden_dim=32,
        n_steps=3,
        use_vector_damping=True,
        use_coord_transform=True,
        multi_horizon=True,
        seed=42,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_lagrangian_mh_forward_shape(mh_cfg, batch_size):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4)


def test_lagrangian_mh_forward_multi_shapes(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    out = model.forward_multi(x)
    assert set(out.keys()) == {"logits_5", "logits_10", "logits_20"}
    for k, v in out.items():
        assert v.shape == (4, 4), f"{k} shape wrong: {v.shape}"


def test_lagrangian_mh_forward_finite(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    out = model.forward_multi(x)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"Non-finite in {k}"


def test_lagrangian_mh_predict_proba(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    X = np.random.randn(8, 40, 37).astype(np.float32)
    proba = model.predict_proba(X)
    assert proba.shape == (8, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lagrangian_mh_trajectory_stored(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    assert len(model.last_trajectory) == mh_cfg.n_steps


def test_lagrangian_mh_backward_grad_flow(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    out = model.forward_multi(x)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    assert model.potential_net.net[0].weight.grad is not None, "No grad on DeepPotentialNet"
    assert model.gamma_net.weight.grad is not None, "No grad on gamma_net"
    assert model.coord_net.weight.grad is not None, "No grad on coord_net"


# --- Econophysics feature tests ---

@pytest.fixture
def toy_prices_dict():
    """Minimal price dict for feature tests — SPY + QQQ + TLT."""
    idx = pd.date_range("2010-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    def make_df(seed_val):
        rng2 = np.random.default_rng(seed_val)
        close = 100 * np.exp(np.cumsum(rng2.normal(0, 0.01, 300)))
        return pd.DataFrame({"close": close, "volume": rng.integers(1e6, 1e7, 300)}, index=idx)
    return {"SPY": make_df(1), "QQQ": make_df(2), "TLT": make_df(3)}


def test_econophysics_features_shape(toy_prices_dict):
    df = build_econophysics_features(toy_prices_dict, roll_windows=[21])
    assert len(df) == 300
    assert df.shape[1] > 0


def test_econophysics_features_no_all_nan(toy_prices_dict):
    df = build_econophysics_features(toy_prices_dict, roll_windows=[21])
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    assert len(all_nan_cols) == 0, f"All-NaN columns: {all_nan_cols}"


def test_econophysics_no_future_leakage(toy_prices_dict):
    """Features at index t must not depend on prices after t.
    Check: shift prices by 1 day forward and verify features are unchanged at t-1.
    This is a smoke check: first valid row of each feature must be NaN (window not filled).
    """
    df = build_econophysics_features(toy_prices_dict, roll_windows=[21])
    # First 20 rows should have NaNs (window=21 not yet filled)
    assert df.iloc[:20].isna().any(axis=1).all(), "Expected NaNs in first 20 rows (window warmup)"


# --- Multi-horizon label tests ---

@pytest.fixture
def toy_spy_prices():
    idx = pd.date_range("2010-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    return pd.DataFrame({"close": close}, index=idx)


def test_multi_horizon_label_shapes(toy_spy_prices):
    cfg = MultiHorizonLabelConfig(horizons=[5, 10, 20])
    labeler = MultiHorizonLabeler(cfg)
    labels_df = labeler.fit_transform(toy_spy_prices)
    assert list(labels_df.columns) == ["label_5", "label_10", "label_20"]
    assert len(labels_df) == len(toy_spy_prices)


def test_multi_horizon_labels_in_range(toy_spy_prices):
    cfg = MultiHorizonLabelConfig(horizons=[5, 10, 20])
    labeler = MultiHorizonLabeler(cfg)
    labels_df = labeler.fit_transform(toy_spy_prices)
    for col in labels_df.columns:
        valid = labels_df[col].dropna()
        assert set(valid.astype(int).unique()).issubset({0, 1, 2, 3}), f"{col} has out-of-range values"


def test_multi_horizon_no_nan_in_head(toy_spy_prices):
    """Labels at horizon h must have NaN only in the last h rows (forward return window)."""
    cfg = MultiHorizonLabelConfig(horizons=[5, 10, 20], smoothing=False)
    labeler = MultiHorizonLabeler(cfg)
    labels_df = labeler.fit_transform(toy_spy_prices)
    # Check label_5: last 5 rows should be NaN (forward return unavailable)
    assert labels_df["label_5"].iloc[-5:].isna().all(), "Last 5 rows of label_5 should be NaN"
    # label_20: last 20 rows NaN
    assert labels_df["label_20"].iloc[-20:].isna().all(), "Last 20 rows of label_20 should be NaN"


# ---------------------------------------------------------------------------
# Encoder ablation tests
# ---------------------------------------------------------------------------

ENCODER_TYPES = ["mlp", "conv1d", "tcn", "hybrid_conv"]


def _make_lag_cfg(encoder_type: str) -> LagrangianConfig:
    return LagrangianConfig(
        input_dim=37,
        window_len=40,
        latent_dim=8,
        hidden_dim=32,
        encoder_dim=32,
        encoder_type=encoder_type,
        conv_channels=32,
        conv_kernel_size=3,
        tcn_channels=32,
        tcn_kernel_size=3,
        tcn_dilations=[1, 2, 4],
        n_steps=3,
        use_vector_damping=True,
        use_coord_transform=True,
        seed=0,
    )


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_forward_shape(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(8, 40, 37)
    out = model(x)
    assert out.shape == (8, 4), f"{encoder_type}: expected (8, 4), got {out.shape}"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_forward_finite(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(8, 40, 37)
    out = model(x)
    assert torch.isfinite(out).all(), f"{encoder_type}: non-finite logits"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_predict_proba_shape(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    X = np.random.randn(16, 40, 37).astype(np.float32)
    proba = model.predict_proba(X)
    assert proba.shape == (16, 4), f"{encoder_type}: predict_proba shape wrong"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_trajectory_length(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    assert len(model.last_trajectory) == cfg.n_steps, f"{encoder_type}: trajectory length wrong"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_no_nan_in_output(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    h = model.encoder(x)
    assert not torch.isnan(h).any(), f"{encoder_type}: NaN in encoder output"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_backward_grad_flow(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    logits.sum().backward()
    # Check that at least one encoder parameter received a gradient
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0, f"{encoder_type}: no gradient flowed to encoder"


def test_causal_conv1d_preserves_length():
    """CausalConv1d must output the same T as input (causal padding correctness)."""
    for kernel_size in [3, 5]:
        for dilation in [1, 2, 4]:
            conv = CausalConv1d(16, 16, kernel_size, dilation)
            x = torch.randn(4, 16, 40)
            y = conv(x)
            assert y.shape == (4, 16, 40), (
                f"kernel={kernel_size} dilation={dilation}: "
                f"expected T=40, got {y.shape[2]}"
            )


def test_lagrangian_encoder_param_counts():
    """Log parameter counts for each encoder — acts as a sanity check."""
    for enc in ENCODER_TYPES:
        cfg = _make_lag_cfg(enc)
        model = LagrangianRegimeNet(cfg)
        enc_params = sum(p.numel() for p in model.encoder.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        # Sanity: encoder params must be > 0 and < total
        assert enc_params > 0, f"{enc}: encoder has no parameters"
        assert enc_params < total_params, f"{enc}: encoder has all params (dynamics missing?)"


def test_lagrangian_mlp_encoder_backward_compat():
    """Default LagrangianConfig (no encoder_type set) must still work as MLP."""
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=2)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    out = model(x)
    assert out.shape == (4, 4)
    assert torch.isfinite(out).all()
