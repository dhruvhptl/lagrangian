# Lagrangian Regime Network — Design Spec

**Date:** 2026-06-08
**Status:** Approved for implementation

---

## 1. Goal

Build `LagrangianRegimeNet` — a discrete Lagrangian-inspired latent dynamics model with diagonal mass and damping — as the core research contribution of the LLRN project. Compare its walk-forward macro F1, Brier score, and ECE against XGBoost, LSTM, GRU, and Neural ODE baselines on the same folds.

**Class name:** `LagrangianRegimeNet` throughout (model class, config references, test parametrize, entrypoint imports).

The hypothesis: structuring the latent update with a learned diagonal mass matrix and a scalar potential (rather than an unconstrained MLP vector field) produces smoother, better-calibrated regime forecasts.

---

## 2. What This Is Not

This is **not** a full general Euler–Lagrange simulator. It is a discrete, structured latent integrator whose update equations are *inspired by* Lagrangian mechanics — specifically the symplectic Euler discretization of a diagonal-mass system with damping. The physics analogy motivates the parameterization; it does not constrain the model to conserve energy or obey any physical law.

---

## 3. Architecture

### 3.1 Input

`x: (batch, window_len, n_features)` — same 3D tensor as LSTM/GRU/NODE.

### 3.2 Encoder

```
flatten: (batch, window_len * n_features)
→ Linear(window_len * n_features, hidden_dim) → ReLU
→ Linear(hidden_dim, hidden_dim) → ReLU
→ split into two heads:
    z_0    = Linear(hidden_dim, latent_dim)   # initial position
    z_dot_0 = Linear(hidden_dim, latent_dim)  # initial velocity
```

Both `z_0` and `z_dot_0` are shape `(batch, latent_dim)`.

### 3.3 Lagrangian Components

**Diagonal mass matrix `M(z)`:**
```
MassNet: Linear(latent_dim, latent_dim) → Softplus + epsilon
```
Output: `(batch, latent_dim)` — the diagonal of `M`. Always positive by construction.
Initialization: weight near zero, bias = `softplus_inverse(1.0) ≈ 0.541` so `M ≈ 1` at init (near-identity).
Epsilon: `1e-4` added after softplus for numerical stability in `M⁻¹`.

**Scalar potential `V(z)`:**
```
PotentialNet: Linear(latent_dim, hidden_dim) → Tanh → Linear(hidden_dim, 1)
```
Output: scalar per batch element. Both layers initialized near zero (small weight init) so `V ≈ 0` at the start of training.

**Damping:**
```python
gamma = F.softplus(self.raw_gamma)  # always positive
```
`raw_gamma` is a learnable scalar parameter, initialized so `softplus(raw_gamma) ≈ damping` from config (e.g., `raw_gamma = softplus_inverse(0.1) ≈ -2.25`).

### 3.4 Discrete Symplectic Euler Integrator

For `step = 0, 1, ..., n_steps - 1`:

```python
# 1. Compute mass diagonal and potential gradient via autograd
# z must have requires_grad=True for autograd.grad to work
M_diag = mass_net(z)                          # (batch, latent_dim), positive
dV_dz  = autograd.grad(V(z).sum(), z, create_graph=True)[0]  # (batch, latent_dim)
gamma  = F.softplus(raw_gamma)                 # scalar

# 2. Compute acceleration
z_ddot = -(dV_dz + gamma * z_dot) / (M_diag + eps)   # (batch, latent_dim)

# 3. Symplectic Euler update (velocity first, then position)
z_dot_new = z_dot + dt * z_ddot
z_new     = z     + dt * z_dot_new

# 4. Store trajectory
trajectory.append(z_new)   # list of (batch, latent_dim) tensors

z, z_dot = z_new, z_dot_new
```

`dt = 1.0` (fixed, from config). `n_steps` from config (default 4).

The trajectory list stores all `n_steps` positions. The **final position `z_T = trajectory[-1]`** is passed to the classifier head.

**Graph handling note:** `autograd.grad(..., create_graph=True)` retains the computation graph for backprop through the potential gradient. To avoid accidental graph retention across batches, the loop must not cache any intermediate tensors outside the step — all intermediate values (`M_diag`, `dV_dz`, `z_ddot`) are local to each iteration and go out of scope. `self.last_trajectory` stores detached tensors (`.detach()`) so trajectory storage does not prevent garbage collection of the training graph after `loss.backward()`.

### 3.5 Exogenous Forcing (disabled by default)

When `use_forcing=True`, a linear projection of the last input timestep's features is added to `z_ddot` before the velocity update:

```python
forcing = forcing_proj(x[:, -1, :])   # Linear(n_features, latent_dim)
z_ddot  = z_ddot + forcing
```

Default: `use_forcing=False`. Enabled via config only.

### 3.6 Classifier Head

```
LayerNorm(latent_dim) → Linear(latent_dim, 4)
```

Raw logits `(batch, 4)`. Same pattern as LSTM/GRU/NODE.

---

## 4. Config Dataclass

```python
@dataclass
class LagrangianConfig:
    input_dim: int = 37        # n_features (set from data at runtime)
    window_len: int = 40       # set from data config at runtime
    latent_dim: int = 8
    hidden_dim: int = 64
    n_steps: int = 4
    damping: float = 0.1       # initial value; raw_gamma = softplus_inverse(damping)
    dt: float = 1.0
    use_forcing: bool = False
    eps: float = 1e-4          # mass inverse numerical stability
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 150
    patience: int = 15
    device: str = "cpu"
```

---

## 5. Training

- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam, `lr` from config
- **Early stopping:** patience on val loss, best-weight restoration via state dict clone
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()`, before `optimizer.step()`
- **Seed:** `torch.manual_seed(cfg.seed)` before model construction each fold
- **Device:** `model.to(device)` before optimizer construction

Training loop structure mirrors `train_node.py / _train_fold` exactly, with gradient clipping added.

---

## 6. Trajectory Storage

`LagrangianRegimeNet.forward()` returns **raw logits** `(batch, 4)` for training (standard interface).

For diagnostics, `LagrangianRegimeNet` stores `self.last_trajectory: list[torch.Tensor]` — a list of **detached** `(batch, latent_dim)` tensors, one per integration step. It is **overwritten on every forward pass** and is diagnostic-only state: it is not part of the stable training interface, not included in `state_dict()`, and must not be relied upon across calls. `predict_proba` and `predict` call `forward()` normally and do not access `last_trajectory`.

---

## 7. Interface Parity

`LagrangianRegimeNet` exposes the same public interface as `RegimeLSTM`, `RegimeGRU`, `RegimeNODE`:

| Method | Signature | Notes |
|--------|-----------|-------|
| `forward(x)` | `(batch, T, F) → (batch, 4)` | raw logits |
| `predict_proba(X)` | `np.ndarray → np.ndarray (N, 4)` | calls `self.eval()` |
| `predict(X)` | `np.ndarray → np.ndarray (N,)` | argmax of proba |
| `fit(...)` | raises `NotImplementedError` | directs to `train_lagrangian` |

---

## 8. New Files

| File | Action | Purpose |
|------|--------|---------|
| `src/models/lagrangian_regime_net.py` | Create | `LagrangianConfig`, `LagrangianRegimeNet` model |
| `src/training/train_lagrangian.py` | Create | Hydra walk-forward entrypoint |
| `configs/model/lagrangian.yaml` | Create | Hydra config for Lagrangian model |
| `tests/test_shapes.py` | Modify | Append shape + behavior tests for `LagrangianRegimeNet` |

---

## 9. Tests

Shape and behavior tests to append to `tests/test_shapes.py`:

1. `test_lagrangian_forward_output_shape` — `(batch, 4)` for batch sizes 1 and 8
2. `test_lagrangian_predict_shape` — `(N,)` 
3. `test_lagrangian_predict_proba_shape` — `(N, 4)`
4. `test_lagrangian_proba_sums_to_one` — softmax sums to 1.0, atol=1e-5
5. `test_lagrangian_predict_in_range` — predictions in {0,1,2,3}
6. `test_lagrangian_predict_proba_switches_to_eval` — eval mode after `predict_proba`
7. `test_lagrangian_trajectory_length` — `last_trajectory` has `n_steps` elements after forward pass
8. `test_lagrangian_trajectory_shape` — each trajectory element is `(batch, latent_dim)`
9. `test_lagrangian_mass_positive` — `M_diag` output is positive for random input
10. `test_lagrangian_damping_positive` — `softplus(raw_gamma) > 0` at init
11. `test_lagrangian_forward_finite` — forward pass on random input produces finite logits (`torch.isfinite(logits).all()`)

---

## 10. Comparison Protocol

All 4 baselines (XGBoost, LSTM, GRU, NODE) and the Lagrangian model must be run on the **same data range** (`start_date: 2004-11-01`, `end_date: 2026-06-08`) with the **same default splits config** before drawing any conclusions. LSTM and GRU are being re-run on the full dataset concurrently.

Primary comparison metric: **mean macro F1 across all walk-forward folds**.
Secondary metrics: mean Brier score, mean ECE, regime switch frequency.

---

## 11. Non-Goals for MVP

- Continuous (torchdiffeq) integration of the Lagrangian
- Full mass matrix (non-diagonal)
- Exogenous forcing enabled by default
- Latent trajectory visualization (UMAP/PCA) — post-MVP
- Energy conservation analysis — post-MVP
