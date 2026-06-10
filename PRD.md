# Product Requirements Document: Lagrangian Regime Modeling Platform

## Overview

This product is a research-to-production machine learning platform for market regime classification built around a Lagrangian-inspired latent dynamics model, benchmarked against XGBoost, LSTM, GRU, and Neural ODE baselines.[1][2][3] The current best result uses a causal 1D-convolutional encoder feeding a discrete Lagrangian latent integrator, achieving mean macro F1 of 0.3976 over 71 walk-forward folds, compared with 0.4062 for LSTM, 0.4008 for GRU, 0.3850 for Neural ODE, 0.3700 for the earlier MLP-encoder Lagrangian, and 0.4208 for XGBoost.[2][3]

The platform should support repeatable experimentation across datasets and systems, modular encoder swaps, feature-set extensions, walk-forward evaluation, and deployment-oriented reporting of classification and calibration metrics.[4][5][6] It should make it easy to reproduce the current architecture, test improved variants, and port the same workflow to new data sources or environments without rewriting the training logic.[2][3][7]

## Product goals

The primary goal is to provide a modular system for developing and benchmarking structured latent-dynamics models for market regime prediction, with a specific focus on treating latent representations as generalized coordinates evolved by discrete Lagrangian-inspired dynamics.[1][8][9] A second goal is to combine this structured dynamics approach with stronger temporal encoders, domain-informed features, and optional multi-horizon supervision to close or surpass the remaining performance gap to LSTM and XGBoost baselines.[2][3][4][10]

Success means the system can: (1) reproduce baseline models on the same data and folds, (2) train Lagrangian variants with plug-in encoders and feature sets, (3) evaluate all models under identical walk-forward settings, and (4) export consistent benchmark tables and per-fold diagnostics for research and deployment review.[6][5][4]

## Users and use cases

Primary users are ML researchers, quant researchers, AI engineers, and software engineers who want a controlled framework for comparing structured sequence models with standard tabular and recurrent baselines on regime prediction tasks.[11][6] Typical use cases include researching new latent-dynamics architectures, testing econophysics-informed feature pipelines, evaluating model calibration for risk-aware workflows, and porting the same stack to new financial instruments or macro datasets.[12][13][14][4]

Secondary users include hiring managers, collaborators, or reviewers who need interpretable benchmark artifacts, clear documentation, and evidence that the system is reproducible and useful beyond a single experiment.[6][5] For these users, the platform should generate concise comparison tables, fold-level metrics, and architecture descriptions that explain why a given variant outperforms or underperforms alternatives.[2][3]

## Scope

### In scope

- Walk-forward training and evaluation over fixed folds with shared data ranges and split logic across all models.[6]
- Baseline model support for XGBoost, LSTM, GRU, Neural ODE, and Lagrangian variants.[6]
- Modular encoder support for MLP, causal Conv1D, TCN, and hybrid convolutional encoders.[2][3][15]
- Lagrangian latent integration with mass, potential, damping, and optional forcing modules.[1][16][8]
- Feature pipeline support for standard tabular features and optional econophysics/statistical-mechanics-inspired features such as volatility clustering and collective-behavior proxies.[12][13][11][10]
- Optional multi-horizon regime targets and multi-head losses for experimentation.[4][17][18]
- Benchmark reporting including macro F1, Brier score, ECE, per-fold deltas, and plots.[19][20][21]

### Out of scope

- High-frequency trading execution systems or live order routing.[14]
- Quantum neural network implementations or Schrödinger/PINN-based finance models.[22][23]
- Production brokerage connectivity or portfolio optimization engines.
- Automated feature discovery beyond the defined experimental framework.

## Problem statement

The original Lagrangian regime model underperformed LSTM and GRU because its flatten-plus-MLP encoder discarded temporal structure in the 40-step input window, forcing the latent integrator to compensate for weak sequence encoding.[2][3] Replacing that encoder with a causal Conv1D encoder improved mean macro F1 from 0.3700 to 0.3976 over 71 folds, demonstrating that time-aware encoding is a first-order design requirement for this class of model.[2][3]

The remaining gap to the best baselines suggests that future gains are likely to come from a combination of less constrained latent dynamics, stronger temporal feature extraction, richer domain features, and carefully chosen supervision rather than from training longer or simply widening the latent space.[16][4][10][6] The platform must therefore support clean ablations where encoder choice, feature sets, and horizon design can be changed independently and compared under identical evaluation conditions.[5][6]

## Product principles

- Preserve fairness: all models must use the same data windowing, label definitions, and walk-forward folds unless a specific experiment explicitly changes one variable.[6]
- Preserve modularity: encoders, feature sets, labelers, trainers, and evaluators should be swappable without breaking the public training interface.[5][6]
- Preserve causality: no encoder, feature computation, or label alignment may introduce future leakage.[2][3][24]
- Optimize for research speed: small, auditable modules and config-driven experiments are preferred over heavy framework abstraction.[25][26]
- Support deployability: outputs must include calibration metrics, fold-level diagnostics, and versioned experiment artifacts, not just top-line F1.[20][27][28]

## Functional requirements

### Data and features

The platform must ingest time-indexed market data and produce rolling windows of shape `(batch, window_len, n_features)` suitable for all model families.[6] It must support a base feature set and an optional econophysics feature set that includes volatility clustering proxies, tail-risk descriptors, and cross-sectional collective-behavior statistics when the underlying data supports them.[12][13][11][10]

Feature engineering must be deterministic, index-safe, and controlled by config flags, including a toggle such as `use_econophysics_features`.[10][12] The pipeline should document whether cross-asset breadth or market-mode features are available in a given environment and degrade gracefully when only single-index inputs exist.[11][14]

### Labels

The system must support the current four-class composite taxonomy based on return and volatility state, with exactly one label per sample and no overlap across classes.[29][30] It must also support optional multi-horizon labels, such as 5-day, 10-day, and 20-day regime targets, generated with the same taxonomy and no leakage.[4][17][18]

### Models

The platform must support at least these model families:

| Model family | Requirement |
|---|---|
| XGBoost | Tabular baseline using engineered features.[6] |
| LSTM | Sequential baseline with recurrent hidden state.[31][32] |
| GRU | Sequential baseline with recurrent gating.[31][32] |
| Neural ODE | Continuous-depth latent dynamics baseline.[33][34] |
| Lagrangian | Discrete latent dynamics model with pluggable encoders.[1][8] |

For the Lagrangian family, the platform must support encoder variants `mlp`, `conv1d`, `tcn`, and `hybrid_conv`, all returning a shared encoder embedding before projection to `z_0` and `z_dot_0`.[2][3][15] The latent integrator must support diagonal mass, expressive potential networks, scalar or vector damping, optional coordinate transforms, and optional exogenous forcing.[16][1][8]

### Training

The training system must support single-horizon and optional multi-horizon objectives, with configurable horizon weights and early stopping on validation loss.[21][35] It must preserve a shared walk-forward loop for fair comparison across models and support subset runs before full 71-fold sweeps.[6]

The default benchmark path should prioritize the best-known architecture for accuracy and reproducibility. Based on current evidence, that default Lagrangian path should be the causal Conv1D encoder with single-horizon loss unless later experiments prove a better configuration.[2][3]

### Evaluation and reporting

The platform must report at minimum:

- Mean macro F1 across folds.[19][36]
- Standard deviation of fold-level F1.[19]
- Mean Brier score.[20]
- Mean ECE.[27][28]
- Per-fold metrics and deltas versus selected baselines.[6]

It should also support optional plots of fold-wise F1 trajectories and export benchmark summaries to CSV for downstream analysis.[6] Comparison reports must clearly identify whether a gain comes from encoder changes, feature changes, or target/loss changes.[5][6]

## Non-functional requirements

### Reproducibility

All experiments must be reproducible through configuration, fixed seeds, versioned data ranges, and saved fold-level outputs.[25][26] The system must allow a new environment to recreate the same benchmark table given the same raw data and config set.[6]

### Maintainability

Data, feature engineering, labels, models, training, and evaluation should remain separated into clean modules with minimal cross-dependencies.[37][38][39] PyTorch dependencies should remain outside pure data/feature/label modules so these pieces stay testable and portable.[37][38]

### Performance

The system should support subset fold runs for rapid iteration and full 71-fold runs for final evaluation.[6] Encoders should be benchmarked not only for F1 but also for parameter count and training cost, since the Conv1D encoder achieved higher F1 than the MLP while using fewer parameters.[2][3]

## Architecture requirements

### Reference architecture

The recommended reference architecture for the current production-research line is:

1. Input windows `(B, T, F)`.[2][3]
2. Causal Conv1D encoder over time, preserving temporal order.[2][3]
3. Shared embedding `h` mapped to `z_0` and `z_dot_0`.[2][3]
4. Discrete Lagrangian-inspired latent integration with mass, potential, damping, and symplectic updates.[1][16][8]
5. Single classifier head for the primary regime horizon.[21]

### Experimental extensions

The platform must allow testing of these extensions without rewriting the stack:

- Vector damping in the latent dynamics.[16]
- Deeper potential networks and richer mass networks.[1][16]
- Econophysics-informed exogenous forcing.[12][13][10]
- Multi-horizon regime heads and multi-task losses.[4][17][18]
- Calibration post-processing such as temperature scaling if desired for deployment.[20][27]

## Success metrics

Primary success metric is mean macro F1 over the full 71-fold benchmark.[19][6] Secondary success metrics are mean Brier score and mean ECE, since regime models used in downstream decision systems need both discriminative power and calibrated probabilities.[20][27][28]

Suggested performance targets for the next release:

| Target | Metric |
|---|---|
| Match or exceed GRU | Mean macro F1 >= 0.401.[2][3] |
| Approach or exceed LSTM | Mean macro F1 >= 0.406.[2][3] |
| Remain competitive on calibration | Mean ECE near 0.147 and Brier near LSTM/GRU levels.[2][3] |
| Maintain reproducibility | 71-fold rerun with consistent aggregate metrics.[6] |

## Testing requirements

The platform must include unit tests for:

- Encoder output shapes across all supported encoder types.[2][3]
- Finite logits and absence of NaNs.[40]
- Correct causal convolution output length and no future leakage in padding logic.[2][3]
- Positive mass outputs and positive damping where constrained by softplus.[41][42]
- Correct label shapes and class ranges across supported horizons.[4][24]
- Feature pipeline integrity, including no all-NaN columns and correct alignment.[10]

Integration tests should verify subset fold runs, benchmark aggregation, and CSV export of fold-level results.[6]

## Rollout plan

### Phase 1

Stabilize the current best architecture: causal Conv1D encoder plus the existing Lagrangian integrator with single-horizon training.[2][3] This phase becomes the new baseline for all future Lagrangian experiments.

### Phase 2

Add econophysics-informed features behind a config flag and test them first with the Conv1D single-horizon model to isolate their effect.[12][13][10] Avoid coupling these features with multi-horizon loss until the single-horizon effect is understood, since the first combined attempt degraded performance substantially in subset testing.[4][21]

### Phase 3

Reintroduce multi-horizon supervision only after either increasing encoder capacity or demonstrating that the Conv1D encoder can absorb the richer objective without losing 5-day F1.[21][43][35] Treat multi-horizon as an experimental extension, not a default requirement.

### Phase 4

If needed, test additional encoder families such as TCN or hybrid convolutional stacks, but only if they show improvement on subset runs over the Conv1D baseline.[2][3][15]

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Flat or regressive performance from combined upgrades | Wasted compute and unclear conclusions | Change one axis at a time: encoder, features, or loss.[5][6] |
| Hidden leakage in time-aware features or labels | Invalid benchmark | Enforce causal indexing and dedicated leakage tests.[24][2] |
| Overfitting through high-capacity multi-task heads | Lower primary-horizon F1 | Keep single-horizon as default and gate multi-horizon behind config.[21][35] |
| Poor portability across systems | Hard to reproduce results elsewhere | Use config-driven paths, modular data interfaces, and environment-independent benchmark outputs.[25][26] |

## Open questions

- Does Conv1D plus econophysics features, with single-horizon loss, outperform the current Conv1D baseline over both subset and full runs?[12][10]
- Can a higher-capacity temporal encoder absorb multi-horizon supervision without sacrificing primary-horizon F1?[21][43]
- Why does XGBoost remain the strongest full-run baseline, and which feature characteristics explain its advantage?[6]
- Which periods or fold types favor structured latent dynamics over tabular baselines, and can those patterns guide future feature or architecture design?[6]

## Acceptance criteria

The PRD is satisfied when a new environment can:

1. Reproduce the baseline benchmark table with the existing 71-fold setup.[6]
2. Train the Conv1D-encoder Lagrangian model and recover performance in the neighborhood of mean macro F1 0.3976 over 71 folds on the same data and splits.[2][3]
3. Run ablations for encoder type, feature flags, and horizon settings through config only.[25][26]
4. Export benchmark tables and per-fold metrics for comparison against XGBoost, LSTM, GRU, NODE, and prior Lagrangian variants.[6]
