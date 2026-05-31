# Reproducibility

This public repository includes the code path needed to reproduce the research pipeline with the user's own licensed WRDS access.

The public repository intentionally excludes raw WRDS/vendor data, row-level Parquet panels, credentials, model binaries, and cluster logs.

## Public script manifest

| Stage | Script | Purpose |
|---|---|---|
| release | `scripts/step09a_public_safety_audit.py` | Public safety audit: checks secrets, forbidden public files, and tracked forbidden files. |
| release | `scripts/step09b_generate_public_docs.py` | Generates README, docs, and public aggregate assets. |
| release | `scripts/step09c_prepare_public_push.py` | Stages only public-safe reproducibility files by allowlist. |
| trace-clean | `scripts/step04a_trace_metadata_audit.py` | Raw TRACE footer/metadata audit. |
| trace-clean | `scripts/step04d_trace_code_policy_audit.py` | TRACE code-policy audit. |
| trace-clean | `scripts/step04e_trace_policy_retention_audit.py` | Cleaning policy retention audit. |
| trace-clean | `scripts/step04f_trace_cleaner.py` | Builds clean TRACE universes. |
| trace-clean | `scripts/step04g_validate_clean_outputs.py` | Validates clean TRACE outputs. |
| trace-clean | `scripts/step04h_build_clean_trace_manifests.py` | Builds final clean TRACE manifests. |
| security-master | `scripts/step05a_security_master_local_qa.py` | Local non-TRACE/FISD/security-master QA. |
| security-master | `scripts/step05b_identifier_coverage_audit.py` | TRACE/FISD/return/link identifier coverage audit. |
| security-master | `scripts/step05c_build_fisd_security_master.py` | Builds local FISD issue/rating/amount dimensions. |
| security-master | `scripts/step05d_trace_fisd_join_audit.py` | Row-level TRACE-to-FISD join audit. |
| security-master | `scripts/step05e_trace_fisd_join_panel_smoke.py` | Smoke joined TRACE-FISD panel. |
| security-master | `scripts/step05f_build_trace_fisd_panel.py` | Builds stable TRACE-FISD processed panel. |
| security-master | `scripts/step05g_validate_trace_fisd_panel.py` | Validates stable TRACE-FISD panel. |
| security-master | `scripts/step05h_curve_ready_universe_audit.py` | Curve-ready issuer-date support audit. |
| curves | `scripts/step06a_curve_fit_smoke.py` | Nelson-Siegel curve-fit smoke. |
| curves | `scripts/step06b_curve_model_comparison_smoke.py` | Guarded curve-family model comparison smoke. |
| curves | `scripts/step06c_build_curve_inputs.py` | Builds issue-date curve input aggregates. |
| curves | `scripts/step06d_validate_curve_inputs.py` | Validates curve-input outputs. |
| curves | `scripts/step06e_curve_fit_residuals.py` | Fits guarded curves and residuals. |
| curves | `scripts/step06f_validate_curve_fit_outputs.py` | Validates curve-fit outputs. |
| features | `scripts/step06g_residual_feature_smoke.py` | Builds residual feature layer. |
| features | `scripts/step06h_validate_residual_features.py` | Validates residual feature outputs. |
| labels | `scripts/step07a_label_source_audit.py` | Audits return-label sources. |
| labels | `scripts/step07b_monthly_label_matrix_smoke.py` | Builds monthly label matrix. |
| labels | `scripts/step07c_validate_monthly_label_matrix.py` | Validates/promotes monthly label matrix. |
| models | `scripts/step08a_monthly_baseline_ridge.py` | Transparent ridge/signal baseline. |
| backtest | `scripts/step08b_monthly_signal_backtest.py` | Cost-aware residual-signal decile backtest. |
| models | `scripts/step08c_gpu_mlp_monthly.py` | GPU PyTorch MLP model. |
| models | `scripts/step08d_lightgbm_monthly.py` | CPU LightGBM tabular model. |
| results | `scripts/step08e_model_leaderboard_and_figures.py` | Builds model leaderboard and figures. |
| robustness | `scripts/step08f_signal_robustness_exposure.py` | Maturity/liquidity robustness and exposure audit. |
| robustness | `scripts/step08g_rating_amount_exposure_audit.py` | As-of rating and amount-outstanding exposure audit. |

## Rebuild outline

```bash
# 1. Configure licensed WRDS credentials locally; do not commit credentials.
# 2. Run extraction and local QA scripts on your own WRDS account.
# 3. Build clean TRACE, FISD joins, curve inputs, curves, residual features, labels, models, and robustness outputs.
# 4. Run the public safety audit before committing.
python scripts/step09a_public_safety_audit.py --workspace .
```

## Public-safe artifacts

Aggregate result tables and figures live under `docs/assets/`. They are public-safe summaries, not raw vendor data.
