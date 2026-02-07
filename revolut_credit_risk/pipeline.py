"""End-to-end pipeline orchestrator.

[Paper §2, Fig. 1] PD Model Development Lifecycle:
Data Collection -> Entity Set Creation -> Feature Generation ->
WoE Transformation -> IV Calculation -> Feature Selection ->
Rank-ordering Model Training -> PD Calibration.
"""
from __future__ import annotations

import logging
import time
import warnings

# Suppress sklearn deprecation warning triggered inside optbinning
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*renamed.*ensure_all_finite.*",
    category=FutureWarning,
)

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from revolut_credit_risk import config
from revolut_credit_risk.logging_config import setup_logging
from revolut_credit_risk.report import ReportWriter
from revolut_credit_risk.data.synthetic_data import (
    generate_synthetic_data, save_dataset, load_dataset,
)
from revolut_credit_risk.features.dfs_engine import run_dfs
from revolut_credit_risk.features.binning import bin_features, transform_woe, plot_binning_tables
from revolut_credit_risk.features.variable_config import get_variable_configs
from revolut_credit_risk.selection.information_value import (
    collect_iv, filter_by_iv, bivariate_analysis,
)
from revolut_credit_risk.selection.miv_selector import run_miv_selection
from revolut_credit_risk.model.scorecard import train_scorecard, benchmark_models
from revolut_credit_risk.model.calibration import calibrate_isotonic
from revolut_credit_risk.evaluation.metrics import (
    compute_all_splits, plot_roc_curve, plot_lorenz_curve,
    plot_calibration_curve, plot_miv_selection,
)
from revolut_credit_risk.monitoring.residual_monitor import run_residual_monitoring

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Run the full credit risk model development pipeline."""
    setup_logging()
    report = ReportWriter()
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Credit Risk Pipeline — START")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Data Generation / Loading  [Assumption]
    # ------------------------------------------------------------------
    logger.info("Phase 1: Data generation / loading")
    if config.GENERATE_NEW_DATA:
        data = generate_synthetic_data()
        save_dataset(data)
    else:
        data = load_dataset()

    report.add_section("1. Data Summary", _data_summary(data))

    # ------------------------------------------------------------------
    # Phase 2: Deep Feature Synthesis  [Paper §2.2.4]
    # ------------------------------------------------------------------
    logger.info("Phase 2: Deep Feature Synthesis")
    feature_matrix = run_dfs(data)

    # Merge with target
    target = data.loan_performance[["application_id", "is_default"]].copy()
    merged = feature_matrix.join(
        target.set_index("application_id"), how="inner"
    )

    y = merged["is_default"]
    X = merged.drop(columns=["is_default"])

    # Drop ID columns — they leak identity information, not genuine features
    id_cols = [c for c in X.columns if c.endswith("_id") or c == "customer_id"]
    if id_cols:
        logger.info("Dropping %d ID columns: %s", len(id_cols), id_cols)
        X = X.drop(columns=id_cols)

    report.add_section("2. Feature Generation (DFS)", (
        f"- Features generated: {X.shape[1]}\n"
        f"- Depth: {config.DFS_DEPTH}\n"
        f"- Observations: {len(X)}\n"
    ))

    # ------------------------------------------------------------------
    # Time-based train / test / OOT split  [Paper §2.1, §2.4.1]
    # ------------------------------------------------------------------
    logger.info("Splitting data: train/test/OOT (%.0f/%.0f/%.0f)",
                config.TRAIN_RATIO * 100, config.TEST_RATIO * 100,
                config.OOT_RATIO * 100)

    # Sort by application_date for time-based splitting
    app_dates = data.credit_applications.set_index("application_id")["application_date"]
    app_dates = app_dates.loc[X.index]
    sorted_idx = app_dates.sort_values().index

    n = len(sorted_idx)
    n_train = int(n * config.TRAIN_RATIO)
    n_test = int(n * config.TEST_RATIO)

    train_idx = sorted_idx[:n_train]
    test_idx = sorted_idx[n_train:n_train + n_test]
    oot_idx = sorted_idx[n_train + n_test:]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    X_oot, y_oot = X.loc[oot_idx], y.loc[oot_idx]

    logger.info(
        "Split sizes — train: %d, test: %d, OOT: %d",
        len(X_train), len(X_test), len(X_oot),
    )

    # ------------------------------------------------------------------
    # Phase 3: Variable Config + Binning + WoE  [Paper §2.2.5, §2.2.6]
    # ------------------------------------------------------------------
    logger.info("Phase 3: Variable preprocessing (binning + WoE)")

    # 3a. Variable configuration
    var_configs = get_variable_configs(X_train.columns.tolist())

    # 3b. Binning
    binning_results = bin_features(X_train, y_train, var_configs)

    # 3c. WoE transform
    X_woe_train = transform_woe(X_train, binning_results)
    X_woe_test = transform_woe(X_test, binning_results)
    X_woe_oot = transform_woe(X_oot, binning_results)

    # Binning summary for report
    binning_rows = []
    for name, res in binning_results.results.items():
        binning_rows.append(f"| {name} | {res.iv:.4f} | {res.n_bins} | {res.monotonic_trend} | {res.status} |")
    binning_table = (
        "| Feature | IV | Bins | Monotonic Trend | Status |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(binning_rows[:50])  # top 50
    )
    report.add_section("3. Binning & WoE Summary", binning_table)

    # ------------------------------------------------------------------
    # Phase 4: IV & Feature Selection  [Paper §2.3.2]
    # ------------------------------------------------------------------
    logger.info("Phase 4: Feature selection (IV + MIV)")

    # 4a. Information Value
    iv_table = collect_iv(binning_results)
    candidate_features = filter_by_iv(iv_table)

    # 4a-bis. Bivariate analysis  [Assumption]
    biv = bivariate_analysis(binning_results, X_woe_train, y_train)
    biv_md = biv.to_markdown()
    report.add_section("3b. Bivariate Analysis", biv_md)

    # 4b. MIV forward selection
    miv_result = run_miv_selection(
        X_woe_train, y_train,
        X_woe_test, y_test,
        iv_table, binning_results,
        candidate_features,
        X_raw_train=X_train,
    )

    # MIV selection report
    miv_rows = []
    for step in miv_result.steps:
        miv_rows.append(
            f"| {step.step} | {step.feature_added} | {step.miv:.4f} "
            f"| {step.auc_train:.4f} | {step.auc_test:.4f} |"
        )
    miv_table = (
        "| Step | Feature Added | MIV | AUC (Train) | AUC (Test) |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(miv_rows)
        + f"\n\n- Stopping reason: {miv_result.stopping_reason}\n"
    )
    report.add_section("4. MIV Feature Selection", miv_table)

    # Plot MIV selection  [Paper Fig. 5]
    if miv_result.steps:
        plot_miv_selection(miv_result.steps)

    selected = miv_result.selected_features
    logger.info("Selected %d features: %s", len(selected), selected)

    # ------------------------------------------------------------------
    # Phase 5: Model Training  [Paper §2.4.2]
    # ------------------------------------------------------------------
    logger.info("Phase 5: Model training (logistic regression scorecard)")

    scorecard = train_scorecard(
        X_woe_train, y_train, selected, binning_results,
    )

    report.add_section(
        "5. Final Model — statsmodels Summary",
        f"```\n{scorecard.summary_text}\n```",
    )

    # Binning plots for selected features  [Paper §2.2.5, §2.2.6]
    plot_binning_tables(binning_results, selected)

    # Scorecard points
    if scorecard.scorecard_points is not None and len(scorecard.scorecard_points) > 0:
        pts_md = scorecard.scorecard_points.to_markdown(index=False)
        report.add_section("6. Scorecard Points", pts_md)

    # ------------------------------------------------------------------
    # Phase 5b: Benchmarking  [Paper §2.4.2, Table 1]
    # ------------------------------------------------------------------
    logger.info("Phase 5b: Benchmarking against tree ensembles")

    woe_cols = [f"woe_{f}" for f in selected]
    X_tr_const = sm.add_constant(X_woe_train[woe_cols])
    X_te_const = sm.add_constant(X_woe_test[woe_cols])
    X_oot_const = sm.add_constant(X_woe_oot[woe_cols])

    prob_train = scorecard.model.predict(X_tr_const)
    prob_test = scorecard.model.predict(X_te_const)
    prob_oot = scorecard.model.predict(X_oot_const)

    lr_auc_train = roc_auc_score(y_train, prob_train)
    lr_auc_test = roc_auc_score(y_test, prob_test)
    lr_auc_oot = roc_auc_score(y_oot, prob_oot)

    bench = benchmark_models(
        X_train, y_train, X_test, y_test, X_oot, y_oot,
        lr_auc_train, lr_auc_test, lr_auc_oot,
    )
    report.add_section("9. Benchmarking", bench.to_markdown(index=False))

    # ------------------------------------------------------------------
    # Phase 6: PD Calibration  [Paper §2.4.1]
    # ------------------------------------------------------------------
    logger.info("Phase 6: PD calibration")

    cal_result = calibrate_isotonic(
        prob_train, y_train.values,
        prob_test, y_test.values,
    )

    report.add_section("8. PD Calibration", (
        f"- Method: {cal_result.method}\n"
        f"- Brier Score (before): {cal_result.brier_before:.4f}\n"
        f"- Brier Score (after): {cal_result.brier_after:.4f}\n"
    ))

    # ------------------------------------------------------------------
    # Phase 7: Evaluation  [Paper §2.4.1]
    # ------------------------------------------------------------------
    logger.info("Phase 7: Evaluation")

    metrics_df = compute_all_splits(
        y_train.values, prob_train,
        y_test.values, prob_test,
        y_oot.values, prob_oot,
    )
    report.add_section("7. Model Performance", metrics_df.to_markdown(index=False))

    # Plots
    plot_roc_curve(y_test.values, prob_test, "LR Scorecard")
    plot_lorenz_curve(y_test.values, prob_test)
    plot_calibration_curve(y_test.values, prob_test, label="Before Calibration")
    if cal_result.calibrated_probs_test is not None:
        plot_calibration_curve(
            y_test.values, cal_result.calibrated_probs_test,
            label="After Calibration",
            save_path=config.PLOTS_DIR / "calibration_curve_after.png",
        )

    # ------------------------------------------------------------------
    # Phase 8: Residual Monitoring  [Paper §3]
    # ------------------------------------------------------------------
    logger.info("Phase 8: Residual monitoring")

    monitor = run_residual_monitoring(
        model_probs=prob_train,
        y_true=y_train.values,
        X_woe=X_woe_train,
        X_raw=X_train,
        selected_features=selected,
        binning_results=binning_results,
    )

    if monitor.candidate_table is not None and len(monitor.candidate_table) > 0:
        top = monitor.candidate_table.head(10)
        report.add_section(
            "10. Residual Monitoring",
            top.to_markdown(index=False),
        )

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report.write()

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("Report: %s", config.REPORT_FILE)
    logger.info("=" * 60)


def _data_summary(data) -> str:
    """Format data summary for the report."""
    n_defaults = data.loan_performance["is_default"].sum()
    n_apps = len(data.loan_performance)
    default_rate = n_defaults / n_apps if n_apps > 0 else 0
    return (
        f"- Customers: {len(data.customers):,}\n"
        f"- Accounts: {len(data.accounts):,}\n"
        f"- Transactions: {len(data.transactions):,}\n"
        f"- Applications: {len(data.credit_applications):,}\n"
        f"- Default rate: {100 * default_rate:.1f}%\n"
    )


if __name__ == "__main__":
    run_pipeline()
