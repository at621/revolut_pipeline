# Revolut Credit Risk Model — DFS + MIV Pipeline

Replication of the methodology from **"Enhancing Credit Risk Models at Revolut by Combining Deep Feature Synthesis and Marginal Information Value"** (Spinella & Krisciunas, Edinburgh Credit Scoring Conference, Aug 2025).

The paper presents an automated credit scorecard development pipeline that combines:
- **Deep Feature Synthesis (DFS)** for automated feature generation from relational data
- **Marginal Information Value (MIV)** for greedy forward feature selection
- **WoE-based logistic regression** for interpretable credit scoring

This implementation uses **synthetic data** that mimics the relational structure of a digital banking platform (customers, accounts, transactions, credit applications, loan performance).

## Pipeline Overview

```
Synthetic Data → DFS Feature Generation → Coarse Binning & WoE → IV Calculation
→ MIV Forward Selection → Logistic Regression Scorecard → PD Calibration
→ Evaluation & Benchmarking → Residual Monitoring
```

| Phase | Description | Paper Reference |
|-------|-------------|-----------------|
| Data Generation | Synthetic relational dataset (customers, accounts, transactions, applications, loans) | §4.1 |
| Feature Engineering | Deep Feature Synthesis via `featuretools` (depth=2) | §2.2.4 |
| Binning & WoE | Optimal binning with monotonicity constraints via `optbinning` | §2.2.5, §2.2.6 |
| Feature Selection | IV pre-filter + greedy MIV forward selection with correlation guard | §2.3.2 |
| Scorecard | `statsmodels` logistic regression with full statistical inference | §2.4.2 |
| Calibration | Platt scaling and isotonic regression | §2.4.1 |
| Evaluation | Gini, AUC, KS, Brier score, Lorenz & calibration curves | §2.4.1 |
| Benchmarking | LR vs Gradient Boosted Trees vs Random Forest | §2.4.2, Table 1 |
| Monitoring | Automated residual monitoring for value-adding features | §3 |

## Project Structure

```
revolut_credit_risk/
├── config.py                      # All hyperparameters in one place
├── logging_config.py              # Logging setup (console + file)
├── report.py                      # Markdown report writer
├── data/
│   ├── synthetic_data.py          # Synthetic relational dataset generator
│   └── datasets/                  # Saved datasets (parquet + metadata)
├── features/
│   ├── dfs_engine.py              # Deep Feature Synthesis wrapper
│   ├── binning.py                 # optbinning wrapper for coarse binning + WoE
│   └── variable_config.py         # Optional LLM-based variable config
├── selection/
│   ├── information_value.py       # IV and MIV calculation
│   └── miv_selector.py            # Greedy forward MIV feature selection
├── model/
│   ├── scorecard.py               # statsmodels logistic regression scorecard
│   └── calibration.py             # PD calibration (Platt scaling, isotonic)
├── evaluation/
│   └── metrics.py                 # Gini, AUC, Brier score, calibration plots
├── monitoring/
│   └── residual_monitor.py        # Automated value-adding feature monitoring
├── pipeline.py                    # End-to-end orchestrator
├── notebooks/
│   └── walkthrough.ipynb          # Interactive demo with visualisations
├── outputs/                       # Pipeline outputs (report, plots, logs)
├── tests/                         # Unit and integration tests
└── requirements.txt
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/at621/revolut_pipeline.git
cd revolut_pipeline

# Install dependencies
pip install -r revolut_credit_risk/requirements.txt

# Run the full pipeline
python -m revolut_credit_risk.pipeline

# Run tests
pytest revolut_credit_risk/tests/ -v
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `featuretools` | Deep Feature Synthesis (cited in the paper) |
| `optbinning` | Optimal binning, WoE transformation, IV computation |
| `statsmodels` | Logistic regression with full statistical output |
| `scikit-learn` | Benchmark models, calibration, metrics |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisations |

## Configuration

All hyperparameters are centralised in `revolut_credit_risk/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CUSTOMERS` | 10,000 | Synthetic dataset size |
| `DEFAULT_RATE` | 0.06 | Target default rate |
| `DFS_DEPTH` | 2 | Feature synthesis depth |
| `IV_THRESHOLD` | 0.02 | Minimum IV to retain a feature |
| `MIV_THRESHOLD` | 0.02 | Minimum MIV to add a feature |
| `CORRELATION_THRESHOLD` | 0.6 | Max pairwise correlation allowed |
| `USE_LLM_CONFIG` | False | Enable LLM-assisted variable config (optional) |

## Outputs

After running the pipeline, results are saved to `revolut_credit_risk/outputs/`:

- **`pipeline_report.md`** — Full results report including statsmodels summary, scorecard points, and performance metrics
- **`pipeline.log`** — Detailed debug log
- **`variable_config.yaml`** — Per-variable binning configuration (auto-generated, human-editable)
- **`plots/`** — Lorenz curve, ROC curve, calibration curve, MIV selection chart, residual monitoring

## References

- Spinella, S. & Krisciunas, A. (2025). *Enhancing Credit Risk Models at Revolut by Combining Deep Feature Synthesis and Marginal Information Value*. Edinburgh Credit Scoring Conference.
- Kanter, J. M. & Veeramachaneni, K. (2015). *Deep Feature Synthesis: Towards Automating Data Science Endeavors*. IEEE DSAA.
- Siddiqi, N. (2017). *Intelligent Credit Scoring*. Wiley.
- Scallan, G. (2011). *Selecting features for credit risk modelling using MIV*.

## License

[MIT](LICENSE)
