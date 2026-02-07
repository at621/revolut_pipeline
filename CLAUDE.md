# Revolut Credit Risk Model — DFS + MIV Pipeline

Replication of the methodology from the Revolut paper on Deep Feature Synthesis + Marginal Information Value for automated credit scorecard development.

## Key references

- Implementation spec: @docs/implementation_spec.md — the single source of truth for all design decisions
- Source paper: `docs/Paper-Enhancing-Credit-Risk-Models-at-Revolut-by-combining-Deep-Feature-Synthesis-and-Marginal-Information-Value.pdf`

## Project structure

All Python code lives under `revolut_credit_risk/`. The spec defines 10 modules — see @docs/implementation_spec.md for the full layout. Key entry points:

- `revolut_credit_risk/pipeline.py` — end-to-end orchestrator, run with `python -m revolut_credit_risk.pipeline`
- `revolut_credit_risk/config.py` — all hyperparameters in one place
- `revolut_credit_risk/notebooks/walkthrough.ipynb` — interactive demo

## Commands

```bash
# Install dependencies
pip install -r revolut_credit_risk/requirements.txt

# Run the full pipeline
python -m revolut_credit_risk.pipeline

# Run tests
pytest revolut_credit_risk/tests/ -v

# Run a single test file
pytest revolut_credit_risk/tests/test_binning.py -v

# Typecheck
mypy revolut_credit_risk/ --ignore-missing-imports
```

## Code style

- Python 3.10+. Use type hints on all function signatures.
- Use `logging` (not print) everywhere. Each module: `logger = logging.getLogger(__name__)`.
- Imports: stdlib first, then third-party, then local. Use `from __future__ import annotations` at the top of every file.
- Prefer `pandas` DataFrames over raw dicts for tabular data.
- Use `pathlib.Path` for file paths, not string concatenation.

## Architecture constraints — IMPORTANT

- **statsmodels for logistic regression**, not sklearn. We need the full `result.summary()` output (coefficients, p-values, CIs, pseudo R²).
- **optbinning for binning/WoE**. Do not manually implement binning — use `OptimalBinning` class.
- **featuretools for DFS**. Do not manually engineer features — use `ft.dfs()` with entity sets and cutoff times.
- The LLM variable config step (`features/variable_config.py`) is **optional**. The pipeline MUST work with `USE_LLM_CONFIG=False` using sensible defaults. Never make the Anthropic API a hard dependency.
- **Pipeline and notebook are independent.** `pipeline.py` must run end-to-end on its own without requiring the notebook. The notebook (`walkthrough.ipynb`) can be run independently and should import/reuse pipeline modules, but the pipeline must never depend on the notebook.

## Provenance annotations

Every implementation decision must be traceable to either the paper or marked as an assumption. When writing code:
- Add a comment like `# [Paper §2.3.2] MIV formula` for paper-sourced logic
- Add a comment like `# [Assumption] patience=2 for AUC plateau` for our design choices
- See the Provenance Legend in @docs/implementation_spec.md for the full annotation format

## Data handling

- Synthetic datasets are saved as `.parquet` files under `revolut_credit_risk/data/datasets/<dataset_name>/`
- Each dataset folder has a `metadata.json` with generation params
- `config.GENERATE_NEW_DATA` controls whether to regenerate or load existing data
- NEVER commit large parquet files. The `data/datasets/` folder should be in `.gitignore`.

## Common gotchas

- `optbinning.OptimalBinning` can fail silently on constant columns or columns with too few unique values. Always check `optb.status` after fitting.
- `featuretools.dfs()` with cutoff times can be slow. Log timing and feature counts.
- WoE transform produces `±inf` for empty bins. Handle with `np.clip` or filter before fitting logistic regression.
- `statsmodels.Logit.fit()` may not converge with perfectly separable data. Use `method='bfgs'` or `maxiter=100` as fallbacks.
- On Windows, `featuretools` may raise warnings about Woodwork dtypes. These are generally safe to suppress.

## Testing approach

- **All functions must pass unit tests.** Every public function in every module must have corresponding unit tests. Do not consider a module complete until its tests pass.
- Unit tests for WoE/IV/MIV calculations using small hand-crafted DataFrames with known expected values.
- Integration test: run `pipeline.py` end-to-end on a small synthetic dataset (N_CUSTOMERS=500) and assert pipeline completes + report file is generated.
- Sanity checks: Gini > 0.4 on train, IV of top features in 0.1-0.5 range, all LR coefficients positive.

## Output files

All pipeline outputs go to `revolut_credit_risk/outputs/`:
- `pipeline_report.md` — full results report with statsmodels summary
- `pipeline.log` — detailed debug log
- `variable_config.yaml` — per-variable binning config (auto-generated, human-editable)
- `plots/` — all generated figures (PNG)
