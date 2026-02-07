"""Cross-validate MIV computation: Python vs R PDtoolkit.

Creates a shared dataset, computes MIV in Python, exports data for R,
then calls R and compares results.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from revolut_credit_risk.selection.miv_selector import _compute_miv
from revolut_credit_risk.features.binning import bin_features, transform_woe

SCRATCH = Path(__file__).resolve().parent


def create_test_data():
    """Create a small dataset with clear signal for cross-validation."""
    rng = np.random.default_rng(123)
    n = 2000

    # Two features with different signal strength
    f1 = rng.normal(5, 2, n)  # strong signal
    f2 = rng.normal(10, 3, n)  # medium signal
    f3 = rng.normal(0, 1, n)  # in-model feature (to create a non-trivial model)

    # Target driven by f3 (in-model) + some residual from f1, f2
    latent = 0.8 * (f3 - f3.mean()) / f3.std() + 0.3 * (f1 - f1.mean()) / f1.std() + 0.15 * (f2 - f2.mean()) / f2.std()
    p = 1 / (1 + np.exp(-latent - 0.5))  # shift to get ~15% default rate
    y = (rng.random(n) < p).astype(int)

    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
    y_series = pd.Series(y, name="target")

    print(f"Dataset: n={n}, default_rate={y.mean():.3f}")
    print(f"  f1 range: [{f1.min():.2f}, {f1.max():.2f}]")
    print(f"  f2 range: [{f2.min():.2f}, {f2.max():.2f}]")
    print(f"  f3 range: [{f3.min():.2f}, {f3.max():.2f}]")

    return X, y_series


def compute_python_miv(X, y):
    """Compute MIV for f1 and f2 with f3 as the in-model feature."""
    # Step 1: Bin all features
    binning_results = bin_features(X, y)

    print("\nBinning results:")
    for name, res in binning_results.results.items():
        print(f"  {name}: IV={res.iv:.4f}, bins={res.n_bins}, status={res.status}")

    # Step 2: WoE transform
    X_woe = transform_woe(X, binning_results)

    # Step 3: Fit a model using f3 only (so we can measure MIV of f1 and f2)
    import statsmodels.api as sm
    X_model = sm.add_constant(X_woe[["woe_f3"]])
    model = sm.Logit(y, X_model).fit(disp=False)
    pred_probs = model.predict(X_model)

    print(f"\nIn-model feature: f3")
    print(f"Model intercept: {model.params['const']:.4f}")
    print(f"Model coef(woe_f3): {model.params['woe_f3']:.4f}")

    # Step 4: Compute MIV for f1 and f2 using raw values
    results = {}
    for feat in ["f1", "f2"]:
        optb = binning_results.get_optb(feat)
        x_raw = X[feat].values.astype(float)
        miv_val, p_val = _compute_miv(x_raw, y.values, pred_probs.values, optb)
        results[feat] = {"miv": miv_val, "p_value": p_val}
        print(f"\nPython MIV({feat}): {miv_val:.6f}, p_value: {p_val:.6f}")

        # Also export the bin-level details for comparison
        bins = optb.transform(x_raw, metric="bins")
        unique_bins = np.unique(bins)

        total_bad = y.sum()
        total_good = len(y) - total_bad
        total_exp_bad = pred_probs.sum()
        total_exp_good = (1 - pred_probs).sum()

        print(f"  Bin details (total_bad={total_bad}, total_good={total_good}):")
        for b in unique_bins:
            mask = bins == b
            n_obs = mask.sum()
            nb_obs = y[mask].sum()
            ng_obs = n_obs - nb_obs
            nb_exp = pred_probs[mask].sum()
            ng_exp = (1 - pred_probs[mask]).sum()

            p_bad = nb_obs / total_bad if total_bad > 0 else 0
            p_good = ng_obs / total_good if total_good > 0 else 0
            woe_obs = np.log(p_bad / p_good) if p_bad > 0 and p_good > 0 else float("nan")

            p_bad_exp = nb_exp / total_exp_bad if total_exp_bad > 0 else 0
            p_good_exp = ng_exp / total_exp_good if total_exp_good > 0 else 0
            woe_exp = np.log(p_bad_exp / p_good_exp) if p_bad_exp > 0 and p_good_exp > 0 else float("nan")

            print(f"    {b:>25s}: n={n_obs:4d}, nb_obs={nb_obs:3.0f}, ng_obs={ng_obs:3.0f}, "
                  f"nb_exp={nb_exp:.2f}, ng_exp={ng_exp:.2f}, "
                  f"woe_obs={woe_obs:+.4f}, woe_exp={woe_exp:+.4f}, delta={woe_obs-woe_exp:+.4f}")

    # Step 5: Export data for R
    # R's PDtoolkit expects categorical (character) risk factors, so we export
    # the bin labels directly
    export_df = pd.DataFrame({"target": y.values})
    export_df["pred"] = pred_probs.values

    for feat in ["f1", "f2"]:
        optb = binning_results.get_optb(feat)
        x_raw = X[feat].values.astype(float)
        bin_labels = optb.transform(x_raw, metric="bins")
        export_df[feat] = bin_labels

    export_path = SCRATCH / "cross_val_data.csv"
    export_df.to_csv(export_path, index=False)
    print(f"\nExported data to {export_path}")
    print(f"  Columns: {list(export_df.columns)}")
    print(f"  Shape: {export_df.shape}")
    print(f"  f1 bins: {sorted(export_df['f1'].unique())}")
    print(f"  f2 bins: {sorted(export_df['f2'].unique())}")

    return results


def compute_r_miv():
    """Call R to compute MIV using PDtoolkit and return results."""
    r_script = SCRATCH / "cross_val_miv.R"
    result_path = SCRATCH / "r_miv_results.json"

    r_code = f'''
library(PDtoolkit)
library(jsonlite)

# Read the shared dataset
df <- read.csv("{str(SCRATCH / 'cross_val_data.csv').replace(chr(92), '/')}")
cat("R: Loaded data with", nrow(df), "rows\\n")
cat("R: Default rate:", mean(df$target), "\\n")
cat("R: Columns:", paste(names(df), collapse=", "), "\\n")
cat("R: f1 bins:", paste(sort(unique(df$f1)), collapse=", "), "\\n")
cat("R: f2 bins:", paste(sort(unique(df$f2)), collapse=", "), "\\n")

# The data already has:
#   - target: binary 0/1
#   - pred: model predicted probabilities
#   - f1, f2: bin labels (character)

# Make sure risk factors are character type (PDtoolkit requirement)
df$f1 <- as.character(df$f1)
df$f2 <- as.character(df$f2)

# We need to create a "current model" that produces the pred column.
# Since we already have the predictions, we'll use the internal miv() function
# by providing the model formula and letting it predict.
# But we need to trick it -- the miv() function calls glm() internally.

# Instead, let's call woe.tbl and compute MIV manually, matching the
# PDtoolkit miv() function logic exactly.

results <- list()

for (rf_name in c("f1", "f2")) {{
    cat("\\n--- Computing MIV for", rf_name, "---\\n")

    # Observed WoE table (using actual target)
    observed <- woe.tbl(tbl = df, x = rf_name, y = "target", y.check = TRUE)
    cat("Observed WoE table:\\n")
    print(observed[, c("bin", "no", "ng", "nb", "woe")])

    # Expected WoE table (using predicted probabilities as target)
    expected <- woe.tbl(tbl = df, x = rf_name, y = "pred", y.check = FALSE)
    cat("Expected WoE table:\\n")
    print(expected[, c("bin", "no", "ng", "nb", "woe")])

    # Merge observed and expected
    comm.cols <- c("bin", "no", "ng", "nb", "woe")
    miv.tbl <- merge(observed[, comm.cols],
                     expected[, comm.cols],
                     by = "bin",
                     all = TRUE,
                     suffixes = c(".o", ".e"))

    cat("Merged MIV table:\\n")
    print(miv.tbl)

    # Compute MIV exactly as PDtoolkit does
    miv.tbl$delta <- miv.tbl$woe.o - miv.tbl$woe.e
    miv.val.g <- sum(miv.tbl$ng.o * miv.tbl$delta) / sum(miv.tbl$ng.o)
    miv.val.b <- sum(miv.tbl$nb.o * miv.tbl$delta) / sum(miv.tbl$nb.o)
    miv.val <- miv.val.g - miv.val.b

    cat("MIV components: miv.val.g =", miv.val.g, ", miv.val.b =", miv.val.b, "\\n")
    cat("MIV =", miv.val, "\\n")

    # Chi-square test
    m.chiq.g <- miv.tbl$ng.o * log(miv.tbl$ng.o / miv.tbl$ng.e)
    m.chiq.b <- miv.tbl$nb.o * log(miv.tbl$nb.o / miv.tbl$nb.e)
    m.chiq.gb <- m.chiq.g + m.chiq.b
    m.chiq.stat <- 2 * sum(m.chiq.gb)
    p.val <- pchisq(m.chiq.stat, nrow(miv.tbl) - 1, lower.tail = FALSE)

    cat("Chi-square stat =", m.chiq.stat, ", p-value =", p.val, "\\n")

    results[[rf_name]] <- list(
        miv = miv.val,
        miv_g = miv.val.g,
        miv_b = miv.val.b,
        chisq_stat = m.chiq.stat,
        p_value = p.val,
        n_bins = nrow(miv.tbl)
    )
}}

# Save results as JSON
json_out <- toJSON(results, auto_unbox = TRUE, pretty = TRUE)
writeLines(json_out, "{str(result_path).replace(chr(92), '/')}")
cat("\\nResults saved to JSON\\n")
'''

    r_script.write_text(r_code)

    print("\n" + "=" * 60)
    print("Running R script...")
    print("=" * 60)

    proc = subprocess.run(
        ["Rscript", str(r_script)],
        capture_output=True, text=True, timeout=60,
    )
    print(proc.stdout)
    if proc.stderr:
        # Filter out package loading messages
        for line in proc.stderr.splitlines():
            if any(skip in line for skip in ["Loading required", "Attaching", "following objects", "masked from", "  filter", "  lag", "  intersect", "  setdiff", "  setequal", "  union", "  src", "  summarize", "  format.pval", "  units", "  power"]):
                continue
            if line.strip():
                print(f"R stderr: {line}")

    if proc.returncode != 0:
        print(f"R script failed with return code {proc.returncode}")
        return None

    with open(result_path) as f:
        r_results = json.load(f)

    return r_results


def compare_results(py_results, r_results):
    """Compare Python and R MIV results."""
    print("\n" + "=" * 60)
    print("COMPARISON: Python vs R PDtoolkit")
    print("=" * 60)

    # NOTE: R uses WoE = log(Good/Bad), Python uses WoE = log(Bad/Good).
    # The MIV formulas are adjusted so the final MIV values should match.
    # R's MIV = sum((P(G) - P(B)) * delta_R) where delta_R = woe_obs_R - woe_exp_R
    # Python's MIV = sum((P(B) - P(G)) * delta_P) where delta_P = woe_obs_P - woe_exp_P
    # Since woe_R = -woe_P, delta_R = -delta_P, so:
    #   R_MIV = sum((P(G)-P(B)) * (-delta_P)) = sum((P(B)-P(G)) * delta_P) = Python_MIV

    all_match = True
    for feat in ["f1", "f2"]:
        py_miv = py_results[feat]["miv"]
        py_pval = py_results[feat]["p_value"]
        r_miv = r_results[feat]["miv"]
        r_pval = r_results[feat]["p_value"]
        r_chisq = r_results[feat]["chisq_stat"]

        miv_diff = abs(py_miv - r_miv)
        pval_diff = abs(py_pval - r_pval)
        miv_match = miv_diff < 1e-4
        pval_match = pval_diff < 1e-4

        print(f"\n  Feature: {feat}")
        print(f"    Python MIV:  {py_miv:+.6f}")
        print(f"    R MIV:       {r_miv:+.6f}")
        print(f"    Difference:  {miv_diff:.2e}  {'MATCH' if miv_match else 'MISMATCH'}")
        print(f"    Python p-val: {py_pval:.6f}")
        print(f"    R p-val:      {r_pval:.6f}")
        print(f"    p-val diff:   {pval_diff:.2e}  {'MATCH' if pval_match else 'MISMATCH'}")

        if not miv_match or not pval_match:
            all_match = False

    print(f"\n{'=' * 60}")
    if all_match:
        print("RESULT: All MIV and p-values MATCH between Python and R!")
    else:
        print("RESULT: MISMATCH detected -- see details above")
    print("=" * 60)

    return all_match


if __name__ == "__main__":
    X, y = create_test_data()
    py_results = compute_python_miv(X, y)
    r_results = compute_r_miv()
    if r_results is not None:
        compare_results(py_results, r_results)
    else:
        print("R computation failed -- cannot compare")
