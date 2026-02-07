# Replication of Revolut DFS + MIV Credit Risk Methodology

## Context

This plan replicates the methodology from "Enhancing Credit Risk Models at Revolut by Combining Deep Feature Synthesis and Marginal Information Value" (Spinella & Krisciunas, Edinburgh Credit Scoring Conference, Aug 2025). The paper presents an automated credit scorecard development pipeline that combines Deep Feature Synthesis (DFS) for automated feature generation from relational data with Marginal Information Value (MIV) for greedy forward feature selection, feeding into a WoE-based logistic regression model.

We will build this end-to-end in Python using **synthetic data** that mimics the relational structure of a digital banking platform (customers, accounts, transactions, credit applications, loan performance).

### Provenance Legend

Throughout this document, every design decision is annotated with its source:

- **`[Paper §X.Y.Z]`** — Directly described in or derived from the Revolut paper, with the specific section reference.
- **`[Paper Fig. N]`** — Based on a figure in the paper.
- **`[Paper Table N]`** — Based on a table in the paper.
- **`[Paper Ref. N]`** — Sourced from a reference cited in the paper (e.g., Siddiqi [2], Scallan [5]).
- **`[Assumption]`** — Our implementation choice, not described in the paper. May follow standard credit risk industry practice or be a pragmatic engineering decision.

---

## Project Structure

> `[Assumption]` The paper does not prescribe a project structure. This modular layout is our engineering choice to keep the codebase maintainable.

```
revolut_credit_risk/
├── config.py                      # Global constants & hyperparameters
├── logging_config.py              # Logging setup (console + file)
├── report.py                      # Markdown report writer
├── data/
│   ├── synthetic_data.py          # Synthetic relational dataset generator
│   └── datasets/                  # Each dataset in its own subfolder
│       └── synthetic_v1/          # Example: auto-created by pipeline
│           ├── customers.parquet
│           ├── accounts.parquet
│           ├── transactions.parquet
│           ├── credit_applications.parquet
│           ├── loan_performance.parquet
│           └── metadata.json      # Generation params, row counts, default rate, timestamp
├── features/
│   ├── dfs_engine.py              # Deep Feature Synthesis wrapper
│   ├── binning.py                 # optbinning wrapper for coarse binning + WoE
│   └── variable_config.py         # LLM-based variable config generator
├── selection/
│   ├── information_value.py       # IV and MIV calculation
│   └── miv_selector.py            # Greedy forward MIV feature selection loop
├── model/
│   ├── scorecard.py               # statsmodels logistic regression scorecard
│   └── calibration.py             # PD calibration (Platt scaling, isotonic)
├── evaluation/
│   └── metrics.py                 # Gini, AUC, Brier score, calibration plots
├── monitoring/
│   └── residual_monitor.py        # Automated value-adding feature monitoring
├── pipeline.py                    # End-to-end orchestrator
├── notebooks/
│   └── walkthrough.ipynb          # Interactive walkthrough with visualisations
├── outputs/                       # All pipeline outputs land here
│   ├── variable_config.yaml       # Per-variable optbinning config (auto-generated, editable)
│   ├── pipeline_report.md         # Full results report with statsmodels summary
│   ├── pipeline.log               # Detailed log file
│   └── plots/                     # All generated figures (PNG)
├── requirements.txt
└── README.md
```

---

## Cross-Cutting: Logging (`logging_config.py`)

> `[Assumption]` The paper does not discuss logging. This is our engineering decision to provide visibility into the pipeline's progress for the modeller.

Every module uses Python's `logging` module. `logging_config.py` sets up:

- **Console handler**: `INFO` level, concise format -- shows the modeller where the pipeline is:
  ```
  2026-02-07 14:23:01 | INFO | dfs_engine | Generating features for 8000 applications (depth=2)...
  2026-02-07 14:23:45 | INFO | dfs_engine | DFS complete: 347 features generated in 44s
  2026-02-07 14:23:45 | INFO | binning | Binning 347 features with optbinning...
  2026-02-07 14:24:02 | INFO | binning | Feature 'MEAN(transactions.amount)': IV=0.312, 6 bins, monotonic=descending
  2026-02-07 14:25:10 | INFO | miv_selector | Step 1: Added 'MEAN(transactions.amount)' (IV=0.312, AUC=0.714)
  2026-02-07 14:25:15 | INFO | miv_selector | Step 2: Added 'STD(transactions.amount)' (MIV=0.189, AUC=0.751)
  ```
- **File handler**: `DEBUG` level, writes to `outputs/pipeline.log` with full detail (binning tables, coefficient values, etc.)

Each module gets its logger via `logger = logging.getLogger(__name__)`.

Key log points per module:
| Module | What gets logged |
|---|---|
| `synthetic_data` | Row counts per entity, default rate, data generation time |
| `dfs_engine` | Number of features generated, DFS runtime, feature name samples |
| `binning` | Per-feature: IV, number of bins, monotonic trend chosen, any binning failures |
| `variable_config` | LLM call made (or skipped), config file loaded/saved path |
| `information_value` | IV distribution summary, count of features passing IV threshold |
| `miv_selector` | Each step: feature added, MIV value, cumulative AUC (train & test), reason for stopping |
| `scorecard` | statsmodels summary (full text), coefficient signs, score conversion params |
| `calibration` | Calibration method used, Brier score before/after |
| `metrics` | All metric values on train/test/OOT |
| `residual_monitor` | Top candidate features by MIV, count evaluated |
| `pipeline` | Phase start/end timestamps, total runtime |

---

## Cross-Cutting: Markdown Report (`report.py`)

> `[Assumption]` The paper does not prescribe a report format. This is our design choice to produce a human-readable summary of all pipeline results. The content of the report mirrors the paper's evaluation structure (Gini, calibration, benchmarking) but the markdown output format is our own.

`report.py` provides a `ReportWriter` class that accumulates sections and writes `outputs/pipeline_report.md` at the end. Each pipeline phase appends its results.

The final report structure:

~~~markdown
# Credit Risk Model Development Report
_Generated: 2026-02-07 14:30:00_

## 1. Data Summary
- Customers: 10,000 | Accounts: 15,234 | Transactions: 498,120
- Applications: 8,000 | Default rate: 5.8%
- Train: 4,800 | Test: 1,600 | OOT: 1,600

## 2. Feature Generation (DFS)
- Features generated: 347
- Depth: 2 | Primitives: sum, mean, std, min, max, count, ...
- Runtime: 44s

## 3. Binning & WoE Summary
| Feature | IV | Bins | Monotonic Trend | Status |
|---|---|---|---|---|
| MEAN(transactions.amount) | 0.312 | 6 | descending | OK |
| STD(transactions.amount) | 0.189 | 5 | ascending | OK |
| ... | ... | ... | ... | ... |
- Features passing IV >= 0.02: 142 / 347

## 3b. Bivariate Analysis — Univariate Gini & IV
| # | Feature | IV | IV Strength | Univariate Gini | Bins |
|---|---|---|---|---|---|
| 1 | MEAN(transactions.amount) | 0.312 | Strong | 0.418 | 6 |
| 2 | STD(transactions.amount) | 0.189 | Medium | 0.337 | 5 |
| ... | ... | ... | ... | ... | ... |
- Summary: 42 Strong, 68 Medium, 32 Weak, 205 Poor

## 4. MIV Feature Selection
| Step | Feature Added | MIV | AUC (Train) | AUC (Test) |
|---|---|---|---|---|
| 1 | MEAN(transactions.amount) | 0.312 | 0.714 | 0.702 |
| 2 | STD(transactions.amount) | 0.189 | 0.751 | 0.738 |
| ... | ... | ... | ... | ... |
- Stopping reason: AUC plateau after step 14
![MIV Selection](plots/miv_selection.png)

## 5. Final Model — statsmodels Summary
```
                           Logit Regression Results
==============================================================================
Dep. Variable:             is_default   No. Observations:                 4800
Model:                          Logit   Df Residuals:                     4786
Method:                           MLE   Df Model:                           13
Date:                Fri, 07 Feb 2026   Pseudo R-squ.:                  0.1842
...
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -2.8341      0.089    -31.843      0.000      -3.008      -2.660
woe_feat_1     0.9123      0.045     20.272      0.000       0.824       1.001
woe_feat_2     0.7845      0.052     15.087      0.000       0.683       0.886
...
==============================================================================
```

## 6. Scorecard Points
| Feature | Bin | WoE | Points |
|---|---|---|---|
| ... | ... | ... | ... |

## 7. Model Performance
| Metric | Train | Test | OOT |
|---|---|---|---|
| AUC | 0.847 | 0.831 | 0.819 |
| Gini | 0.694 | 0.662 | 0.638 |
| KS | 0.512 | 0.498 | 0.480 |
| Brier Score | 0.048 | 0.051 | 0.054 |
![Lorenz Curve](plots/lorenz_curve.png)
![ROC Curve](plots/roc_curve.png)

## 8. PD Calibration
- Method: Isotonic Regression
- Brier Score (before): 0.051 | (after): 0.046
![Calibration Curve](plots/calibration_curve.png)

## 9. Benchmarking
| Model | Gini (Train) | Gini (Test) | Gini (OOT) |
|---|---|---|---|
| WoE Logistic Regression | 0.694 | 0.662 | 0.638 |
| Gradient Boosted Trees | 0.710 | 0.658 | 0.630 |
| Random Forest | 0.720 | 0.645 | 0.620 |

## 10. Residual Monitoring
| Candidate Feature | MIV vs Current Model |
|---|---|
| ... | ... |
![Residual Monitor](plots/residual_monitor.png)
~~~

---

## Phase 1: Data Generation / Loading (`data/synthetic_data.py`)

> `[Assumption]` The paper uses proprietary Revolut data. Synthetic data generation is entirely our design to enable replication without access to real banking data.

### Generate vs. Load

> `[Assumption]` The generate-vs-load toggle and parquet-based dataset storage are our engineering choices for reproducibility and flexibility.

Controlled by `config.GENERATE_NEW_DATA`:

- **`True`** (default): Generate a new synthetic dataset, save each entity as a `.parquet` file under `config.DATASET_PATH` (e.g. `data/datasets/synthetic_v1/`), and write a `metadata.json` with generation parameters, row counts, default rate, and timestamp.
- **`False`**: Load existing parquet files from `config.DATASET_PATH`. The pipeline validates that all 5 required files exist and logs the metadata. This lets you re-run feature engineering / modelling without regenerating data, or plug in a different dataset entirely.

```python
if config.GENERATE_NEW_DATA:
    data = generate_synthetic_data(config)
    save_dataset(data, config.DATASET_PATH)   # writes parquet + metadata.json
else:
    data = load_dataset(config.DATASET_PATH)   # reads parquet files
```

The subfolder structure means you can keep multiple datasets side-by-side (e.g. `synthetic_v1`, `synthetic_v2_high_default`, `real_anonymised`) and switch between them by changing `DATASET_PATH`.

### Goal
Generate a realistic relational dataset that mimics the data a neobank like Revolut would have. The data must support DFS entity-set construction with meaningful relationships.

### Entity Schema

> `[Assumption]` The specific entity schemas below are our design. The paper mentions that DFS operates on "a collection of interconnected entities and their corresponding data tables" `[Paper §2.2.4]` and that Revolut uses "transactional data" and "internal data" `[Paper §4.1]`, but does not publish its internal schema. Our schema is designed to be a plausible approximation of a digital bank's relational data model.

**1. `customers` (target entity ancestor)**
| Column | Type | Description |
|---|---|---|
| `customer_id` | int (PK) | Unique customer identifier |
| `signup_date` | datetime | Account creation date |
| `age` | int | Customer age (18-75) |
| `income_band` | categorical | Low / Medium / High / Very High |
| `country` | categorical | UK / IE / LT / DE / FR |
| `employment_status` | categorical | Employed / Self-Employed / Unemployed / Student / Retired |

> `[Assumption]` Country values (UK, IE, LT, DE, FR) are chosen to reflect Revolut's main European markets mentioned in the paper's multi-jurisdictional context `[Paper §1.1]`. Specific demographic columns are our design.

**2. `accounts` (one customer can have multiple accounts)**
| Column | Type | Description |
|---|---|---|
| `account_id` | int (PK) | Unique account identifier |
| `customer_id` | int (FK -> customers) | Owner |
| `account_type` | categorical | Current / Savings / Trading |
| `open_date` | datetime | Account opening date |
| `currency` | categorical | GBP / EUR / USD |

> `[Assumption]` Account types reflect Revolut's known product offerings. The multi-account structure enables DFS to generate aggregation features across accounts.

**3. `transactions` (many per account, time-series)**
| Column | Type | Description |
|---|---|---|
| `transaction_id` | int (PK) | Unique transaction identifier |
| `account_id` | int (FK -> accounts) | Source account |
| `transaction_date` | datetime | Transaction timestamp |
| `amount` | float | Transaction amount (positive = credit, negative = debit) |
| `category` | categorical | Groceries / Travel / Entertainment / Utilities / Salary / Transfer / Rent / Shopping / Restaurants / Other |
| `merchant_name` | str | Simulated merchant |
| `transaction_state` | categorical | COMPLETED / PENDING / DECLINED |

> `[Paper §4.1, §3.1, Fig. 9]` The paper explicitly mentions "transactional data" and "travel-related spending" as predictive features (Fig. 9 shows `SUM(ft_table_completed_tx_cards.amount_gbp_cards WHERE budget_category = TRAVEL)`). Transaction categories like Travel are directly inspired by this. The general use of transaction data for credit risk is a core theme of the paper `[Paper §1.1, §4.1]`.

**4. `credit_applications` (the modelling target entity)**
| Column | Type | Description |
|---|---|---|
| `application_id` | int (PK) | Unique application identifier |
| `customer_id` | int (FK -> customers) | Applicant |
| `application_date` | datetime | Date of credit application |
| `product_type` | categorical | Personal Loan / Credit Card |
| `requested_amount` | float | Amount requested |
| `approved` | bool | Whether the application was approved |

> `[Paper §4.1, §4.2]` Product types (Personal Loan, Credit Card) match the paper's case studies which cover "personal unsecured personal loans portfolio" `[Paper §4.1]` and "credit card portfolio" `[Paper §4.2]`.

**5. `loan_performance` (outcome for approved applications)**
| Column | Type | Description |
|---|---|---|
| `application_id` | int (FK -> credit_applications) | The application |
| `months_on_book` | int | Number of months since disbursement |
| `dpd_max` | int | Maximum days past due reached |
| `is_default` | bool | **Target variable**: 1 if 90+ DPD within 12 months, else 0 |

> `[Paper §2.2.1, §2.2.2, §2.2.3]` The target variable definition follows the paper's discussion of "Bad Definition" and prediction horizon. The paper states the bad event should be "terminal" with "a high probability of rolling over to worse statuses" `[Paper §2.2.2]`. The 90+ DPD threshold is standard industry practice; the paper's Fig. 3 shows cumulative default rate curves for various DPD thresholds. The prediction horizon concept (12 months) follows `[Paper §2.2.3]` which discusses using cumulative bad rate curves to choose an appropriate window, and mentions the "12 MoB Threshold" in Fig. 3.

### Data Generation Parameters

> `[Assumption]` All numeric parameters below are our design choices. The paper does not disclose dataset sizes.

- **N_customers**: 10,000
- **N_accounts**: ~15,000 (1-3 per customer)
- **N_transactions**: ~500,000 (20-100 per account, spanning 24 months)
- **N_applications**: ~8,000 (not all customers apply)
- **Default rate**: ~5-8% (realistic for unsecured consumer lending)
- **Prediction horizon**: 12 months `[Paper §2.2.3, Fig. 3]`

### Generating Realistic Default Signal

> `[Assumption]` The latent risk score design is entirely our construction. The paper uses real observed defaults. We engineer the synthetic default signal to ensure DFS-generated features have genuine (but not trivially perfect) predictive power. The risk factors below are chosen to be plausible credit risk drivers consistent with the paper's mention of "transactional data and user behaviour" `[Paper Abstract]`.

The synthetic default probability for each application will be driven by a latent risk score that is a function of:
- Customer age (U-shaped: very young and very old = higher risk)
- Income band (lower income = higher risk)
- Employment status (unemployed = higher risk)
- Transaction behaviour: average balance (lower = higher risk), spending volatility (higher = higher risk), salary regularity (irregular = higher risk), proportion of gambling/entertainment spending (higher = higher risk)
- Account tenure (newer accounts = higher risk)

This latent score is passed through a logistic function with added noise to produce the binary default outcome. This ensures the DFS-generated features will have genuine predictive power while not being trivially perfect.

---

## Phase 2: Deep Feature Synthesis (`features/dfs_engine.py`)

### Goal

> `[Paper §2.2.4]` DFS is the core feature generation method described in the paper: "Deep Feature Synthesis (DFS) is an automated feature engineering technique that generates features from relational data by stacking simple mathematical operations (e.g., sum, average, etc.) across paths in a set of relational entities."

Use the `featuretools` library to automatically generate features at the `credit_applications` level by traversing the relational schema.

> `[Paper Ref. 11]` The paper cites featuretools (Alteryx, 2025) as the DFS implementation: "Available at: https://featuretools.alteryx.com/en/stable/index.html". Using `featuretools` directly follows this reference.

### Implementation

1. **Entity Set Construction** `[Paper §2.2.4]`: Define entities and relationships. The paper describes DFS operating on "a collection of interconnected entities and their corresponding data tables" with features generated via forward and backward relationships.
   ```
   credit_applications (target)
     └── customer_id -> customers
                          └── customer_id -> accounts
                                              └── account_id -> transactions
   ```

2. **Cutoff Times** `[Paper §2.2.3]`: For each application, the cutoff time is `application_date`. Only transactions/data **before** this date are used (prevents data leakage). The paper discusses censoring and the importance of the prediction horizon window, which implies temporal filtering of input features.

3. **DFS Configuration**:
   - **Depth**: 2 `[Paper §2.2.4]` — The paper describes recursive feature generation: "rfeat and dfeat features for a target entity are synthesized using features from its backward and forward related entities [...] This recursive process terminates when a predetermined depth is reached." Depth=2 gives relational features like `AVG(transactions.amount)` and deep features like `AVG(accounts.SUM(transactions.amount))`.
   - **Aggregation primitives**: `sum`, `mean`, `std`, `min`, `max`, `count`, `trend`, `percent_true`, `num_unique`, `skew` — `[Paper §2.2.4]` mentions "mathematical functions (e.g., min, max, sum, count)" as examples. `[Assumption]` The full list of primitives (trend, percent_true, num_unique, skew) is our extension beyond the paper's examples, using what featuretools provides.
   - **Transform primitives**: `month`, `year`, `weekday`, `is_weekend` (for timestamps) — `[Paper §2.2.4]` mentions "entity features (efeat) are computed for each entry within a single entity table, often involving [...] extracting components from a timestamp." `[Assumption]` Specific transform choices are ours.
   - **Where primitives** (interesting values): Filter transactions by `category` (e.g., SUM of amount WHERE category = 'Travel') and by `transaction_state` (COMPLETED only) — `[Paper §3.1, Fig. 9]` The paper's Fig. 9 shows a feature `SUM(ft_table_completed_tx_cards.amount_gbp_cards WHERE budget_category = TRAVEL)`, demonstrating WHERE-clause filtered aggregations in practice.
   - **Max features**: Cap at ~500 to keep tractable — `[Assumption]` The paper does not specify a cap; this is a pragmatic choice.

4. **Output**: A flat DataFrame with `application_id` as index and ~200-500 auto-generated columns.

---

## Phase 3: Variable Preprocessing (`features/binning.py`, `features/variable_config.py`)

> `[Paper §2.2.5, §2.2.6]` The paper describes coarse binning and WoE transformation as standard preprocessing steps.

We use the **`optbinning`** package (`OptimalBinning` class) for binning and WoE transformation. It handles decision-tree-based optimal binning, monotonicity constraints, bin merging, WoE calculation, and IV computation in one pass.

> `[Assumption]` The specific choice of the `optbinning` library is ours. The paper describes the methodology (decision-tree-based binning, bin merging, WoE) `[Paper §2.2.5, §2.2.6]` but does not name a specific Python package.

### 3a. Per-Variable Configuration (`features/variable_config.py`)

> `[Assumption]` The entire per-variable configuration system (LLM-assisted and default modes) is our engineering design. The paper does not discuss tooling for variable configuration. The paper does mention that monotonicity is enforced during binning ("If two contiguous bins present a reversal in trend, they are merged" `[Paper §2.2.5]`), which motivates the need for per-variable monotonic trend settings.

Since DFS generates features dynamically, we can't hard-code per-variable configs upfront. The pipeline supports two modes:

**Mode A -- LLM-assisted (optional):** After DFS generates feature names, we send them to **Claude Sonnet** (`claude-sonnet-4-5-20250929`) via the Anthropic API with structured outputs using **Pydantic** models to guarantee a valid, typed response.

> `[Assumption]` LLM-assisted configuration, Claude Sonnet, and Pydantic structured outputs are entirely our design. The paper mentions LLMs only in the context of future research for feature generation `[Paper §1.2.1]`, not for configuration.

Pydantic schema:

```python
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class MonotonicTrend(str, Enum):
    ascending = "ascending"
    descending = "descending"
    auto_asc_desc = "auto_asc_desc"
    peak = "peak"
    valley = "valley"

class VariableBinningConfig(BaseModel):
    """Binning configuration for a single DFS-generated feature."""
    feature_name: str = Field(description="Exact DFS feature name")
    dtype: Literal["numerical", "categorical"] = Field(
        description="Data type: numerical for continuous/count features, categorical for string/enum features"
    )
    monotonic_trend: MonotonicTrend = Field(
        description="Expected monotonic relationship with default risk"
    )
    max_n_bins: int = Field(ge=2, le=10, description="Maximum number of bins")
    rationale: str = Field(
        description="Brief credit risk rationale for the chosen monotonicity direction"
    )

class VariableConfigResponse(BaseModel):
    """LLM response containing binning config for all features."""
    variables: list[VariableBinningConfig]
```

The LLM is prompted with:

> "You are a credit risk expert. For each feature below, infer the optimal binning parameters based on the feature name and its likely relationship with credit default risk."

We use the Anthropic SDK's structured output support to pass the `VariableConfigResponse` Pydantic model, ensuring the response is always valid JSON matching the schema -- no manual parsing needed.

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json", "schema": VariableConfigResponse.model_json_schema()},
)
config = VariableConfigResponse.model_validate_json(response.content[0].text)
```

The validated Pydantic response is saved to `outputs/variable_config.yaml` for human review and reproducibility. The `rationale` field documents why each monotonicity direction was chosen.

**Mode B -- Defaults (no LLM):** If the user sets `USE_LLM_CONFIG=False` in `config.py`, all features use these defaults:

| Parameter | Default | Source |
|---|---|---|
| `dtype` | `"numerical"` (auto-detected from pandas dtype) | `[Assumption]` |
| `monotonic_trend` | `"auto_asc_desc"` (optbinning auto-detects best direction) | `[Assumption]` Inspired by `[Paper §2.2.5]` which enforces monotonicity via bin merging |
| `max_n_bins` | `10` | `[Assumption]` |
| `min_bin_size` | `0.05` | `[Paper §2.2.5]` "ensuring each bin contains a minimum percentage (e.g., 5%) of observations" |
| `min_prebin_size` | `0.02` | `[Assumption]` |
| `solver` | `"cp"` (constraint programming) | `[Assumption]` optbinning-specific parameter |
| `divergence` | `"iv"` (optimise for information value) | `[Paper §2.3.2]` IV is the univariate metric used throughout |
| `special_codes` | `None` | `[Assumption]` |

These defaults are sensible for credit risk -- `"auto_asc_desc"` lets optbinning try both ascending and descending monotonic trends and pick the one with higher IV.

### 3b. Generated Config File (`outputs/variable_config.yaml`)

> `[Assumption]` The YAML config file format and the load-on-subsequent-run behaviour are our engineering choices for reproducibility and human-in-the-loop review. The paper mentions that "the model developer will review the outcome of the process" `[Paper §2.3.2]` — this config file facilitates that review.

Regardless of mode, the pipeline always exports the final config used to a YAML file:

```yaml
# Auto-generated by pipeline. Edit and re-run to override.
_defaults:
  dtype: numerical
  monotonic_trend: auto_asc_desc
  max_n_bins: 10
  min_bin_size: 0.05

variables:
  MEAN(transactions.amount):
    monotonic_trend: descending   # higher avg spend = lower risk
    max_n_bins: 8
  STD(transactions.amount):
    monotonic_trend: ascending    # higher volatility = higher risk
  COUNT(transactions WHERE category = Salary):
    monotonic_trend: descending   # more salary deposits = lower risk
  # ... one entry per DFS feature
```

On subsequent runs, if this file exists, the pipeline loads it instead of calling the LLM again. The user can edit it manually between runs.

### 3c. Binning & WoE (`features/binning.py`)

> `[Paper §2.2.5]` Coarse binning: "A decision-tree-based approach is used to split the numeric part of a feature, minimising Gini impurity within each bin." Bins are merged based on statistical tests or trend reversals.
>
> `[Paper §2.2.6]` WoE transformation: "WoE replaces each bin with a single numeric value that represents the log-odds ratio of bad to good outcomes within that bin." Formula: `WoE(X=a) = log(P(X=a|Bad) / P(X=a|Good))`.

For each feature, using `optbinning.OptimalBinning`:

```python
from optbinning import OptimalBinning

optb = OptimalBinning(
    name=feature_name,
    dtype=config["dtype"],
    monotonic_trend=config["monotonic_trend"],  # [Paper §2.2.5] monotonicity enforcement
    max_n_bins=config["max_n_bins"],
    min_bin_size=config["min_bin_size"],         # [Paper §2.2.5] "minimum percentage (e.g., 5%)"
    solver="cp",                                 # [Assumption] optbinning-specific
    divergence="iv",                             # [Paper §2.3.2] optimise for IV
)
optb.fit(X_train[feature_name], y_train)
```

After fitting, we extract:
- **WoE-transformed values**: `optb.transform(X, metric="woe")` — `[Paper §2.2.6]`
- **Binning table**: `optb.binning_table.build()` -- includes bin edges, counts, event rates, WoE, IV per bin
- **Total IV**: `optb.binning_table.build()["IV"].sum()` — `[Paper §2.3.2]`
- **Fitted binning object**: Saved for applying to test/OOT data

**Output**: A WoE-transformed feature matrix + a dict of fitted `OptimalBinning` objects (for scoring new data).

---

## Phase 4: Feature Selection (`selection/`)

### 4a. Information Value Calculation (`information_value.py`)

> `[Paper §2.3.2]` IV formula and interpretation thresholds are directly from the paper, which cites Siddiqi [2] and Scallan [5].

IV is computed directly by `optbinning` during the binning step (Phase 3). Each fitted `OptimalBinning` object provides `optb.binning_table.build()` which includes per-bin IV and the total IV.

For reference, the formula `[Paper §2.3.2, Ref. 2, 5]`:
```
IV(X) = SUM over bins [ (P(bin|Bad) - P(bin|Good)) * WoE(bin) ]
```

Interpretation thresholds `[Paper §2.3.2, Ref. 2]`:
| IV Range | Predictive Power |
|---|---|
| <= 0.02 | Poor |
| 0.02 - 0.1 | Weak |
| 0.1 - 0.3 | Medium |
| 0.3 - 0.5 | Strong |
| > 0.5 | Very Strong (suspect) |

`information_value.py` collects IV from all fitted binning objects, ranks features, and pre-filters those with IV < 0.02 before MIV selection. `[Paper §2.3.2]` The paper uses 2% as an example threshold.

### 4a-bis. Bivariate Analysis (`selection/information_value.py`)

> `[Assumption]` The bivariate analysis table (univariate Gini per feature alongside IV) is our addition. The paper computes IV `[Paper §2.3.2]` and Gini `[Paper §2.4.1]` but does not explicitly describe a per-feature bivariate analysis table. This is standard credit risk modelling practice.

Before MIV selection, produce a full bivariate analysis table for **every** binned feature. For each feature, compute:

| Metric | How | Source |
|---|---|---|
| **IV** | From `optb.binning_table.build()["IV"].sum()` | `[Paper §2.3.2]` |
| **Gini (univariate)** | Fit single-variable logistic regression (WoE of that feature only) vs target, compute `2 * AUC - 1` | `[Assumption]` Standard practice; Gini formula from `[Paper §2.4.1]` |

Output a sorted DataFrame and write it to the markdown report:

```markdown
## 3b. Bivariate Analysis — Univariate Gini & IV (all features)
| # | Feature | IV | IV Strength | Univariate Gini | Bins |
|---|---|---|---|---|---|
| 1 | MEAN(transactions.amount) | 0.312 | Strong | 0.418 | 6 |
| 2 | STD(transactions.amount) | 0.189 | Medium | 0.337 | 5 |
| 3 | COUNT(transactions WHERE category=Salary) | 0.154 | Medium | 0.291 | 4 |
| ... | ... | ... | ... | ... | ... |
| 142 | MIN(accounts.open_date.YEAR) | 0.021 | Weak | 0.062 | 3 |
--- features below IV threshold (0.02) excluded ---
| 143 | MAX(transactions.amount WHERE category=Other) | 0.018 | Poor | 0.041 | 2 |
| ... | ... | ... | ... | ... | ... |
```

Also log a summary: `INFO | bivariate | 347 features analysed: 42 Strong, 68 Medium, 32 Weak, 205 Poor`

### 4b. Marginal Information Value Selection (`miv_selector.py`)

> `[Paper §2.3.2]` The MIV algorithm is the paper's core feature selection method. The formula, greedy forward selection logic, stopping criteria, and correlation filter are all described in §2.3.2.

Greedy forward selection algorithm:

```
ALGORITHM MIV_Selection:
  INPUT: WoE-transformed features F, target y, train/test split
  OUTPUT: selected feature set S, trained models at each step

  1. Compute IV for all features in F                              [Paper §2.3.2]
  2. Remove features with IV < 0.02                                [Paper §2.3.2] "e.g. 2%"
  3. Select f* = feature with highest IV; add to S                 [Paper §2.3.2] "begins by selecting the feature with the highest individual IV"
  4. Train logistic regression on S (train set)                    [Paper §2.4.2]
  5. Record AUC on train and test sets                             [Paper §2.4.1]

  6. REPEAT:
     a. For each candidate feature f_i not in S:
        i.   Compute WoE_expected for each bin of f_i              [Paper §2.3.2] "WoE_expected is calculated based on the current model's scores"
             using current model's predicted probabilities
             - For each bin of f_i, WoE_expected = ln(P(bin|Bad_predicted) / P(bin|Good_predicted))
        ii.  Compute MIV(f_i) = SUM over bins [                   [Paper §2.3.2] MIV formula
               (P(bin|Bad) - P(bin|Good)) * (WoE_observed - WoE_expected)
             ]
        iii. Check Pearson correlation of f_i with all features    [Paper §2.3.2] "pairwise Pearson correlation threshold (e.g. 40%-60%)"
             in S; skip if max |corr| > 0.6                        [Assumption] We use 0.6 (upper end of paper's 40-60% range)
     b. Select f* = feature with highest MIV                       [Paper §2.3.2] "the feature with the highest MIV is added"
     c. If MIV(f*) < threshold (0.02) -> STOP                     [Paper §2.3.2] "MIV falls below a set threshold (e.g. 2%)"
     d. Add f* to S
     e. Retrain logistic regression on S                           [Paper §2.4.2]
     f. Record AUC on train and test                               [Paper §2.4.1, Fig. 5]
     g. If AUC on test has not improved for 2 consecutive          [Paper §2.3.2] "model performance on a test set plateaus"
        steps -> STOP                                              [Assumption] "2 consecutive steps" is our specific patience parameter

  7. RETURN S, final model
```

**Key detail for WoE_expected** `[Paper §2.3.2]`: At each iteration, use the current model's predicted log-odds to compute what the WoE of each candidate feature's bins *would be* if the candidate was already explained by the model. The MIV measures the residual information not yet captured.

### Visualisations `[Paper Fig. 5]`:
- Plot 1: AUC (train & test) vs. step number — replicates the top panel of Fig. 5
- Plot 2: Bar chart of MIV value at each step (log scale y-axis) — replicates the bottom panel of Fig. 5

---

## Phase 5: Model Training (`model/scorecard.py`)

### Rank-Ordering Scorecard

> `[Paper §2.1, §2.4.2]` The paper describes a "rank-ordering application scorecard" as the first component of the PD model `[Paper §2.1]`, and states that "the MIV feature selection technique is theoretically related to logistic regression" `[Paper §2.4.2]`, with benchmarking showing logistic regression performs on par with or better than tree ensembles.

1. **Model**: `statsmodels.api.Logit` on WoE-transformed selected features.

   > `[Paper §2.4.2]` Logistic regression is the paper's chosen model. `[Assumption]` Using `statsmodels` specifically (vs sklearn) is our choice to get full statistical inference output (coefficients, std errors, z-values, p-values, CIs, pseudo R², AIC/BIC).

   - Use `sm.add_constant()` to add intercept.
   - Fit with `model.fit(disp=True)` to get full MLE output.
   - Extract and log the **full model summary** via `result.summary()` and `result.summary2()`.
   - The statsmodels summary provides: coefficients, std errors, z-values, p-values, confidence intervals, pseudo R-squared, log-likelihood, AIC/BIC -- all written verbatim into the markdown report.

   ```python
   import statsmodels.api as sm

   X_woe_const = sm.add_constant(X_woe_train[selected_features])
   logit_model = sm.Logit(y_train, X_woe_const)
   result = logit_model.fit()

   # Full summary -> markdown report & log
   logger.info("\n" + str(result.summary()))
   report.add_section("5. Final Model", f"```\n{result.summary()}\n```")
   ```

2. **Coefficient sign check** `[Paper §2.2.6]`: All WoE feature coefficients should be **positive** (since WoE is defined as `ln(P(Bad)/P(Good))` `[Paper §2.2.6]`, a higher WoE means more bads, and the model should assign higher default probability). Any negative coefficient is flagged with a warning in the log and report. `[Assumption]` The specific sign-check validation is our quality control step; the paper implies this via WoE's definition.

3. **Statistical significance check** `[Assumption]`: Flag features where p-value > 0.05 in the report. This is standard statistical modelling practice not explicitly discussed in the paper.

4. **Score conversion** `[Paper §2.1, Ref. 2, 3]`: Convert log-odds to a points-based scorecard. This follows standard scorecard methodology from Siddiqi [2] and Refaat [3], cited by the paper.
   ```
   Score = Offset + Factor * ln(odds)
   ```
   Where `Factor = PDO / ln(2)` and `Offset = Target_Score - Factor * ln(Target_Odds)`.
   Use PDO = 20, Target Score = 600 at odds 50:1. `[Assumption]` Specific PDO/score/odds values are industry convention choices.

### Benchmarking

> `[Paper §2.4.2, Table 1]` The paper benchmarks logistic regression against tree ensembles and reports that "logistic regression models achieved a higher or equal Gini coefficient compared to tree ensembles."

Train additional models for comparison using sklearn:
- `sklearn.ensemble.GradientBoostingClassifier` (tree ensemble) — `[Paper §2.4.2]` mentions "tree ensembles (gradient boosted trees, random forests)"
- `sklearn.ensemble.RandomForestClassifier` — `[Paper §2.4.2]`
- These use the **raw (non-WoE)** DFS features to give tree models their full flexibility. `[Assumption]` Using raw features for tree models (rather than WoE) is our choice to give them maximum flexibility, since tree ensembles handle non-linear relationships natively.
- Compare Gini coefficients on train, test, and OOT samples. `[Paper §2.4.1, Table 1]`
- Results table written to markdown report.

---

## Phase 6: PD Calibration (`model/calibration.py`)

### Goal

> `[Paper §2.1, §2.4.1 "Calibration"]` The paper describes PD calibration as a separate step from rank-ordering: "A separate calibration step is used to transform the rank-ordering scores into accurate PD estimates."

Transform rank-ordering scores into calibrated Probability of Default estimates.

### Methods (both implemented, user chooses):

1. **Platt Scaling** `[Paper §2.4.1, Ref. 10, 12]`: Fit a logistic regression on the model's raw scores vs. observed defaults (on a holdout calibration set). The paper cites Platt [10] and Nehrebecka [12].

2. **Isotonic Regression** `[Paper §2.4.1, Ref. 12]`: Fit `sklearn.isotonic.IsotonicRegression` on scores vs. observed defaults. The paper states: "isotonic regression being preferred for larger datasets, as it can capture non-monotonic relationships without requiring input transformations."

### Evaluation:
- **Calibration plot** `[Paper §2.4.1, Fig. 7]`: x-axis = score quantile, y-axis = observed default rate vs. predicted PD. Replicates paper Fig. 7.
- **Brier Score** `[Paper §2.4.1, Ref. 13]`: `(1/N) * SUM((y_i - p_hat_i)^2)` -- lower is better. Formula given explicitly in the paper citing Brier [13].

---

## Phase 7: Evaluation (`evaluation/metrics.py`)

### Metrics to Implement

1. **Gini Coefficient** `[Paper §2.4.1]`: `Gini = 2 * AUC - 1` (derived from Lorenz curve). The paper states: "The primary metric for this is the Gini coefficient, which is derived from the Lorenz curve."
2. **ROC AUC** `[Paper Fig. 5]`: `sklearn.metrics.roc_auc_score`. The paper's Fig. 5 plots "ROC AUC values for models at Step."
3. **Lorenz Curve** `[Paper §2.4.1, Fig. 6]`: Plot replicating paper Fig. 6 (x = score quantile, y = cumulative fraction of bads).
4. **Brier Score** `[Paper §2.4.1, Ref. 13]`: As defined above.
5. **Calibration Curve** `[Paper §2.4.1, Fig. 7]`: Predicted vs. observed default rate by decile.
6. **KS Statistic** `[Assumption]`: Maximum separation between cumulative distributions of goods and bads. Standard credit scoring metric not explicitly mentioned in the paper but widely used alongside Gini.

### Data Splits

> `[Paper §2.1, §2.4.1]` The paper mentions "development, hold-out/test and out-of-time samples" `[Paper §2.1 item 1b]` and that "performance must be stable across time on development, test, and out-of-time samples" `[Paper §2.4.1]`. `[Assumption]` The specific 60/20/20 ratios and time-based splitting are our choices; the paper does not specify ratios.

- **Train**: 60% of applications (by date, not random -- earlier applications)
- **Test**: 20% (middle period)
- **Out-of-time (OOT)**: 20% (most recent applications)

Report all metrics on all three splits.

---

## Phase 8: Automated Feature Monitoring (`monitoring/residual_monitor.py`)

### Goal

> `[Paper §3]` This phase replicates the "Automated Value-Adding Feature Monitoring" system described in Section 3 of the paper: "The combination of DFS and MIV permits to establish an automated risk-splitting, value-adding feature monitoring along the lines of the famed, but highly manual, residual monitoring (ReMo) approach."

### Implementation

> `[Paper §3.1]` The paper describes the system architecture: "the orchestrator calculates the MIV for the full set of generated features against a selected delinquency target."

1. Take the final trained scorecard from Phase 5.
2. Score all observations with the scorecard to get risk grades (e.g., quintile buckets: Very Low / Low / Average / High). `[Paper §3.1, Fig. 9]` Fig. 9 shows coarse risk segments: "Very low", "Low", "Average", "High".
3. For each candidate feature not in the current model:
   a. Compute its MIV against the current model's scores. `[Paper §3.1]`
   b. Within each risk grade, compute the bad rate for different levels of the candidate feature. `[Paper §3.1, Fig. 9]`
4. Produce a report ranking all candidate features by their MIV. `[Paper §3.1]`
5. **Visualisation** `[Paper Fig. 9]`: Bar chart showing bad rate by risk grade, segmented by candidate feature bins. Replicates paper Fig. 9 which shows "the observed early (14 days past due in the first 3 months-on-book) delinquency rate" by coarse risk segment, with bars grouped by feature value ranges.

---

## Phase 9: End-to-End Pipeline (`pipeline.py`)

> `[Paper §2, Fig. 1]` The overall pipeline sequence follows the paper's "PD Model Development Lifecycle" (Fig. 1): Data Collection → Entity Set Creation → Feature Generation → WoE Transformation → IV Calculation → Feature Selection → Rank-ordering Model Training → PD Calibration.

Orchestrate all steps:

```python
def run_pipeline():
    # 1. Generate synthetic data                              [Assumption]
    # 2. Build entity set & run DFS                           [Paper §2.2.4]
    # 3. Preprocess (bin + WoE transform)                     [Paper §2.2.5, §2.2.6]
    # 4. Compute IV, bivariate analysis, filter weak features [Paper §2.3.2] + [Assumption]
    # 5. Run MIV forward selection                            [Paper §2.3.2]
    # 6. Train logistic regression scorecard (statsmodels)    [Paper §2.4.2] + [Assumption]
    # 7. Benchmark against tree ensembles                     [Paper §2.4.2, Table 1]
    # 8. Calibrate PD                                         [Paper §2.4.1]
    # 9. Evaluate on train/test/OOT                           [Paper §2.4.1]
    # 10. Run residual monitoring on held-out features        [Paper §3]
    # 11. Generate all plots and write pipeline_report.md     [Assumption]
```

---

## Phase 10: Jupyter Walkthrough (`notebooks/walkthrough.ipynb`)

> `[Assumption]` The interactive notebook is our addition for educational purposes. The paper does not include a notebook.

Interactive notebook with section-by-section execution and rich visualisations:
1. Data exploration & entity relationship diagram
2. DFS output inspection (sample features, feature names) — `[Paper §2.2.4]`
3. Binning & WoE visualisation for top features — `[Paper §2.2.5, §2.2.6]`
4. IV distribution histogram — `[Paper §2.3.2]`
5. Bivariate analysis table (IV + univariate Gini for all features) — `[Assumption]`
6. MIV selection process (AUC curve + MIV bar chart) — `[Paper Fig. 5]`
7. Final scorecard coefficients & statsmodels summary interpretation — `[Paper §2.4.2]` + `[Assumption]`
8. Lorenz curve, calibration curve, ROC curve — `[Paper Fig. 6, Fig. 7, Fig. 5]`
9. Benchmark comparison table — `[Paper Table 1]`
10. Residual monitoring example — `[Paper §3, Fig. 9]`

---

## Python Dependencies (`requirements.txt`)

> `[Assumption]` All library choices except featuretools `[Paper Ref. 11]` are our implementation decisions.

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
statsmodels>=0.14               # [Assumption] chosen for full statistical inference output
featuretools>=1.28              # [Paper Ref. 11] cited directly by the paper
optbinning>=0.19                # [Assumption] implements binning methodology from [Paper §2.2.5]
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
pyyaml>=6.0
anthropic>=0.40                 # [Assumption] Optional: only needed if USE_LLM_CONFIG=True
pydantic>=2.0                   # [Assumption] Optional: only needed if USE_LLM_CONFIG=True
jupyter>=1.0
```

---

## Key Configuration Parameters (`config.py`)

> `[Assumption]` Unless otherwise noted, specific default values are our choices. The configuration system itself is our engineering design.

| Parameter | Default | Description | Source |
|---|---|---|---|
| **Data** | | | |
| `GENERATE_NEW_DATA` | `True` | `True` = generate fresh synthetic data; `False` = load from `DATASET_PATH` | `[Assumption]` |
| `DATASET_PATH` | `"data/datasets/synthetic_v1"` | Folder to save to (if generating) or load from (if reusing) | `[Assumption]` |
| `RANDOM_SEED` | 42 | Reproducibility | `[Assumption]` |
| `N_CUSTOMERS` | 10,000 | Number of synthetic customers | `[Assumption]` |
| `DEFAULT_RATE` | 0.06 | Target default rate | `[Assumption]` |
| `DFS_DEPTH` | 2 | Deep feature synthesis depth | `[Paper §2.2.4]` recursive depth |
| `DFS_MAX_FEATURES` | 500 | Max features from DFS | `[Assumption]` |
| **Binning** | | | |
| `BINNING_MONOTONIC_TREND` | `"auto_asc_desc"` | Default monotonicity constraint | `[Assumption]` inspired by `[Paper §2.2.5]` |
| `BINNING_MAX_N_BINS` | 10 | Max bins per feature | `[Assumption]` |
| `BINNING_MIN_BIN_SIZE` | 0.05 | Min fraction of observations per bin | `[Paper §2.2.5]` "e.g., 5%" |
| `BINNING_SOLVER` | `"cp"` | Solver (constraint programming) | `[Assumption]` optbinning-specific |
| `BINNING_DIVERGENCE` | `"iv"` | Optimisation criterion | `[Paper §2.3.2]` |
| **LLM variable config** | | | |
| `USE_LLM_CONFIG` | `False` | Whether to call LLM for per-variable optbinning params | `[Assumption]` |
| `LLM_MODEL` | `"claude-sonnet-4-5-20250929"` | Model to use for variable config | `[Assumption]` |
| `VARIABLE_CONFIG_PATH` | `"outputs/variable_config.yaml"` | Path to generated/loaded config | `[Assumption]` |
| **Feature selection** | | | |
| `IV_THRESHOLD` | 0.02 | Minimum IV to keep feature | `[Paper §2.3.2]` "e.g. 2%" |
| `MIV_THRESHOLD` | 0.02 | Minimum MIV to add feature | `[Paper §2.3.2]` "e.g. 2%" |
| `CORRELATION_THRESHOLD` | 0.6 | Max pairwise correlation allowed | `[Paper §2.3.2]` "e.g. 40%-60%"; `[Assumption]` we use upper end |
| `MAX_FEATURES` | 20 | Max features in final model | `[Assumption]` |
| **Scorecard** | | | |
| `PDO` | 20 | Points to double the odds | `[Assumption]` industry convention, per Siddiqi `[Paper Ref. 2]` |
| `BASE_SCORE` | 600 | Base score at target odds | `[Assumption]` industry convention |
| `BASE_ODDS` | 50 | Target odds (good:bad) | `[Assumption]` industry convention |
| **Data splits** | | | |
| `TRAIN_RATIO` | 0.6 | Training set proportion | `[Assumption]` paper mentions dev/test/OOT `[Paper §2.1]` but not ratios |
| `TEST_RATIO` | 0.2 | Test set proportion | `[Assumption]` |
| `OOT_RATIO` | 0.2 | Out-of-time proportion | `[Assumption]` |

---

## Paper References Index

For convenience, the key paper sections referenced throughout this document:

| Section | Topic | Key content used |
|---|---|---|
| §1.1 | Business Motivation | Multi-jurisdictional context, automated pipeline motivation |
| §1.2.1 | Feature Generation (Related Work) | DFS vs GP vs LLMs; LLMs noted as future research |
| §2.1 | Model Structure | Rank-ordering scorecard + PD calibration split; dev/test/OOT samples |
| §2.2.1 | Target Variable Definition | Bad/default event + prediction horizon |
| §2.2.2 | Bad Definition vs DoD | Terminal event criterion, roll rate analysis |
| §2.2.3 | Prediction Horizon and Censoring | Cumulative bad rate curves, censoring exclusion |
| §2.2.4 | DFS Feature Generation | Entity sets, efeat/dfeat/rfeat, recursive depth, primitives |
| §2.2.5 | Coarse Binning | Decision-tree split, 5% min bin size, bin merging, monotonicity |
| §2.2.6 | WoE Transformation | Formula, interpretation of positive/negative WoE |
| §2.3.2 | MIV Feature Selection | IV formula, IV thresholds, MIV formula, greedy forward selection, stopping criteria, correlation threshold |
| §2.4.1 | Performance Evaluation | Gini/Lorenz curve, calibration (Platt/isotonic), Brier score |
| §2.4.2 | Model Choice & Benchmarking | LR vs tree ensembles, Table 1 benchmarks |
| §3 | Automated Feature Monitoring | Residual monitoring system, MIV vs incumbent scorecard |
| §3.1 | Background and Architecture | Orchestrator, risk grades, Fig. 8 architecture, Fig. 9 example |
| §4.1 | Case Study: Personal Loans | Transactional data, travel spending, 30% sales uplift |
| §4.2 | Case Study: Credit Cards | RAROC improvement |
| Fig. 1 | PD Model Development Lifecycle | End-to-end pipeline schematic |
| Fig. 3 | Cumulative Default Rate Curves | Bad rate curves by DPD threshold and MoB |
| Fig. 5 | MIV Stepwise Selection | AUC vs step + MIV bar chart |
| Fig. 6 | Lorenz Curve | Score quantile vs cumulative bads |
| Fig. 7 | Calibration Curve | Score quantile vs default rate |
| Fig. 9 | Residual Monitoring Example | Bad rate by risk grade, segmented by feature bins |
| Table 1 | Benchmarking Results | Tree Ensemble vs Logistic Regression Gini |
| Ref. 1 | Kanter & Veeramachaneni (2015) | Original DFS paper |
| Ref. 2 | Siddiqi (2017) | Credit scoring reference (IV thresholds, WoE, scorecard methodology) |
| Ref. 5 | Scallan (2011) | MIV / IV methodology |
| Ref. 10 | Platt (1999) | Platt Scaling |
| Ref. 11 | Alteryx/featuretools (2025) | DFS library |
| Ref. 12 | Nehrebecka (2017) | PD calibration methodology |
| Ref. 13 | Brier (1950) | Brier Score |
| Ref. 14 | Clemons & Thatcher (1998) | Residual monitoring (ReMo) |

---

## Verification Plan

> `[Assumption]` The verification plan is our quality assurance design. Expected value ranges are informed by the paper's benchmarks (e.g., Gini >= 50% `[Paper §2.4.1]`, LR within 5% of tree ensemble Gini `[Paper §2.4.2, Table 1]`).

1. **Unit tests**: Verify WoE/IV/MIV calculations on small hand-crafted examples
2. **Pipeline smoke test**: Run `pipeline.py` end-to-end, confirm it completes without error
3. **Sanity checks**:
   - Gini on training set should be > 40% (synthetic data has embedded signal) — informed by `[Paper §2.4.1]` "at least 50%" for real data; we set lower bar for synthetic
   - IV of top features should be in 0.1-0.5 range — `[Paper §2.3.2]` IV thresholds
   - MIV should decrease monotonically across selection steps — `[Paper Fig. 5]`
   - Logistic regression coefficients should all have consistent sign — `[Paper §2.2.6]` WoE definition
   - statsmodels p-values should be < 0.05 for selected features — `[Assumption]`
   - Calibration curve should roughly follow 45-degree line after calibration — `[Paper Fig. 7]`
4. **Benchmark**: Logistic regression Gini should be within 5% of tree ensemble Gini — `[Paper §2.4.2, Table 1]`
5. **Notebook**: Run all cells top-to-bottom, confirm all plots render
6. **Report**: Verify `outputs/pipeline_report.md` contains all sections including full statsmodels summary
