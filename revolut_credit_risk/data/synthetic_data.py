"""Synthetic relational dataset generator.

[Assumption] The paper uses proprietary Revolut data. This entire module is
our design to enable replication without access to real banking data.

Design: each customer gets a latent risk_score based on demographics. This
risk_score drives BOTH the transaction patterns (salary size, spending
volatility, entertainment fraction) AND the default outcome. This ensures
DFS-generated features (MEAN, STD, COUNT of transactions) have genuine
predictive power for default, matching the paper's premise that
"transactional data and user behaviour" predict credit risk [Paper §1.1].
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from revolut_credit_risk import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------


@dataclass
class SyntheticData:
    """Container for all five entity DataFrames."""

    customers: pd.DataFrame
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    credit_applications: pd.DataFrame
    loan_performance: pd.DataFrame


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_synthetic_data() -> SyntheticData:
    """Generate a full relational dataset with embedded default signal.

    Returns
    -------
    SyntheticData
        Contains customers, accounts, transactions, credit_applications,
        and loan_performance DataFrames.
    """
    t0 = time.time()
    rng = np.random.default_rng(config.RANDOM_SEED)

    customers = _generate_customers(rng)
    accounts = _generate_accounts(rng, customers)

    # Compute risk scores ONCE — used for both transactions and defaults
    # [Assumption] Shared latent risk score ensures DFS features align with target
    risk_rng = np.random.default_rng(config.RANDOM_SEED + 1)
    risk_scores = _compute_customer_risk_scores(customers, risk_rng)

    # Generate transactions that vary by each customer's risk profile
    transactions = _generate_transactions(rng, accounts, customers, risk_scores)

    credit_applications = _generate_credit_applications(rng, customers)
    loan_performance = _generate_loan_performance(
        rng, customers, credit_applications, risk_scores
    )

    elapsed = time.time() - t0
    logger.info(
        "Synthetic data generated in %.1fs — customers=%d, accounts=%d, "
        "transactions=%d, applications=%d, defaults=%d (%.1f%%)",
        elapsed,
        len(customers),
        len(accounts),
        len(transactions),
        len(credit_applications),
        loan_performance["is_default"].sum(),
        100 * loan_performance["is_default"].mean(),
    )

    return SyntheticData(
        customers=customers,
        accounts=accounts,
        transactions=transactions,
        credit_applications=credit_applications,
        loan_performance=loan_performance,
    )


# ---------------------------------------------------------------------------
# Per-entity generators
# ---------------------------------------------------------------------------

_INCOME_BANDS = ["Low", "Medium", "High", "Very High"]
_COUNTRIES = ["UK", "IE", "LT", "DE", "FR"]  # [Assumption] Revolut's main European markets
_EMPLOYMENT = ["Employed", "Self-Employed", "Unemployed", "Student", "Retired"]
_ACCOUNT_TYPES = ["Current", "Savings", "Trading"]
_CURRENCIES = ["GBP", "EUR", "USD"]
_TX_CATEGORIES = [
    "Groceries", "Travel", "Entertainment", "Utilities", "Salary",
    "Transfer", "Rent", "Shopping", "Restaurants", "Other",
]
_TX_STATES = ["COMPLETED", "PENDING", "DECLINED"]
_MERCHANTS = [
    "Tesco", "Sainsbury", "Amazon", "Uber", "TfL", "Netflix", "Spotify",
    "Shell", "BP", "Booking.com", "Deliveroo", "JustEat", "Zara", "IKEA",
    "Apple", "Google", "EE", "Vodafone", "PayPal", "Revolut",
]


def _compute_customer_risk_scores(customers: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Compute a latent risk score for each customer from demographics.

    [Assumption] This latent score drives both transaction patterns and default.
    Higher score = riskier.  Range roughly -2 to +3.
    """
    n = len(customers)
    risk = np.zeros(n)

    ages = customers["age"].values
    income_bands = customers["income_band"].values.astype(str)
    emp_statuses = customers["employment_status"].values.astype(str)

    # Age: U-shaped (very young / very old = riskier)
    young = ages < 25
    old = ages > 60
    risk[young] += (25 - ages[young]) * 0.06
    risk[old] += (ages[old] - 60) * 0.04

    # Income band
    income_map = {"Low": 0.8, "Medium": 0.0, "High": -0.5, "Very High": -0.8}
    risk += np.array([income_map.get(ib, 0.0) for ib in income_bands])

    # Employment
    emp_map = {
        "Employed": -0.3, "Self-Employed": 0.2,
        "Unemployed": 1.2, "Student": 0.5, "Retired": 0.1,
    }
    risk += np.array([emp_map.get(es, 0.0) for es in emp_statuses])

    # Tenure: newer signups are riskier
    tenure_days = (pd.Timestamp("2025-01-01") - customers["signup_date"]).dt.days.values
    tenure_months = np.maximum(1, tenure_days // 30)
    risk += np.where(tenure_months < 6, 0.5, np.where(tenure_months < 12, 0.2, 0.0))

    # Add persistent per-customer noise (this becomes part of the customer's "type")
    risk += rng.normal(0, 0.3, size=n)

    return risk


def _generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    n = config.N_CUSTOMERS
    # [Assumption] signup dates spanning ~3 years before observation
    signup_dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, size=n), unit="D"
    )
    ages = rng.integers(18, 76, size=n)
    income_probs = [0.25, 0.40, 0.25, 0.10]
    income_bands = rng.choice(_INCOME_BANDS, size=n, p=income_probs)
    country_probs = [0.40, 0.10, 0.10, 0.25, 0.15]
    countries = rng.choice(_COUNTRIES, size=n, p=country_probs)
    emp_probs = [0.55, 0.15, 0.10, 0.10, 0.10]
    employment = rng.choice(_EMPLOYMENT, size=n, p=emp_probs)

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": signup_dates,
        "age": ages,
        "income_band": pd.Categorical(income_bands, categories=_INCOME_BANDS),
        "country": pd.Categorical(countries, categories=_COUNTRIES),
        "employment_status": pd.Categorical(employment, categories=_EMPLOYMENT),
    })


def _generate_accounts(
    rng: np.random.Generator, customers: pd.DataFrame
) -> pd.DataFrame:
    # [Assumption] 1-3 accounts per customer — vectorised
    n = len(customers)
    n_accounts_per = rng.integers(1, 4, size=n)
    total = int(n_accounts_per.sum())

    cust_ids = np.repeat(customers["customer_id"].values, n_accounts_per)
    signup_dates = np.repeat(customers["signup_date"].values, n_accounts_per)
    open_offsets = rng.integers(0, 180, size=total)

    df = pd.DataFrame({
        "account_id": np.arange(1, total + 1),
        "customer_id": cust_ids,
        "account_type": rng.choice(_ACCOUNT_TYPES, size=total),
        "open_date": signup_dates + pd.to_timedelta(open_offsets, unit="D"),
        "currency": rng.choice(_CURRENCIES, size=total),
    })
    df["account_type"] = pd.Categorical(df["account_type"], categories=_ACCOUNT_TYPES)
    df["currency"] = pd.Categorical(df["currency"], categories=_CURRENCIES)
    logger.info("Generated %d accounts for %d customers", len(df), n)
    return df


def _generate_transactions(
    rng: np.random.Generator, accounts: pd.DataFrame,
    customers: pd.DataFrame, risk_scores: np.ndarray,
) -> pd.DataFrame:
    """Generate transactions whose patterns vary by customer risk profile.

    [Assumption] Risky customers have lower salaries, higher spending volatility,
    and more entertainment/travel spending.  This is what makes DFS features
    predictive: e.g., MEAN(transactions.amount) will be lower for risky customers.
    """
    cust_risk = pd.Series(risk_scores, index=customers["customer_id"].values)

    n_acc = len(accounts)
    n_tx_per = rng.integers(20, 101, size=n_acc)
    total = int(n_tx_per.sum())

    acc_ids = np.repeat(accounts["account_id"].values, n_tx_per)
    acc_cust_ids = np.repeat(accounts["customer_id"].values, n_tx_per)
    open_dates = np.repeat(accounts["open_date"].values, n_tx_per)
    tx_offsets = rng.integers(0, 730, size=total)

    # Look up each transaction's customer risk score
    tx_risk = cust_risk.reindex(acc_cust_ids).values

    # Risk-driven category distribution — vectorised
    # Higher risk → more Entertainment/Travel, less Salary
    # [Assumption] Category probabilities shift based on risk
    p_salary = np.clip(0.15 - tx_risk * 0.04, 0.03, 0.20)
    p_ent = np.clip(0.08 + tx_risk * 0.04, 0.04, 0.20)
    p_travel = np.clip(0.06 + tx_risk * 0.03, 0.03, 0.15)
    p_remaining = 1.0 - p_salary - p_ent - p_travel
    p_other = p_remaining / 7.0  # split among 7 other categories

    # Build (total, 10) probability matrix — columns match _TX_CATEGORIES order
    prob_matrix = np.column_stack([
        p_other,   # Groceries
        p_travel,  # Travel
        p_ent,     # Entertainment
        p_other,   # Utilities
        p_salary,  # Salary
        p_other,   # Transfer
        p_other,   # Rent
        p_other,   # Shopping
        p_other,   # Restaurants
        p_other,   # Other
    ])
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

    # Vectorised category sampling via cumulative probabilities
    cdf = np.cumsum(prob_matrix, axis=1)
    u = rng.random(total)[:, np.newaxis]
    cat_indices = (u < cdf).argmax(axis=1)
    cat_array = np.array(_TX_CATEGORIES)
    categories = cat_array[cat_indices]

    # Risk-driven amounts:
    # Lower risk → higher salary, lower spending magnitude
    amounts = np.empty(total)
    is_salary = categories == "Salary"
    is_rent = categories == "Rent"
    is_other = ~is_salary & ~is_rent

    # Salary: low-risk customers earn more
    # [Assumption] Salary range shifts down with risk
    salary_base = np.clip(4000 - tx_risk[is_salary] * 800, 1200, 6000)
    salary_spread = salary_base * 0.3
    amounts[is_salary] = np.maximum(
        800, rng.normal(salary_base, salary_spread)
    )

    # Rent: fairly stable across risk levels
    amounts[is_rent] = -rng.uniform(400, 2000, size=is_rent.sum())

    # Other spending: riskier customers spend more erratically
    n_other = is_other.sum()
    other_risk = tx_risk[is_other]
    is_debit = rng.random(n_other) < 0.85
    # Higher risk → larger average spend
    spend_scale = np.clip(40 + other_risk * 20, 20, 120)
    amounts_other = np.where(
        is_debit,
        -rng.exponential(spend_scale),
        rng.uniform(10, 500, size=n_other),
    )
    amounts[is_other] = amounts_other
    amounts = np.round(amounts, 2)

    # Transaction state: riskier customers have more declined transactions
    # [Assumption] Declined rate increases with risk — vectorised
    state_probs = np.column_stack([
        np.clip(0.92 - tx_risk * 0.03, 0.80, 0.98),  # COMPLETED
        np.full(total, 0.03),                           # PENDING
        np.clip(0.05 + tx_risk * 0.03, 0.02, 0.17),   # DECLINED
    ])
    state_probs = state_probs / state_probs.sum(axis=1, keepdims=True)
    state_cdf = np.cumsum(state_probs, axis=1)
    u_state = rng.random(total)[:, np.newaxis]
    state_indices = (u_state < state_cdf).argmax(axis=1)
    state_array = np.array(_TX_STATES)
    tx_states = state_array[state_indices]

    df = pd.DataFrame({
        "transaction_id": np.arange(1, total + 1),
        "account_id": acc_ids,
        "transaction_date": open_dates + pd.to_timedelta(tx_offsets, unit="D"),
        "amount": amounts,
        "category": categories,
        "merchant_name": rng.choice(_MERCHANTS, size=total),
        "transaction_state": tx_states,
    })
    df["category"] = pd.Categorical(df["category"], categories=_TX_CATEGORIES)
    df["transaction_state"] = pd.Categorical(
        df["transaction_state"], categories=_TX_STATES
    )
    logger.info("Generated %d transactions", len(df))
    return df


def _generate_credit_applications(
    rng: np.random.Generator, customers: pd.DataFrame
) -> pd.DataFrame:
    # [Assumption] ~80% of customers apply for credit — vectorised
    n_applicants = int(len(customers) * 0.8)
    applicant_ids = rng.choice(
        customers["customer_id"].values, size=n_applicants, replace=False
    )
    # Look up signup dates via merge rather than iterrows
    cust_lookup = customers.set_index("customer_id")["signup_date"]
    signup_dates = cust_lookup.loc[applicant_ids].values
    app_offsets = rng.integers(30, 365 * 2, size=n_applicants)

    df = pd.DataFrame({
        "application_id": np.arange(1, n_applicants + 1),
        "customer_id": applicant_ids,
        "application_date": signup_dates + pd.to_timedelta(app_offsets, unit="D"),
        "product_type": rng.choice(["Personal Loan", "Credit Card"], size=n_applicants),
        "requested_amount": np.round(rng.uniform(500, 25000, size=n_applicants), 2),
        "approved": True,
    })
    logger.info("Generated %d credit applications", len(df))
    return df


def _generate_loan_performance(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    applications: pd.DataFrame,
    risk_scores: np.ndarray,
) -> pd.DataFrame:
    """Generate loan outcomes driven by the same latent risk score.

    [Assumption] Default probability is a logistic function of the customer's
    risk score (same score that drove transaction patterns) plus noise.
    This ensures DFS features are genuinely predictive.
    """
    cust_risk = pd.Series(risk_scores, index=customers["customer_id"].values)

    app_cids = applications["customer_id"].values
    n = len(applications)

    # Look up each application's customer risk score
    app_risk = cust_risk.reindex(app_cids).values

    # Add application-level noise (so risk isn't perfectly deterministic)
    app_risk += rng.normal(0, 0.4, size=n)

    # Convert to default probability via logistic function
    # [Assumption] intercept tuned to achieve target default rate ~5-8%
    p_default = 1 / (1 + np.exp(-(app_risk - 3.0)))
    is_default = (rng.random(n) < p_default).astype(int)
    dpd_max = np.where(is_default, rng.integers(90, 180, size=n), rng.integers(0, 30, size=n))

    df = pd.DataFrame({
        "application_id": applications["application_id"].values,
        "months_on_book": rng.integers(6, 25, size=n),
        "dpd_max": dpd_max,
        "is_default": is_default,
    })

    actual_rate = df["is_default"].mean()
    logger.info(
        "Loan performance: %d records, default rate=%.2f%%",
        len(df), 100 * actual_rate,
    )
    return df


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_dataset(data: SyntheticData, path: Path | None = None) -> None:
    """Persist all entity DataFrames as parquet + metadata.json."""
    path = path or config.DATASET_PATH
    path.mkdir(parents=True, exist_ok=True)

    for name in ("customers", "accounts", "transactions",
                 "credit_applications", "loan_performance"):
        df: pd.DataFrame = getattr(data, name)
        df.to_parquet(path / f"{name}.parquet", index=False)

    meta = {
        "generated_at": datetime.now().isoformat(),
        "n_customers": len(data.customers),
        "n_accounts": len(data.accounts),
        "n_transactions": len(data.transactions),
        "n_applications": len(data.credit_applications),
        "n_defaults": int(data.loan_performance["is_default"].sum()),
        "default_rate": float(data.loan_performance["is_default"].mean()),
        "random_seed": config.RANDOM_SEED,
    }
    (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Dataset saved to %s", path)


def load_dataset(path: Path | None = None) -> SyntheticData:
    """Load a previously saved dataset from parquet files."""
    path = path or config.DATASET_PATH
    required = [
        "customers", "accounts", "transactions",
        "credit_applications", "loan_performance",
    ]
    for name in required:
        fpath = path / f"{name}.parquet"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing dataset file: {fpath}")

    data = SyntheticData(
        customers=pd.read_parquet(path / "customers.parquet"),
        accounts=pd.read_parquet(path / "accounts.parquet"),
        transactions=pd.read_parquet(path / "transactions.parquet"),
        credit_applications=pd.read_parquet(path / "credit_applications.parquet"),
        loan_performance=pd.read_parquet(path / "loan_performance.parquet"),
    )

    meta_path = path / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded dataset from %s — %d customers, %d applications, "
            "default_rate=%.2f%%, seed=%s",
            path, meta["n_customers"], meta["n_applications"],
            100 * meta["default_rate"], meta.get("random_seed"),
        )
    else:
        logger.info("Loaded dataset from %s (no metadata.json found)", path)

    return data
