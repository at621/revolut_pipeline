"""Synthetic relational dataset generator.

[Assumption] The paper uses proprietary Revolut data. This entire module is
our design to enable replication without access to real banking data.
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
    transactions = _generate_transactions(rng, accounts)
    credit_applications = _generate_credit_applications(rng, customers)
    loan_performance = _generate_loan_performance(
        rng, customers, accounts, transactions, credit_applications
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
    # [Assumption] 1-3 accounts per customer
    rows: list[dict] = []
    aid = 1
    for _, cust in customers.iterrows():
        n_acc = rng.integers(1, 4)
        for _ in range(n_acc):
            open_offset = rng.integers(0, 180)
            rows.append({
                "account_id": aid,
                "customer_id": cust["customer_id"],
                "account_type": rng.choice(_ACCOUNT_TYPES),
                "open_date": cust["signup_date"] + pd.Timedelta(days=int(open_offset)),
                "currency": rng.choice(_CURRENCIES),
            })
            aid += 1

    df = pd.DataFrame(rows)
    df["account_type"] = pd.Categorical(df["account_type"], categories=_ACCOUNT_TYPES)
    df["currency"] = pd.Categorical(df["currency"], categories=_CURRENCIES)
    logger.info("Generated %d accounts for %d customers", len(df), len(customers))
    return df


def _generate_transactions(
    rng: np.random.Generator, accounts: pd.DataFrame
) -> pd.DataFrame:
    # [Assumption] 20-100 transactions per account over ~24 months
    rows: list[dict] = []
    tid = 1
    for _, acc in accounts.iterrows():
        n_tx = rng.integers(20, 101)
        for _ in range(n_tx):
            tx_offset = rng.integers(0, 730)  # up to 24 months
            category = rng.choice(_TX_CATEGORIES)
            # Salary is always positive; others mostly negative (debits)
            if category == "Salary":
                amount = float(rng.uniform(1500, 6000))
            elif category == "Rent":
                amount = -float(rng.uniform(400, 2000))
            else:
                amount = -float(rng.exponential(50)) if rng.random() < 0.85 else float(rng.uniform(10, 500))

            rows.append({
                "transaction_id": tid,
                "account_id": acc["account_id"],
                "transaction_date": acc["open_date"] + pd.Timedelta(days=int(tx_offset)),
                "amount": round(amount, 2),
                "category": category,
                "merchant_name": rng.choice(_MERCHANTS),
                "transaction_state": rng.choice(_TX_STATES, p=[0.90, 0.05, 0.05]),
            })
            tid += 1

    df = pd.DataFrame(rows)
    df["category"] = pd.Categorical(df["category"], categories=_TX_CATEGORIES)
    df["transaction_state"] = pd.Categorical(
        df["transaction_state"], categories=_TX_STATES
    )
    logger.info("Generated %d transactions", len(df))
    return df


def _generate_credit_applications(
    rng: np.random.Generator, customers: pd.DataFrame
) -> pd.DataFrame:
    # [Assumption] ~80% of customers apply for credit
    n_applicants = int(len(customers) * 0.8)
    applicant_ids = rng.choice(
        customers["customer_id"].values, size=n_applicants, replace=False
    )
    rows: list[dict] = []
    app_id = 1
    for cid in applicant_ids:
        cust = customers.loc[customers["customer_id"] == cid].iloc[0]
        # Application comes after signup
        app_offset = rng.integers(30, 365 * 2)
        rows.append({
            "application_id": app_id,
            "customer_id": int(cid),
            "application_date": cust["signup_date"] + pd.Timedelta(days=int(app_offset)),
            "product_type": rng.choice(["Personal Loan", "Credit Card"]),
            "requested_amount": round(float(rng.uniform(500, 25000)), 2),
            "approved": True,  # we only model approved applications
        })
        app_id += 1

    df = pd.DataFrame(rows)
    logger.info("Generated %d credit applications", len(df))
    return df


def _generate_loan_performance(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    applications: pd.DataFrame,
) -> pd.DataFrame:
    """Generate loan outcomes driven by a latent risk score.

    [Assumption] The latent risk score ensures DFS features have genuine
    (but not trivially perfect) predictive power.
    """
    # Pre-compute per-customer behavioural features for the risk score
    cust_features = _compute_customer_risk_features(
        customers, accounts, transactions, applications
    )

    rows: list[dict] = []
    for _, app in applications.iterrows():
        cid = app["customer_id"]
        feats = cust_features.get(cid, {})

        # --- Latent risk score (higher = riskier) ---
        risk = 0.0

        # Age: U-shaped (very young / very old = riskier)
        age = feats.get("age", 35)
        if age < 25:
            risk += (25 - age) * 0.04
        elif age > 60:
            risk += (age - 60) * 0.03

        # Income band
        income_map = {"Low": 0.6, "Medium": 0.0, "High": -0.3, "Very High": -0.5}
        risk += income_map.get(feats.get("income_band", "Medium"), 0.0)

        # Employment
        emp_map = {
            "Employed": -0.2, "Self-Employed": 0.1,
            "Unemployed": 0.8, "Student": 0.3, "Retired": 0.1,
        }
        risk += emp_map.get(feats.get("employment_status", "Employed"), 0.0)

        # Average balance (lower = riskier)
        avg_bal = feats.get("avg_balance", 0)
        risk -= np.clip(avg_bal / 5000, -1.0, 1.0)

        # Spending volatility (higher = riskier)
        vol = feats.get("spending_volatility", 50)
        risk += np.clip(vol / 200, 0, 0.8)

        # Salary regularity — fraction of months with salary deposit
        salary_reg = feats.get("salary_regularity", 0.5)
        risk -= salary_reg * 0.5

        # Entertainment / gambling fraction
        ent_frac = feats.get("entertainment_fraction", 0.1)
        risk += ent_frac * 1.5

        # Account tenure (months since signup)
        tenure = feats.get("tenure_months", 12)
        if tenure < 6:
            risk += 0.4
        elif tenure < 12:
            risk += 0.1

        # Add noise
        risk += rng.normal(0, 0.5)

        # Convert to default probability via logistic function
        # [Assumption] intercept tuned to achieve target default rate ~5-8%
        p_default = 1 / (1 + np.exp(-(risk - 2.5)))

        is_default = bool(rng.random() < p_default)
        dpd_max = int(rng.integers(90, 180)) if is_default else int(rng.integers(0, 30))

        rows.append({
            "application_id": app["application_id"],
            "months_on_book": int(rng.integers(6, 25)),
            "dpd_max": dpd_max,
            "is_default": is_default,
        })

    df = pd.DataFrame(rows)
    df["is_default"] = df["is_default"].astype(int)

    actual_rate = df["is_default"].mean()
    logger.info(
        "Loan performance: %d records, default rate=%.2f%%",
        len(df), 100 * actual_rate,
    )
    return df


def _compute_customer_risk_features(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    applications: pd.DataFrame,
) -> dict[int, dict]:
    """Pre-compute per-customer behavioural features for the latent risk score."""
    # Index customers
    cust_map: dict[int, dict] = {}
    for _, c in customers.iterrows():
        cust_map[int(c["customer_id"])] = {
            "age": int(c["age"]),
            "income_band": str(c["income_band"]),
            "employment_status": str(c["employment_status"]),
            "signup_date": c["signup_date"],
        }

    # Account-level aggregation
    acc_by_cust = accounts.groupby("customer_id")["account_id"].apply(list).to_dict()

    # Transaction-level aggregation (by account)
    tx_grouped = transactions.groupby("account_id")

    for cid, info in cust_map.items():
        acc_ids = acc_by_cust.get(cid, [])
        all_amounts: list[float] = []
        salary_months: set[str] = set()
        entertainment_total = 0.0
        total_abs = 0.0

        for aid in acc_ids:
            if aid not in tx_grouped.groups:
                continue
            acc_txs = tx_grouped.get_group(aid)
            all_amounts.extend(acc_txs["amount"].tolist())
            # Salary regularity
            sal_txs = acc_txs[acc_txs["category"] == "Salary"]
            for _, stx in sal_txs.iterrows():
                salary_months.add(stx["transaction_date"].strftime("%Y-%m"))
            # Entertainment fraction
            ent_txs = acc_txs[acc_txs["category"].isin(["Entertainment", "Travel"])]
            entertainment_total += ent_txs["amount"].abs().sum()
            total_abs += acc_txs["amount"].abs().sum()

        if all_amounts:
            info["avg_balance"] = float(np.mean(all_amounts))
            info["spending_volatility"] = float(np.std(all_amounts))
        else:
            info["avg_balance"] = 0.0
            info["spending_volatility"] = 50.0

        info["salary_regularity"] = len(salary_months) / 24.0  # over ~24 months
        info["entertainment_fraction"] = (
            entertainment_total / total_abs if total_abs > 0 else 0.1
        )
        # Tenure
        signup = info["signup_date"]
        info["tenure_months"] = max(
            1, (pd.Timestamp("2025-01-01") - signup).days // 30
        )

    return cust_map


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
