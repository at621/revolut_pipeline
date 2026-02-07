"""Deep Feature Synthesis wrapper using featuretools.

[Paper §2.2.4] DFS is the core feature generation method: "an automated
feature engineering technique that generates features from relational data
by stacking simple mathematical operations across paths in a set of
relational entities."
"""
from __future__ import annotations

import logging
import time
import warnings

import featuretools as ft
import numpy as np
import pandas as pd
from woodwork.logical_types import (
    Boolean, Categorical, Datetime, Double, Integer,
)

from revolut_credit_risk import config
from revolut_credit_risk.data.synthetic_data import SyntheticData

logger = logging.getLogger(__name__)


def run_dfs(data: SyntheticData) -> pd.DataFrame:
    """Run Deep Feature Synthesis on the relational dataset.

    Parameters
    ----------
    data : SyntheticData
        The relational dataset with all five entity tables.

    Returns
    -------
    pd.DataFrame
        Flat feature matrix indexed by ``application_id``.
    """
    t0 = time.time()

    es = _build_entity_set(data)
    cutoff_times = _build_cutoff_times(data)

    logger.info(
        "Generating features for %d applications (depth=%d)...",
        len(cutoff_times), config.DFS_DEPTH,
    )

    # [Paper §2.2.4] aggregation + transform primitives
    agg_primitives = [
        "sum", "mean", "std", "min", "max", "count",
        "percent_true", "num_unique", "skew",
    ]
    trans_primitives = ["month", "year", "weekday", "is_weekend"]

    # [Paper §3.1, Fig. 9] WHERE-clause filtered aggregations
    es.add_interesting_values(
        dataframe_name="transactions",
        values={
            "category": _TX_CATEGORIES_INTERESTING,
            "transaction_state": ["COMPLETED"],
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress Woodwork dtype warnings on Windows
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="credit_applications",
            cutoff_time=cutoff_times,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            where_primitives=["sum", "mean", "count"],
            max_depth=config.DFS_DEPTH,
            max_features=config.DFS_MAX_FEATURES,
            verbose=False,
        )

    # Clean up: keep only numeric columns, drop all-NaN, fill remaining NaN
    n_before = feature_matrix.shape[1]
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
    feature_matrix = feature_matrix[numeric_cols]
    feature_matrix = feature_matrix.dropna(axis=1, how="all")
    feature_matrix = feature_matrix.fillna(0)

    elapsed = time.time() - t0
    logger.info(
        "DFS complete: %d features generated (%d dropped) in %.1fs",
        feature_matrix.shape[1], n_before - feature_matrix.shape[1], elapsed,
    )
    logger.debug("Sample feature names: %s", list(feature_matrix.columns[:10]))

    return feature_matrix


# [Paper §3.1, Fig. 9] Travel spending shown as predictive
_TX_CATEGORIES_INTERESTING = [
    "Salary", "Travel", "Entertainment", "Groceries", "Rent",
]


def _build_entity_set(data: SyntheticData) -> ft.EntitySet:
    """Construct a featuretools EntitySet from the relational tables.

    [Paper §2.2.4] Entity set with relationships:
        credit_applications -> customers -> accounts -> transactions
    """
    es = ft.EntitySet(id="revolut_credit_risk")

    # Add dataframes with explicit logical types to avoid Woodwork
    # "Could not infer format" warnings on date columns
    es = es.add_dataframe(
        dataframe_name="customers",
        dataframe=data.customers.copy(),
        index="customer_id",
        time_index="signup_date",
        logical_types={
            "signup_date": Datetime,
            "income_band": Categorical,
            "country": Categorical,
            "employment_status": Categorical,
        },
    )
    es = es.add_dataframe(
        dataframe_name="accounts",
        dataframe=data.accounts.copy(),
        index="account_id",
        time_index="open_date",
        logical_types={
            "open_date": Datetime,
            "account_type": Categorical,
            "currency": Categorical,
        },
    )

    tx = data.transactions.copy()
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=tx,
        index="transaction_id",
        time_index="transaction_date",
        logical_types={
            "transaction_date": Datetime,
            "amount": Double,
            "category": Categorical,
            "merchant_name": Categorical,
            "transaction_state": Categorical,
        },
    )

    apps = data.credit_applications.copy()
    es = es.add_dataframe(
        dataframe_name="credit_applications",
        dataframe=apps,
        index="application_id",
        time_index="application_date",
        logical_types={
            "application_date": Datetime,
            "product_type": Categorical,
            "requested_amount": Double,
            "approved": Boolean,
        },
    )

    # Add relationships
    # credit_applications.customer_id -> customers.customer_id
    es = es.add_relationship("customers", "customer_id",
                             "credit_applications", "customer_id")
    # customers.customer_id -> accounts.customer_id
    es = es.add_relationship("customers", "customer_id",
                             "accounts", "customer_id")
    # accounts.account_id -> transactions.account_id
    es = es.add_relationship("accounts", "account_id",
                             "transactions", "account_id")

    logger.debug("EntitySet built: %s", es)
    return es


def _build_cutoff_times(data: SyntheticData) -> pd.DataFrame:
    """Build cutoff-time DataFrame for temporal filtering.

    [Paper §2.2.3] Only data before application_date is used to prevent
    data leakage.
    """
    ct = data.credit_applications[["application_id", "application_date"]].copy()
    ct = ct.rename(columns={"application_date": "time"})
    ct = ct.set_index("application_id", drop=False)
    return ct
