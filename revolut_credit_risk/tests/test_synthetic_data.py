"""Tests for synthetic data generation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from revolut_credit_risk import config
from revolut_credit_risk.data.synthetic_data import (
    SyntheticData,
    generate_synthetic_data,
    save_dataset,
    load_dataset,
)


@pytest.fixture(scope="module")
def small_data(monkeypatch_module):
    """Generate a small dataset for testing (500 customers)."""
    monkeypatch_module.setattr(config, "N_CUSTOMERS", 500)
    return generate_synthetic_data()


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch
    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture
def data():
    """Generate a small dataset for testing."""
    original = config.N_CUSTOMERS
    config.N_CUSTOMERS = 500
    d = generate_synthetic_data()
    config.N_CUSTOMERS = original
    return d


def test_generates_all_entities(data):
    assert isinstance(data, SyntheticData)
    assert isinstance(data.customers, pd.DataFrame)
    assert isinstance(data.accounts, pd.DataFrame)
    assert isinstance(data.transactions, pd.DataFrame)
    assert isinstance(data.credit_applications, pd.DataFrame)
    assert isinstance(data.loan_performance, pd.DataFrame)


def test_customer_count(data):
    assert len(data.customers) == 500


def test_accounts_linked_to_customers(data):
    assert set(data.accounts["customer_id"]).issubset(
        set(data.customers["customer_id"])
    )


def test_transactions_linked_to_accounts(data):
    assert set(data.transactions["account_id"]).issubset(
        set(data.accounts["account_id"])
    )


def test_applications_linked_to_customers(data):
    assert set(data.credit_applications["customer_id"]).issubset(
        set(data.customers["customer_id"])
    )


def test_loan_performance_linked_to_applications(data):
    assert set(data.loan_performance["application_id"]).issubset(
        set(data.credit_applications["application_id"])
    )


def test_target_is_binary(data):
    assert set(data.loan_performance["is_default"].unique()).issubset({0, 1})


def test_default_rate_reasonable(data):
    rate = data.loan_performance["is_default"].mean()
    assert 0.01 < rate < 0.30, f"Default rate {rate:.2%} outside expected range"


def test_save_and_load(data, tmp_path):
    save_dataset(data, tmp_path / "test_dataset")
    loaded = load_dataset(tmp_path / "test_dataset")

    assert len(loaded.customers) == len(data.customers)
    assert len(loaded.accounts) == len(data.accounts)
    assert len(loaded.transactions) == len(data.transactions)
    assert len(loaded.credit_applications) == len(data.credit_applications)
    assert len(loaded.loan_performance) == len(data.loan_performance)
