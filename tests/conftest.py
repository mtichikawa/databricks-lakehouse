"""
conftest.py — Shared pytest fixtures for databricks-lakehouse tests.

Provides:
- temp_dir:          a temporary directory that cleans itself up after each test
- sample_bronze_df:  a realistic bronze-schema DataFrame (pre-metadata-columns)
- sample_raw_df:     raw TLC-schema DataFrame (no metadata columns)
- sample_silver_df:  silver-schema DataFrame (renamed + derived columns)
- dirty_df:          DataFrame with known data quality issues
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def temp_dir(tmp_path):
    """A temporary directory scoped to a single test."""
    return tmp_path


# ---------------------------------------------------------------------------
# Raw / Bronze fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_raw_df() -> pd.DataFrame:
    """
    Raw TLC-schema DataFrame (20 clean rows).
    Matches the schema of the TLC yellow taxi Parquet files.
    """
    base = pd.Timestamp("2023-01-15 08:00:00")
    n = 20
    rng = np.random.default_rng(0)

    pickup = pd.date_range(base, periods=n, freq="5min")
    dropoff = pickup + pd.to_timedelta(rng.integers(300, 1800, n), unit="s")

    return pd.DataFrame({
        "VendorID": rng.choice([1, 2], n),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "passenger_count": rng.integers(1, 4, n).astype(float),
        "trip_distance": rng.uniform(0.5, 15.0, n).round(2),
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": rng.integers(1, 265, n),
        "DOLocationID": rng.integers(1, 265, n),
        "payment_type": rng.choice([1, 2], n),
        "fare_amount": rng.uniform(5.0, 50.0, n).round(2),
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": rng.uniform(0.0, 10.0, n).round(2),
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": rng.uniform(6.0, 65.0, n).round(2),
        "congestion_surcharge": 2.5,
        "airport_fee": 0.0,
    })


@pytest.fixture()
def sample_bronze_df(sample_raw_df) -> pd.DataFrame:
    """
    Bronze-schema DataFrame: raw columns + metadata columns.
    Simulates what BronzeIngester.ingest_file() produces.
    """
    df = sample_raw_df.copy()
    df["_source_file"] = "yellow_tripdata_2023-01.parquet"
    df["_ingested_at"] = pd.Timestamp("2023-02-01 10:00:00")
    df["_batch_id"] = "test-batch-001"
    return df


# ---------------------------------------------------------------------------
# Silver fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_silver_df() -> pd.DataFrame:
    """
    Silver-schema DataFrame: snake_case columns + derived columns.
    20 clean rows, no invalid values.
    """
    base = pd.Timestamp("2023-01-15 08:00:00")
    n = 20
    rng = np.random.default_rng(1)

    pickup = pd.date_range(base, periods=n, freq="5min")
    dropoff = pickup + pd.to_timedelta(rng.integers(300, 1800, n), unit="s")
    trip_distance = rng.uniform(0.5, 15.0, n).round(2)
    fare_amount = rng.uniform(5.0, 50.0, n).round(2)
    duration_min = (dropoff - pickup).total_seconds() / 60

    return pd.DataFrame({
        "vendor_id": rng.choice([1, 2], n),
        "pickup_at": pickup,
        "dropoff_at": dropoff,
        "passenger_count": rng.integers(1, 4, n).astype(float),
        "trip_distance": trip_distance,
        "rate_code_id": 1.0,
        "store_and_fwd_flag": "N",
        "pickup_location_id": rng.integers(1, 265, n),
        "dropoff_location_id": rng.integers(1, 265, n),
        "payment_type_code": rng.choice([1, 2], n),
        "payment_type": "credit_card",
        "fare_amount": fare_amount,
        "extra_charge": 0.5,
        "mta_tax": 0.5,
        "tip_amount": rng.uniform(0.0, 10.0, n).round(2),
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": (fare_amount + 3.0).round(2),
        "congestion_surcharge": 2.5,
        "airport_fee": 0.0,
        "trip_duration_min": duration_min,
        "fare_per_mile": (fare_amount / trip_distance).round(4),
        "fare_per_minute": (fare_amount / duration_min).round(4),
        "_source_file": "yellow_tripdata_2023-01.parquet",
        "_ingested_at": pd.Timestamp("2023-02-01 10:00:00"),
        "_batch_id": "test-batch-001",
    })


# ---------------------------------------------------------------------------
# Dirty data fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def dirty_bronze_df(sample_bronze_df) -> pd.DataFrame:
    """
    Bronze-schema DataFrame with injected data quality issues.
    Appends 6 bad rows to the clean 20-row fixture.
    """
    bad_rows = [
        # null pickup timestamp
        {**sample_bronze_df.iloc[0].to_dict(), "tpep_pickup_datetime": None},
        # zero fare
        {**sample_bronze_df.iloc[1].to_dict(), "fare_amount": 0.0},
        # extreme distance
        {**sample_bronze_df.iloc[2].to_dict(), "trip_distance": 500.0},
        # zero passengers
        {**sample_bronze_df.iloc[3].to_dict(), "passenger_count": 0.0},
        # negative fare
        {**sample_bronze_df.iloc[4].to_dict(), "fare_amount": -5.0},
        # extreme fare
        {**sample_bronze_df.iloc[5].to_dict(), "fare_amount": 999.0},
    ]
    bad_df = pd.DataFrame(bad_rows)
    return pd.concat([sample_bronze_df, bad_df], ignore_index=True)
