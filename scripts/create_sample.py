"""
create_sample.py — Generate a small synthetic NYC taxi sample Parquet file.

Produces ~300 rows with the real TLC schema, including intentional data quality
issues for testing (null timestamps, zero fares, extreme distances, etc.).

Run from project root:
    python scripts/create_sample.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_sample(n_clean: int = 250, n_dirty: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # Clean rows (~250) — realistic NYC taxi trips
    # ---------------------------------------------------------------
    base_ts = pd.Timestamp("2023-01-15 00:00:00")
    pickup_offsets = pd.to_timedelta(rng.integers(0, 86400, n_clean), unit="s")
    pickup_times = base_ts + pickup_offsets
    duration_secs = rng.integers(120, 3600, n_clean)  # 2 min – 1 hour
    dropoff_times = pickup_times + pd.to_timedelta(duration_secs, unit="s")

    distances = rng.uniform(0.5, 25.0, n_clean).round(2)
    fares = (distances * rng.uniform(1.5, 3.5, n_clean)).round(2)
    tips = (fares * rng.uniform(0, 0.3, n_clean)).round(2)
    totals = (fares + tips + rng.uniform(0, 3, n_clean)).round(2)

    clean = pd.DataFrame({
        "VendorID": rng.choice([1, 2], n_clean),
        "tpep_pickup_datetime": pickup_times,
        "tpep_dropoff_datetime": dropoff_times,
        "passenger_count": rng.integers(1, 5, n_clean).astype(float),
        "trip_distance": distances,
        "RatecodeID": rng.choice([1, 2, 3], n_clean).astype(float),
        "store_and_fwd_flag": rng.choice(["N", "Y"], n_clean),
        "PULocationID": rng.integers(1, 265, n_clean),
        "DOLocationID": rng.integers(1, 265, n_clean),
        "payment_type": rng.choice([1, 2, 3, 4], n_clean),
        "fare_amount": fares,
        "extra": rng.choice([0.0, 0.5, 1.0], n_clean),
        "mta_tax": 0.5,
        "tip_amount": tips,
        "tolls_amount": rng.choice([0.0, 6.12], n_clean),
        "improvement_surcharge": 0.3,
        "total_amount": totals,
        "congestion_surcharge": rng.choice([0.0, 2.5], n_clean),
        "airport_fee": 0.0,
    })

    # ---------------------------------------------------------------
    # Dirty rows (~50) — various data quality issues
    # ---------------------------------------------------------------
    dirty_pickup = pd.Timestamp("2023-01-15 12:00:00")
    dirty_dropoff = pd.Timestamp("2023-01-15 12:30:00")

    dirty = pd.DataFrame({
        "VendorID": [1] * n_dirty,
        "tpep_pickup_datetime": [dirty_pickup] * n_dirty,
        "tpep_dropoff_datetime": [dirty_dropoff] * n_dirty,
        "passenger_count": [1.0] * n_dirty,
        "trip_distance": [5.0] * n_dirty,
        "RatecodeID": [1.0] * n_dirty,
        "store_and_fwd_flag": ["N"] * n_dirty,
        "PULocationID": [100] * n_dirty,
        "DOLocationID": [200] * n_dirty,
        "payment_type": [1] * n_dirty,
        "fare_amount": [10.0] * n_dirty,
        "extra": [0.0] * n_dirty,
        "mta_tax": [0.5] * n_dirty,
        "tip_amount": [2.0] * n_dirty,
        "tolls_amount": [0.0] * n_dirty,
        "improvement_surcharge": [0.3] * n_dirty,
        "total_amount": [12.8] * n_dirty,
        "congestion_surcharge": [0.0] * n_dirty,
        "airport_fee": [0.0] * n_dirty,
    })

    # Inject specific bad values
    # Row 0: null pickup timestamp
    dirty.loc[0, "tpep_pickup_datetime"] = pd.NaT
    # Row 1: zero fare
    dirty.loc[1, "fare_amount"] = 0.0
    dirty.loc[1, "total_amount"] = 0.0
    # Row 2: extremely long trip distance
    dirty.loc[2, "trip_distance"] = 300.0
    # Row 3: zero passengers
    dirty.loc[3, "passenger_count"] = 0.0
    # Row 4: negative fare
    dirty.loc[4, "fare_amount"] = -5.0
    # Rows 5-9: exact duplicates of row 10 (for dedup testing)
    for i in range(5, 10):
        dirty.loc[i, "tpep_pickup_datetime"] = dirty_pickup
        dirty.loc[i, "tpep_dropoff_datetime"] = dirty_dropoff
        dirty.loc[i, "PULocationID"] = 50
        dirty.loc[i, "DOLocationID"] = 51
        dirty.loc[i, "fare_amount"] = 99.99

    # ---------------------------------------------------------------
    # Combine and write
    # ---------------------------------------------------------------
    df = pd.concat([clean, dirty], ignore_index=True)
    output_path = PROJECT_ROOT / "data" / "sample" / "yellow_tripdata_2023-01_sample.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)
    print(f"Sample written: {output_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    generate_sample()
