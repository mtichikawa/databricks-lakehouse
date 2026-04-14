"""
silver.py — SilverTransformer: bronze → cleaned silver + quarantine.

Design:
- Renames columns to snake_case.
- Casts timestamps, numerics, categoricals.
- Adds derived columns: trip_duration_min, fare_per_mile, fare_per_minute.
- Validates rows; quarantines invalid ones with a _quarantine_reason column.
- Deduplicates on (pickup_at, dropoff_at, pickup_location_id, dropoff_location_id, fare_amount).
- Row count invariant: bronze == silver + quarantine (no silent drops).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import (
    append_delta,
    get_logger,
    overwrite_delta,
    read_delta,
)

logger = get_logger(__name__)


class SilverTransformer:
    """
    Transforms bronze data into cleaned silver + quarantine tables.

    Usage:
        transformer = SilverTransformer()
        summary = transformer.transform(bronze_df)
    """

    def __init__(self, delta_dir: Optional[Path] = None):
        self.delta_dir = Path(delta_dir) if delta_dir else cfg.DELTA_DIR
        self.silver_path = self.delta_dir / cfg.SILVER_TABLE
        self.quarantine_path = self.delta_dir / cfg.SILVER_QUARANTINE_TABLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, bronze_df: pd.DataFrame) -> dict:
        """
        Full bronze → silver pipeline.

        1. Rename columns
        2. Cast types
        3. Add derived columns
        4. Validate rows → split into valid + quarantine
        5. Deduplicate valid rows
        6. Write silver + quarantine Delta tables
        7. Return summary dict

        Returns:
            dict with keys: silver_rows, quarantine_rows, total_input_rows, deduped_rows
        """
        logger.info("Starting silver transform on %d rows", len(bronze_df))

        df = self._rename_columns(bronze_df.copy())
        df = self._cast_types(df)
        df = self._add_derived_columns(df)

        clean_df, quarantine_df = self._validate_rows(df)
        clean_df = self._deduplicate(clean_df)

        # Write tables
        overwrite_delta(clean_df, self.silver_path)
        overwrite_delta(quarantine_df, self.quarantine_path)

        summary = {
            "total_input_rows": len(bronze_df),
            "silver_rows": len(clean_df),
            "quarantine_rows": len(quarantine_df),
            "deduped_rows": len(bronze_df) - len(quarantine_df) - len(clean_df),
        }
        logger.info("Silver transform complete: %s", summary)
        return summary

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Public convenience: run the cleaning pipeline without writing Delta tables.

        Returns (clean_df, quarantine_df).
        Useful for unit tests that don't need Delta I/O.
        """
        df = self._rename_columns(df.copy())
        df = self._cast_types(df)
        df = self._add_derived_columns(df)
        clean_df, quarantine_df = self._validate_rows(df)
        clean_df = self._deduplicate(clean_df)
        return clean_df, quarantine_df

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map original TLC column names to snake_case equivalents."""
        col_map = {k: v for k, v in cfg.SILVER_COLUMN_MAP.items() if k in df.columns}
        df = df.rename(columns=col_map)
        logger.debug("Renamed %d column(s)", len(col_map))
        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to appropriate types; missing columns get NaN."""

        def safe_to_datetime(series: pd.Series) -> pd.Series:
            return pd.to_datetime(series, errors="coerce", utc=False)

        def safe_to_float(series: pd.Series) -> pd.Series:
            return pd.to_numeric(series, errors="coerce").astype(float)

        def safe_to_int(series: pd.Series) -> pd.Series:
            return pd.to_numeric(series, errors="coerce")

        # Timestamps
        for col in ("pickup_at", "dropoff_at"):
            if col in df.columns:
                df[col] = safe_to_datetime(df[col])

        # Floats
        for col in (
            "fare_amount", "tip_amount", "total_amount",
            "extra_charge", "mta_tax", "tolls_amount",
            "improvement_surcharge", "congestion_surcharge", "airport_fee",
            "trip_distance",
        ):
            if col in df.columns:
                df[col] = safe_to_float(df[col])

        # Integers
        for col in (
            "vendor_id", "passenger_count", "rate_code_id",
            "pickup_location_id", "dropoff_location_id", "payment_type_code",
        ):
            if col in df.columns:
                df[col] = safe_to_int(df[col])

        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns: duration, fare_per_mile, fare_per_minute, payment label."""

        # Trip duration in minutes
        if "pickup_at" in df.columns and "dropoff_at" in df.columns:
            duration_secs = (
                pd.to_datetime(df["dropoff_at"]) - pd.to_datetime(df["pickup_at"])
            ).dt.total_seconds()
            df["trip_duration_min"] = duration_secs / 60.0
        else:
            df["trip_duration_min"] = np.nan

        # Derived per-unit metrics (handle division by zero gracefully)
        if "fare_amount" in df.columns and "trip_distance" in df.columns:
            df["fare_per_mile"] = np.where(
                df["trip_distance"] > 0,
                df["fare_amount"] / df["trip_distance"],
                np.nan,
            )
        else:
            df["fare_per_mile"] = np.nan

        if "fare_amount" in df.columns and "trip_duration_min" in df.columns:
            df["fare_per_minute"] = np.where(
                df["trip_duration_min"] > 0,
                df["fare_amount"] / df["trip_duration_min"],
                np.nan,
            )
        else:
            df["fare_per_minute"] = np.nan

        # Human-readable payment label
        if "payment_type_code" in df.columns:
            df["payment_type"] = (
                df["payment_type_code"]
                .map(cfg.PAYMENT_TYPE_LABELS)
                .fillna("unknown")
            )

        return df

    def _validate_rows(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split df into (valid_rows, quarantine_rows).

        Invalid rows go to quarantine with a _quarantine_reason column.
        Multiple reasons are joined with "; ".
        """
        reasons = pd.Series([""] * len(df), index=df.index)

        # Null timestamps
        if "pickup_at" in df.columns:
            mask = df["pickup_at"].isna()
            reasons[mask] += "null pickup_at; "

        if "dropoff_at" in df.columns:
            mask = df["dropoff_at"].isna()
            reasons[mask] += "null dropoff_at; "

        # Trip duration bounds
        if "trip_duration_min" in df.columns:
            duration_secs = df["trip_duration_min"] * 60
            mask_low = duration_secs <= 0
            mask_high = duration_secs > cfg.DURATION_MAX_SECONDS
            reasons[mask_low] += "trip_duration <= 0s; "
            reasons[mask_high] += "trip_duration > 24h; "

        # Distance bounds
        if "trip_distance" in df.columns:
            mask_low = df["trip_distance"] <= 0
            mask_high = df["trip_distance"] > cfg.DISTANCE_MAX
            reasons[mask_low] += "trip_distance <= 0; "
            reasons[mask_high] += f"trip_distance > {cfg.DISTANCE_MAX}; "

        # Fare bounds
        if "fare_amount" in df.columns:
            mask_low = df["fare_amount"] <= 0
            mask_high = df["fare_amount"] > cfg.FARE_MAX
            reasons[mask_low] += "fare_amount <= 0; "
            reasons[mask_high] += f"fare_amount > {cfg.FARE_MAX}; "

        # Passenger count bounds
        if "passenger_count" in df.columns:
            mask_zero = df["passenger_count"] == 0
            mask_high = df["passenger_count"] > cfg.PASSENGER_MAX
            reasons[mask_zero] += "passenger_count = 0; "
            reasons[mask_high] += f"passenger_count > {cfg.PASSENGER_MAX}; "

        invalid_mask = reasons.str.strip() != ""
        quarantine_df = df[invalid_mask].copy()
        quarantine_df["_quarantine_reason"] = reasons[invalid_mask].str.strip().str.rstrip(";").str.strip()
        clean_df = df[~invalid_mask].copy()

        logger.info(
            "Validation: %d valid, %d quarantined",
            len(clean_df),
            len(quarantine_df),
        )
        return clean_df, quarantine_df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact duplicates on the business dedup key."""
        dedup_cols = [c for c in cfg.SILVER_DEDUP_COLS if c in df.columns]
        if not dedup_cols:
            return df
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols)
        dropped = before - len(df)
        if dropped:
            logger.info("Deduplication dropped %d duplicate row(s)", dropped)
        return df
