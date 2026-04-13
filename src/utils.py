"""
utils.py — Shared helpers for the Databricks Lakehouse project.

Provides:
- Logging setup
- Delta table read/write wrappers (local deltalake library)
- Date/path utilities
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import deltalake
    from deltalake import DeltaTable, write_deltalake
    HAS_DELTALAKE = True
except ImportError:
    HAS_DELTALAKE = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def new_batch_id() -> str:
    """Generate a UUID-based batch identifier."""
    return str(uuid.uuid4())


def utcnow_str() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Delta table helpers
# ---------------------------------------------------------------------------

def table_exists(table_path: Path) -> bool:
    """Return True if a Delta table exists at the given path."""
    if not HAS_DELTALAKE:
        return False
    try:
        DeltaTable(str(table_path))
        return True
    except Exception:
        return False


def read_delta(table_path: Path) -> pd.DataFrame:
    """
    Read a Delta table into a pandas DataFrame.

    Falls back gracefully if the table doesn't exist — returns an empty
    DataFrame so callers don't need to guard every read.
    """
    if not table_exists(table_path):
        return pd.DataFrame()
    dt = DeltaTable(str(table_path))
    return dt.to_pandas()


def write_delta(
    df: pd.DataFrame,
    table_path: Path,
    mode: str = "append",
    schema_mode: str = "merge",
) -> None:
    """
    Write a pandas DataFrame to a Delta table.

    Args:
        df: DataFrame to write.
        table_path: Filesystem path for the Delta table.
        mode: "append" or "overwrite".
        schema_mode: "merge" allows schema evolution; "overwrite" replaces schema.
    """
    if df.empty:
        return

    table_path.mkdir(parents=True, exist_ok=True)
    write_deltalake(
        str(table_path),
        df,
        mode=mode,
        schema_mode=schema_mode,
    )


def append_delta(df: pd.DataFrame, table_path: Path) -> None:
    """Convenience wrapper: append rows to an existing (or new) Delta table."""
    write_delta(df, table_path, mode="append")


def overwrite_delta(df: pd.DataFrame, table_path: Path) -> None:
    """Convenience wrapper: overwrite a Delta table completely."""
    write_delta(df, table_path, mode="overwrite", schema_mode="overwrite")


def get_distinct_values(table_path: Path, column: str) -> set:
    """
    Return distinct values of a column from a Delta table.

    Used by BronzeIngester to find already-ingested filenames.
    Returns an empty set if the table doesn't exist.
    """
    df = read_delta(table_path)
    if df.empty or column not in df.columns:
        return set()
    return set(df[column].dropna().unique())


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def write_json(data: dict, path: Path) -> None:
    """Write a dict to a JSON file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: Path) -> Optional[dict]:
    """Read a JSON file; return None if the file doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame."""
    return pd.read_parquet(str(path))


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), index=False)


# ---------------------------------------------------------------------------
# Schema utilities
# ---------------------------------------------------------------------------

def df_schema_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert a DataFrame's dtypes to a serialisable dict.

    Example output: {"vendor_id": "int64", "pickup_at": "datetime64[ns, UTC]"}
    """
    return {col: str(dtype) for col, dtype in df.dtypes.items()}
