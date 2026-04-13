"""
bronze.py — BronzeIngester: raw Parquet → bronze Delta table.

Design:
- Append-only: never modify or delete source records.
- Idempotent: skip files already tracked in the _source_file column.
- Adds metadata columns: _source_file, _ingested_at, _batch_id.
- Preserves original schema exactly — no type casting, no renaming.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg
from .utils import (
    append_delta,
    get_distinct_values,
    get_logger,
    new_batch_id,
    read_parquet,
    utcnow,
)

logger = get_logger(__name__)


class BronzeIngester:
    """
    Ingests raw Parquet files into the bronze Delta table.

    Usage:
        ingester = BronzeIngester(raw_dir="data/raw", delta_dir="data/delta")
        summary = ingester.ingest_all()
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        delta_dir: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else cfg.RAW_DATA_DIR
        self.delta_dir = Path(delta_dir) if delta_dir else cfg.DELTA_DIR
        self.bronze_path = self.delta_dir / cfg.BRONZE_TABLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_raw_files(self) -> list[Path]:
        """Return all Parquet files under raw_dir, sorted by name."""
        if not self.raw_dir.exists():
            return []
        files = sorted(self.raw_dir.glob("*.parquet"))
        logger.debug("Found %d raw file(s) in %s", len(files), self.raw_dir)
        return files

    def get_ingested_files(self) -> set[str]:
        """
        Query bronze table for distinct _source_file values.

        Returns an empty set if the table doesn't exist yet (first run).
        """
        ingested = get_distinct_values(self.bronze_path, "_source_file")
        logger.debug("Bronze table has %d ingested file record(s)", len(ingested))
        return ingested

    def ingest_file(self, path: Path, batch_id: Optional[str] = None) -> int:
        """
        Read one Parquet file, add metadata columns, append to bronze table.

        Returns the number of rows ingested.
        """
        if batch_id is None:
            batch_id = new_batch_id()

        logger.info("Ingesting %s ...", path.name)
        df = read_parquet(path)

        # Add metadata
        df["_source_file"] = path.name
        df["_ingested_at"] = utcnow()
        df["_batch_id"] = batch_id

        # Coerce _ingested_at to a plain (timezone-naive) datetime so
        # deltalake can infer the Arrow schema without error on all platforms.
        df["_ingested_at"] = pd.to_datetime(df["_ingested_at"]).dt.tz_localize(None)

        row_count = len(df)
        append_delta(df, self.bronze_path)
        logger.info("  → %d rows written to bronze", row_count)
        return row_count

    def ingest_all(self) -> dict:
        """
        Ingest all raw Parquet files, skipping those already in bronze.

        Returns a summary dict with counts of new/skipped/failed files.
        """
        raw_files = self.list_raw_files()
        if not raw_files:
            logger.warning("No raw Parquet files found in %s", self.raw_dir)
            return {
                "total_files": 0,
                "new_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "total_rows_ingested": 0,
            }

        ingested_files = self.get_ingested_files()
        batch_id = new_batch_id()

        new_files = 0
        skipped_files = 0
        failed_files = 0
        total_rows = 0

        for path in raw_files:
            if path.name in ingested_files:
                logger.info("Skipping already-ingested file: %s", path.name)
                skipped_files += 1
                continue
            try:
                rows = self.ingest_file(path, batch_id=batch_id)
                total_rows += rows
                new_files += 1
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", path.name, exc)
                failed_files += 1

        summary = {
            "total_files": len(raw_files),
            "new_files": new_files,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
            "total_rows_ingested": total_rows,
            "batch_id": batch_id,
        }
        logger.info("Bronze ingest complete: %s", summary)
        return summary
