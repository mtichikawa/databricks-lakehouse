"""
schema_tracker.py — SchemaEvolutionTracker.

Detects added, removed, and type-changed columns between pipeline runs.
Stores schema snapshots as JSON in data/schemas/.
Logs changes but does not raise — human reviews before promotion.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg
from .utils import get_logger, utcnow_str

logger = get_logger(__name__)


@dataclass
class SchemaChange:
    change_type: str    # "added" | "removed" | "type_changed"
    column_name: str
    old_value: Optional[str]  # old dtype string, or None if added
    new_value: Optional[str]  # new dtype string, or None if removed


class SchemaEvolutionTracker:
    """
    Tracks schema changes for Delta table layers.

    Snapshots are stored as JSON files:
      data/schemas/<table_name>.json

    Each JSON file contains:
      {
        "table_name": "bronze_trips",
        "snapshot_at": "2024-01-01T12:00:00",
        "schema": {"col_name": "dtype_str", ...}
      }

    Usage:
        tracker = SchemaEvolutionTracker()
        changes = tracker.detect_changes(df, "bronze_trips")
        tracker.snapshot_schema(df, "bronze_trips")
        tracker.log_changes("bronze_trips", changes)
    """

    def __init__(self, schema_dir: Optional[Path] = None):
        self.schema_dir = Path(schema_dir) if schema_dir else cfg.SCHEMA_DIR
        self.schema_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot_schema(self, df: pd.DataFrame, table_name: str) -> Path:
        """
        Save current schema (column → dtype) as a JSON snapshot.

        Returns the path of the written file.
        Overwrites the previous snapshot for the same table_name.
        """
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        snapshot = {
            "table_name": table_name,
            "snapshot_at": utcnow_str(),
            "schema": schema,
        }
        path = self._snapshot_path(table_name)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        logger.info("Schema snapshot saved: %s (%d columns)", path.name, len(schema))
        return path

    def detect_changes(
        self, df: pd.DataFrame, table_name: str
    ) -> list[SchemaChange]:
        """
        Compare df's schema against the saved snapshot for table_name.

        Returns a list of SchemaChange objects.
        Returns an empty list if no snapshot exists yet.
        """
        previous = self._load_snapshot(table_name)
        if previous is None:
            logger.info("No existing snapshot for '%s' — skipping change detection", table_name)
            return []

        current_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        prev_schema = previous.get("schema", {})

        changes: list[SchemaChange] = []

        # Detect added columns
        for col in current_schema:
            if col not in prev_schema:
                changes.append(SchemaChange(
                    change_type="added",
                    column_name=col,
                    old_value=None,
                    new_value=current_schema[col],
                ))

        # Detect removed columns
        for col in prev_schema:
            if col not in current_schema:
                changes.append(SchemaChange(
                    change_type="removed",
                    column_name=col,
                    old_value=prev_schema[col],
                    new_value=None,
                ))

        # Detect type changes
        for col in current_schema:
            if col in prev_schema and current_schema[col] != prev_schema[col]:
                changes.append(SchemaChange(
                    change_type="type_changed",
                    column_name=col,
                    old_value=prev_schema[col],
                    new_value=current_schema[col],
                ))

        if changes:
            logger.warning(
                "Schema changes detected in '%s': %d change(s)", table_name, len(changes)
            )
            for c in changes:
                logger.warning("  [%s] %s: %s → %s", c.change_type, c.column_name, c.old_value, c.new_value)
        else:
            logger.info("No schema changes detected for '%s'", table_name)

        return changes

    def log_changes(self, table_name: str, changes: list[SchemaChange]) -> Optional[Path]:
        """
        Append schema changes to a JSONL evolution log file.

        Returns the log path, or None if changes is empty.
        """
        if not changes:
            return None

        log_path = self.schema_dir / "schema_evolution_log.jsonl"
        entry = {
            "logged_at": utcnow_str(),
            "table_name": table_name,
            "changes": [asdict(c) for c in changes],
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.info("Schema evolution logged to %s", log_path)
        return log_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_path(self, table_name: str) -> Path:
        return self.schema_dir / f"{table_name}.json"

    def _load_snapshot(self, table_name: str) -> Optional[dict]:
        path = self._snapshot_path(table_name)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
