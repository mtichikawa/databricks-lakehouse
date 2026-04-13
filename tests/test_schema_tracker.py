"""
test_schema_tracker.py — Tests for SchemaEvolutionTracker.

Covers:
- snapshot_schema: JSON file written with correct structure
- detect_changes: add a column → detected as "added"
- detect_changes: remove a column → detected as "removed"
- detect_changes: change column type → detected as "type_changed"
- No changes → empty list
- First-run (no existing snapshot) → empty list
"""

import json

import pandas as pd
import pytest

from src.schema_tracker import SchemaEvolutionTracker, SchemaChange


def base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "vendor_id": [1, 2],
        "fare_amount": [10.0, 20.0],
        "pickup_at": pd.to_datetime(["2023-01-15", "2023-01-16"]),
    })


class TestSnapshotSchema:
    def test_creates_json_file(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df = base_df()
        path = tracker.snapshot_schema(df, "bronze_trips")
        assert path.exists()
        assert path.suffix == ".json"

    def test_json_has_correct_structure(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df = base_df()
        path = tracker.snapshot_schema(df, "bronze_trips")
        with open(path) as f:
            data = json.load(f)
        assert data["table_name"] == "bronze_trips"
        assert "snapshot_at" in data
        assert "schema" in data

    def test_schema_contains_column_dtypes(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df = base_df()
        path = tracker.snapshot_schema(df, "bronze_trips")
        with open(path) as f:
            data = json.load(f)
        assert "vendor_id" in data["schema"]
        assert "fare_amount" in data["schema"]

    def test_overwrites_previous_snapshot(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        tracker.snapshot_schema(df1, "test_table")
        tracker.snapshot_schema(df2, "test_table")
        with open(temp_dir / "test_table.json") as f:
            data = json.load(f)
        assert "b" in data["schema"]
        assert "a" not in data["schema"]


class TestDetectChanges:
    def test_no_snapshot_returns_empty_list(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df = base_df()
        changes = tracker.detect_changes(df, "bronze_trips")
        assert changes == []

    def test_no_changes_returns_empty_list(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df = base_df()
        tracker.snapshot_schema(df, "test_table")
        changes = tracker.detect_changes(df, "test_table")
        assert changes == []

    def test_added_column_detected(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df_old = base_df()
        tracker.snapshot_schema(df_old, "test_table")

        df_new = base_df()
        df_new["new_col"] = 42
        changes = tracker.detect_changes(df_new, "test_table")

        added = [c for c in changes if c.change_type == "added"]
        assert len(added) == 1
        assert added[0].column_name == "new_col"
        assert added[0].old_value is None

    def test_removed_column_detected(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df_old = base_df()
        tracker.snapshot_schema(df_old, "test_table")

        df_new = base_df().drop(columns=["fare_amount"])
        changes = tracker.detect_changes(df_new, "test_table")

        removed = [c for c in changes if c.change_type == "removed"]
        assert len(removed) == 1
        assert removed[0].column_name == "fare_amount"
        assert removed[0].new_value is None

    def test_type_change_detected(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df_old = pd.DataFrame({"amount": [1.0, 2.0]})  # float64
        tracker.snapshot_schema(df_old, "test_table")

        df_new = pd.DataFrame({"amount": ["a", "b"]})  # object
        changes = tracker.detect_changes(df_new, "test_table")

        type_changes = [c for c in changes if c.change_type == "type_changed"]
        assert len(type_changes) == 1
        assert type_changes[0].column_name == "amount"
        assert type_changes[0].old_value != type_changes[0].new_value

    def test_multiple_changes_detected(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        df_old = pd.DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
        tracker.snapshot_schema(df_old, "test_table")

        df_new = pd.DataFrame({"a": [1], "d": [True]})  # b removed, c removed, d added
        changes = tracker.detect_changes(df_new, "test_table")

        added = [c for c in changes if c.change_type == "added"]
        removed = [c for c in changes if c.change_type == "removed"]
        assert len(added) == 1
        assert len(removed) == 2


class TestLogChanges:
    def test_no_changes_returns_none(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        result = tracker.log_changes("test_table", [])
        assert result is None

    def test_changes_written_to_log(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        changes = [
            SchemaChange(change_type="added", column_name="new_col", old_value=None, new_value="int64")
        ]
        log_path = tracker.log_changes("test_table", changes)
        assert log_path is not None
        assert log_path.exists()
        assert log_path.suffix == ".jsonl"

    def test_log_is_valid_jsonl(self, temp_dir):
        tracker = SchemaEvolutionTracker(schema_dir=temp_dir)
        changes = [
            SchemaChange(change_type="removed", column_name="old_col", old_value="float64", new_value=None)
        ]
        log_path = tracker.log_changes("test_table", changes)
        with open(log_path) as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["table_name"] == "test_table"
        assert len(entry["changes"]) == 1
