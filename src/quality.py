"""
quality.py — Data quality framework.

Classes:
- QualityResult: outcome of a single check (pass/warn/fail + metric)
- QualityCheck: wraps a check function with warn/fail thresholds
- QualitySummary: aggregated result of a QualityRunner run
- QualityRunner: executes a list of QualityChecks and logs results

Pre-built check factories:
- null_rate_check(column, warn_threshold, fail_threshold)
- row_count_match(expected_count)
- range_check(column, min_val, max_val)
- uniqueness_check(columns)
- freshness_check(date_column, expected_dates)
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from .utils import get_logger, utcnow_str

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    check_name: str
    status: str          # "pass" (continue) | "warn" (log + continue) | "fail" (halt pipeline)
    metric_value: Any    # the raw computed metric (float, int, bool …)
    message: str         # human-readable description


@dataclass
class QualitySummary:
    run_at: str
    source_table: str
    target_table: str
    results: list[QualityResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.status in ("pass", "warn") for r in self.results)

    @property
    def has_failures(self) -> bool:
        return any(r.status == "fail" for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.status == "warn" for r in self.results)

    def __str__(self) -> str:
        lines = [
            f"Quality summary [{self.source_table} → {self.target_table}]",
            f"  Run at : {self.run_at}",
            f"  Status : {'PASS' if self.passed else 'FAIL'}",
        ]
        for r in self.results:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(r.status, "?")
            lines.append(f"  {icon} [{r.status.upper():4s}] {r.check_name}: {r.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

class QualityCheck:
    """
    A single named quality check.

    Args:
        name:            Human-readable name.
        check_fn:        fn(df) -> (metric_value, status_str, message)
                         status_str must be "pass", "warn", or "fail".
    """

    def __init__(self, name: str, check_fn: Callable[[pd.DataFrame], tuple]):
        self.name = name
        self.check_fn = check_fn

    def run(self, df: pd.DataFrame) -> QualityResult:
        try:
            metric_value, status, message = self.check_fn(df)
            return QualityResult(
                check_name=self.name,
                status=status,
                metric_value=metric_value,
                message=message,
            )
        except Exception as exc:
            return QualityResult(
                check_name=self.name,
                status="fail",
                metric_value=None,
                message=f"Check raised exception: {exc}",
            )


class QualityRunner:
    """
    Executes a list of QualityChecks against a DataFrame.

    Usage:
        runner = QualityRunner(checks=[...], source_table="bronze", target_table="silver")
        summary = runner.run_all(df)
        runner.log_results(summary, log_dir=Path("data/quality_logs"))
    """

    def __init__(
        self,
        checks: list[QualityCheck],
        source_table: str = "",
        target_table: str = "",
    ):
        self.checks = checks
        self.source_table = source_table
        self.target_table = target_table

    def run_all(self, df: pd.DataFrame) -> QualitySummary:
        """Run all checks and return a QualitySummary."""
        summary = QualitySummary(
            run_at=utcnow_str(),
            source_table=self.source_table,
            target_table=self.target_table,
        )
        for check in self.checks:
            result = check.run(df)
            summary.results.append(result)
            log_fn = logger.warning if result.status == "warn" else (
                logger.error if result.status == "fail" else logger.info
            )
            log_fn("[%s] %s: %s", result.status.upper(), result.check_name, result.message)

        return summary

    def log_results(self, summary: QualitySummary, log_dir: Optional[Path] = None) -> Path:
        """
        Write the summary to a JSON file under log_dir.

        Returns the path of the written file.
        """
        if log_dir is None:
            from . import config as cfg
            log_dir = cfg.QUALITY_LOG_DIR

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"quality_{summary.source_table}_{summary.target_table}_"
            f"{summary.run_at[:19].replace(':', '-').replace('T', '_')}.json"
        )
        path = log_dir / filename

        data = {
            "run_at": summary.run_at,
            "source_table": summary.source_table,
            "target_table": summary.target_table,
            "passed": summary.passed,
            "results": [asdict(r) for r in summary.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Quality log written to %s", path)
        return path


# ---------------------------------------------------------------------------
# Pre-built check factories
# ---------------------------------------------------------------------------

def null_rate_check(
    column: str,
    warn_threshold: float = 0.05,
    fail_threshold: float = 0.20,
) -> QualityCheck:
    """Check that the null rate for a column stays within thresholds."""

    def _check(df: pd.DataFrame) -> tuple:
        if column not in df.columns:
            return None, "fail", f"Column '{column}' not found in DataFrame"
        if len(df) == 0:
            return 0.0, "pass", "DataFrame is empty — no nulls"
        rate = df[column].isna().mean()
        if rate > fail_threshold:
            status = "fail"
            msg = f"Null rate {rate:.1%} exceeds fail threshold {fail_threshold:.1%}"
        elif rate > warn_threshold:
            status = "warn"
            msg = f"Null rate {rate:.1%} exceeds warn threshold {warn_threshold:.1%}"
        else:
            status = "pass"
            msg = f"Null rate {rate:.1%} is within threshold ({warn_threshold:.1%})"
        return rate, status, msg

    return QualityCheck(f"null_rate[{column}]", _check)


def row_count_match(expected_count: int, name: str = "row_count_match") -> QualityCheck:
    """Check that the DataFrame has exactly expected_count rows."""

    def _check(df: pd.DataFrame) -> tuple:
        actual = len(df)
        if actual == expected_count:
            return actual, "pass", f"Row count matches: {actual}"
        else:
            diff = actual - expected_count
            return actual, "fail", (
                f"Row count mismatch: got {actual}, expected {expected_count} "
                f"(diff={diff:+d})"
            )

    return QualityCheck(name, _check)


def range_check(
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> QualityCheck:
    """Check that all values in column fall within [min_val, max_val]."""

    def _check(df: pd.DataFrame) -> tuple:
        if column not in df.columns:
            return None, "fail", f"Column '{column}' not found"
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        violations = 0
        if min_val is not None:
            violations += (series < min_val).sum()
        if max_val is not None:
            violations += (series > max_val).sum()
        if violations > 0:
            return violations, "fail", (
                f"{violations} value(s) in '{column}' outside "
                f"range [{min_val}, {max_val}]"
            )
        return 0, "pass", f"All values in '{column}' within range [{min_val}, {max_val}]"

    return QualityCheck(f"range_check[{column}]", _check)


def uniqueness_check(columns: list[str]) -> QualityCheck:
    """Check that the combination of columns has no duplicate rows."""
    col_str = ", ".join(columns)

    def _check(df: pd.DataFrame) -> tuple:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return None, "fail", f"Columns not found: {missing}"
        total = len(df)
        unique = df[columns].drop_duplicates().shape[0]
        duplicates = total - unique
        if duplicates > 0:
            return duplicates, "fail", (
                f"{duplicates} duplicate row(s) on columns ({col_str})"
            )
        return 0, "pass", f"No duplicates on ({col_str})"

    return QualityCheck(f"uniqueness[{col_str}]", _check)


def freshness_check(date_column: str, min_rows: int = 1) -> QualityCheck:
    """Check that the table contains at least min_rows rows (data is not empty)."""

    def _check(df: pd.DataFrame) -> tuple:
        if date_column not in df.columns:
            return None, "fail", f"Column '{date_column}' not found"
        non_null = df[date_column].dropna()
        if len(non_null) < min_rows:
            return len(non_null), "fail", (
                f"Only {len(non_null)} non-null rows in '{date_column}', "
                f"expected >= {min_rows}"
            )
        return len(non_null), "pass", f"{len(non_null)} rows with data in '{date_column}'"

    return QualityCheck(f"freshness[{date_column}]", _check)
