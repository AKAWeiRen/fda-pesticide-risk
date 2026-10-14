"""Summarize sample-level data quality and yearly outcomes."""

from __future__ import annotations

import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "fda_samples_2019_2023.csv"
)
EXPECTED_YEARS = tuple(str(year) for year in range(2019, 2024))
REQUIRED_COLUMNS = ("sample_id", "Year", "LabClass")
VALID_LAB_CLASSES = {"1", "2", "3", "4", "5"}


class SampleOverviewError(ValueError):
    """Raised when sample data cannot support the requested overview."""


def _clean(value: object) -> str:
    """Normalize one CSV value to stripped text."""
    return "" if value is None else str(value).strip()


def load_samples(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Load sample rows from *path*."""
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise SampleOverviewError(f"找不到樣本層級資料：{path}") from exc
    except OSError as exc:
        raise SampleOverviewError(f"無法開啟樣本層級資料：{path}") from exc

    with handle:
        try:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
        except (csv.Error, UnicodeError) as exc:
            raise SampleOverviewError(f"無法解析樣本層級資料：{path}") from exc
        missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
        if missing:
            raise SampleOverviewError(
                f"樣本層級資料缺少必要欄位：{', '.join(missing)}"
            )
    return fieldnames, rows


def build_overview(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    expected_years: Sequence[str] = EXPECTED_YEARS,
) -> dict[str, Any]:
    """Build a data-quality and yearly summary."""
    names = tuple(fieldnames)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(names))
    if missing_columns:
        raise SampleOverviewError(
            f"樣本層級資料缺少必要欄位：{', '.join(missing_columns)}"
        )
    if not rows:
        raise SampleOverviewError("樣本層級資料沒有任何樣本")

    expected = tuple(str(year) for year in expected_years)
    expected_set = set(expected)
    missing_counts = {name: 0 for name in names}
    unknown_counts = {name: 0 for name in names}
    seen_sample_ids: set[str] = set()
    yearly_counts = {
        year: {"samples": 0, "labclass3": 0}
        for year in expected
    }

    for row_number, row in enumerate(rows, start=2):
        sample_id = _clean(row.get("sample_id"))
        if not sample_id:
            raise SampleOverviewError(f"第 {row_number} 列的 sample_id 是空白")
        if sample_id in seen_sample_ids:
            raise SampleOverviewError(f"sample_id 重複：{sample_id}")
        seen_sample_ids.add(sample_id)

        year = _clean(row.get("Year"))
        if year not in expected_set:
            raise SampleOverviewError(
                f"第 {row_number} 列的年度不在預期範圍：{year or '空白'}"
            )
        lab_class = _clean(row.get("LabClass"))
        if lab_class not in VALID_LAB_CLASSES:
            raise SampleOverviewError(
                f"第 {row_number} 列的 LabClass 無效：{lab_class or '空白'}"
            )

        for name in names:
            value = _clean(row.get(name))
            if not value:
                missing_counts[name] += 1
            elif value.casefold() == "unknown":
                unknown_counts[name] += 1

        yearly_counts[year]["samples"] += 1
        if lab_class == "3":
            yearly_counts[year]["labclass3"] += 1

    absent_years = [
        year for year in expected if yearly_counts[year]["samples"] == 0
    ]
    if absent_years:
        raise SampleOverviewError(f"樣本層級資料缺少年度：{', '.join(absent_years)}")

    yearly = []
    for year in expected:
        samples = yearly_counts[year]["samples"]
        labclass3 = yearly_counts[year]["labclass3"]
        yearly.append(
            {
                "year": year,
                "samples": samples,
                "labclass3": labclass3,
                "rate": labclass3 / samples,
            }
        )

    return {
        "sample_count": len(rows),
        "fieldnames": names,
        "missing": missing_counts,
        "unknown": unknown_counts,
        "yearly": yearly,
    }


def format_overview(overview: Mapping[str, object]) -> str:
    """Format one overview for terminal display."""
    fieldnames = overview["fieldnames"]
    missing = overview["missing"]
    unknown = overview["unknown"]
    yearly = overview["yearly"]
    if not isinstance(fieldnames, Sequence):
        raise SampleOverviewError("概況缺少欄位清單")
    if not isinstance(missing, Mapping) or not isinstance(unknown, Mapping):
        raise SampleOverviewError("概況缺少資料品質統計")
    if not isinstance(yearly, Sequence):
        raise SampleOverviewError("概況缺少年度統計")

    lines = [
        "FDA農藥殘留專案 樣本層級資料品質與年度概況",
        "",
        f"總樣本數：{int(overview['sample_count']):,}",
        "",
        "欄位資料品質：",
    ]
    for name in fieldnames:
        lines.append(
            f"- {name}：空白 {int(missing[name]):,}；"
            f"Unknown {int(unknown[name]):,}"
        )

    lines.extend(["", "年度概況："])
    for item in yearly:
        if not isinstance(item, Mapping):
            raise SampleOverviewError("年度概況格式無效")
        lines.append(
            f"- {item['year']}：樣本 {int(item['samples']):,}；"
            f"LabClass 3 為 {int(item['labclass3']):,}；"
            f"比例 {float(item['rate']):.2%}"
        )
    lines.extend(
        [
            "",
            "限制：年度比例只描述 FDA 目標式抽驗資料，",
            "不代表整體市場違規率，也不能直接判定食品安全或不安全。",
        ]
    )
    return "\n".join(lines)


def main(
    input_path: Path | None = None,
    *,
    expected_years: Sequence[str] = EXPECTED_YEARS,
) -> int:
    """Print the overview and return a process status code."""
    path = DEFAULT_INPUT_PATH if input_path is None else input_path
    try:
        fieldnames, rows = load_samples(path)
        overview = build_overview(
            fieldnames,
            rows,
            expected_years=expected_years,
        )
        print(format_overview(overview))
    except (SampleOverviewError, OSError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
