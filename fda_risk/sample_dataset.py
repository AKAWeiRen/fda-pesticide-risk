"""Build one analysis-ready row per FDA sample."""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "_fda_feasibility" / "extracted"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "fda_samples_2019_2023.csv"
)
EXPECTED_YEARS = tuple(str(year) for year in range(2019, 2024))

INVARIANT_FIELDS = (
    "Year",
    "SplNo",
    "ProdCode",
    "ProdName",
    "Origin",
    "Country",
    "PAC",
    "CollDate",
    "LabClass",
)
REQUIRED_COLUMNS = (*INVARIANT_FIELDS, "VioType")
OUTPUT_FIELDS = (
    "sample_id",
    *INVARIANT_FIELDS,
    "CollMonth",
    "analytical_rows",
    "n_or_x_rows",
    "violative",
)
DISPLAY_UNKNOWN_FIELDS = {"ProdCode", "ProdName", "Origin", "Country", "PAC"}
N_OR_X_CODES = {"N", "X"}


class SampleDataError(ValueError):
    """Raised when source records cannot safely form a sample dataset."""


def _clean(value: object) -> str:
    """Normalize one source value to stripped text."""
    return "" if value is None else str(value).strip()


def load_records(
    source_root: Path, years: Sequence[str] = EXPECTED_YEARS
) -> Iterator[dict[str, str]]:
    """Yield validated records from the requested yearly source files."""
    for year in years:
        path = source_root / year / f"SampleData{year}.txt"
        try:
            handle = path.open("r", encoding="latin-1", newline="")
        except FileNotFoundError as exc:
            raise SampleDataError(f"找不到來源檔：{path}") from exc

        with handle:
            reader = csv.DictReader(handle, delimiter="\t")
            actual_columns = set(reader.fieldnames or ())
            missing = sorted(set(REQUIRED_COLUMNS) - actual_columns)
            if missing:
                raise SampleDataError(
                    f"{path.name} 缺少必要欄位：{', '.join(missing)}"
                )

            try:
                for line_number, row in enumerate(reader, start=2):
                    actual_year = _clean(row.get("Year"))
                    if actual_year != year:
                        raise SampleDataError(
                            f"{path.name} 第 {line_number} 列年度應為 {year}，"
                            f"實際為 {actual_year or '空白'}"
                        )
                    yield row
            except csv.Error as exc:
                raise SampleDataError(
                    f"無法解析 {path.name} 第 {reader.line_num} 列：{exc}"
                ) from exc


def collapse_records(
    records: Iterable[Mapping[str, object]],
) -> list[dict[str, str | int]]:
    """Collapse analytical rows into one validated row per Year + SplNo."""
    grouped: dict[str, dict[str, str | int]] = {}

    for source_row_number, record in enumerate(records, start=1):
        year = _clean(record.get("Year"))
        sample_number = _clean(record.get("SplNo"))
        if not year or not sample_number:
            raise SampleDataError(
                f"第 {source_row_number} 筆資料的 Year 或 SplNo 是空白"
            )

        sample_id = f"{year}-{sample_number}"
        values = {field: _clean(record.get(field)) for field in INVARIANT_FIELDS}
        existing = grouped.get(sample_id)

        if existing is None:
            output_values: dict[str, str | int] = {
                "sample_id": sample_id,
                **values,
                "CollMonth": (
                    values["CollDate"][4:6]
                    if len(values["CollDate"]) >= 6
                    else "Unknown"
                ),
                "analytical_rows": 0,
                "n_or_x_rows": 0,
                "violative": int(values["LabClass"] == "3"),
            }
            for field in DISPLAY_UNKNOWN_FIELDS:
                if not output_values[field]:
                    output_values[field] = "Unknown"
            grouped[sample_id] = output_values
            existing = output_values
        else:
            for field, new_value in values.items():
                old_value = _clean(existing[field])
                if field in DISPLAY_UNKNOWN_FIELDS and old_value == "Unknown":
                    old_value = ""
                if old_value != new_value:
                    raise SampleDataError(
                        f"樣本 {sample_id} 的 {field} 不一致："
                        f"{old_value or '空白'} / {new_value or '空白'}"
                    )

        existing["analytical_rows"] = int(existing["analytical_rows"]) + 1
        violation_type = _clean(record.get("VioType")).upper()
        if violation_type in N_OR_X_CODES:
            existing["n_or_x_rows"] = int(existing["n_or_x_rows"]) + 1

    if not grouped:
        raise SampleDataError("來源資料沒有任何分析列")

    samples = list(grouped.values())
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise SampleDataError("輸出的 sample_id 不是唯一值")
    return samples


def write_samples(samples: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write sample rows as an Excel-friendly UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def main(
    source_root: Path | None = None,
    output_path: Path | None = None,
    *,
    years: Sequence[str] = EXPECTED_YEARS,
) -> int:
    """Build the sample-level CSV and print an acceptance summary."""
    actual_source_root = DEFAULT_SOURCE_ROOT if source_root is None else source_root
    actual_output_path = DEFAULT_OUTPUT_PATH if output_path is None else output_path
    try:
        samples = collapse_records(load_records(actual_source_root, years))
        source_rows = sum(int(sample["analytical_rows"]) for sample in samples)
        if source_rows <= 0:
            raise SampleDataError("來源列數驗證失敗")
        write_samples(samples, actual_output_path)
    except (SampleDataError, OSError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    violative_samples = sum(int(sample["violative"]) for sample in samples)
    flagged_samples = sum(int(sample["n_or_x_rows"]) > 0 for sample in samples)
    print("FDA農藥殘留專案 樣本層級資料集")
    print(f"來源分析列數：{source_rows:,}")
    print(f"輸出樣本數：{len(samples):,}")
    print(f"LabClass 3 樣本數：{violative_samples:,}")
    print(f"至少一列 VioType 為 N 或 X 的樣本數：{flagged_samples:,}")
    print(f"輸出位置：{actual_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
