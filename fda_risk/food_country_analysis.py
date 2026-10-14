"""Explore sample-level food and country outcomes."""

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
REQUIRED_COLUMNS = ("sample_id", "ProdName", "Country", "LabClass")
VALID_LAB_CLASSES = {"1", "2", "3", "4", "5"}


class FoodCountryAnalysisError(ValueError):
    """Raised when sample data cannot support the requested analysis."""


def _normalize(value: object) -> str:
    """Strip surrounding whitespace and collapse internal whitespace."""
    return " ".join(str(value).split()) if value is not None else ""


def load_samples(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Load the sample-level CSV and validate its required columns."""
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise FoodCountryAnalysisError(f"找不到樣本層級資料：{path}") from exc
    except OSError as exc:
        raise FoodCountryAnalysisError(f"無法開啟樣本層級資料：{path}") from exc

    with handle:
        try:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
        except (csv.Error, UnicodeError) as exc:
            raise FoodCountryAnalysisError(
                f"無法解析樣本層級資料：{path}"
            ) from exc

    missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
    if missing:
        raise FoodCountryAnalysisError(
            f"樣本層級資料缺少必要欄位：{', '.join(missing)}"
        )
    return fieldnames, rows


def _rank_groups(
    groups: Mapping[str, Mapping[str, int]],
    *,
    top_n: int,
    min_rate_samples: int,
) -> dict[str, Any]:
    items = [
        {
            "name": name,
            "samples": counts["samples"],
            "violative": counts["violative"],
            "rate": counts["violative"] / counts["samples"],
        }
        for name, counts in groups.items()
    ]
    volume_top = sorted(
        items,
        key=lambda item: (-item["samples"], item["name"]),
    )[:top_n]
    eligible = [
        item for item in items if item["samples"] >= min_rate_samples
    ]
    rate_top = sorted(
        eligible,
        key=lambda item: (-item["rate"], -item["samples"], item["name"]),
    )[:top_n]
    return {
        "group_count": len(groups),
        "volume_top": volume_top,
        "rate_top": rate_top,
    }


def build_analysis(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    top_n: int = 10,
    min_rate_samples: int = 50,
) -> dict[str, Any]:
    """Aggregate sample counts and LabClass 3 rates by food and country."""
    names = tuple(fieldnames)
    missing = sorted(set(REQUIRED_COLUMNS) - set(names))
    if missing:
        raise FoodCountryAnalysisError(
            f"樣本層級資料缺少必要欄位：{', '.join(missing)}"
        )
    if type(top_n) is not int or top_n <= 0:
        raise FoodCountryAnalysisError("top_n 必須是正整數")
    if type(min_rate_samples) is not int or min_rate_samples <= 0:
        raise FoodCountryAnalysisError("min_rate_samples 必須是正整數")
    if not rows:
        raise FoodCountryAnalysisError("樣本層級資料沒有任何樣本")

    product_groups: dict[str, dict[str, int]] = {}
    country_groups: dict[str, dict[str, int]] = {}
    seen_sample_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        sample_id = _normalize(row.get("sample_id"))
        if not sample_id:
            raise FoodCountryAnalysisError(f"第 {row_number} 列的 sample_id 是空白")
        if sample_id in seen_sample_ids:
            raise FoodCountryAnalysisError(f"sample_id 重複：{sample_id}")
        seen_sample_ids.add(sample_id)

        product = _normalize(row.get("ProdName"))
        country = _normalize(row.get("Country"))
        if not product:
            raise FoodCountryAnalysisError(f"第 {row_number} 列的 ProdName 是空白")
        if not country:
            raise FoodCountryAnalysisError(f"第 {row_number} 列的 Country 是空白")

        lab_class = _normalize(row.get("LabClass"))
        if lab_class not in VALID_LAB_CLASSES:
            raise FoodCountryAnalysisError(
                f"第 {row_number} 列的 LabClass 無效：{lab_class or '空白'}"
            )

        for group_map, label in (
            (product_groups, product),
            (country_groups, country),
        ):
            counts = group_map.setdefault(label, {"samples": 0, "violative": 0})
            counts["samples"] += 1
            if lab_class == "3":
                counts["violative"] += 1

    return {
        "sample_count": len(rows),
        "top_n": top_n,
        "min_rate_samples": min_rate_samples,
        "products": _rank_groups(
            product_groups,
            top_n=top_n,
            min_rate_samples=min_rate_samples,
        ),
        "countries": _rank_groups(
            country_groups,
            top_n=top_n,
            min_rate_samples=min_rate_samples,
        ),
    }


def _format_ranking(title: str, items: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [title]
    for rank, item in enumerate(items, start=1):
        lines.append(
            f"{rank}. {item['name']}：樣本 {int(item['samples']):,}；"
            f"LabClass 3 為 {int(item['violative']):,}；"
            f"比例 {float(item['rate']):.2%}"
        )
    return lines


def format_analysis(analysis: Mapping[str, object]) -> str:
    """Format the analysis as a Traditional Chinese terminal report."""
    top_n = int(analysis["top_n"])
    min_rate_samples = int(analysis["min_rate_samples"])
    products = analysis["products"]
    countries = analysis["countries"]
    if not isinstance(products, Mapping) or not isinstance(countries, Mapping):
        raise FoodCountryAnalysisError("探索分析缺少食品或來源國統計")

    lines = [
        "FDA農藥殘留專案 食品與來源國探索分析",
        "",
        f"總樣本數：{int(analysis['sample_count']):,}",
        f"食品群組數：{int(products['group_count']):,}",
        f"來源國群組數：{int(countries['group_count']):,}",
        f"比例排行最低樣本數：{min_rate_samples:,}",
        "",
    ]
    lines.extend(_format_ranking(f"食品樣本數 Top {top_n}", products["volume_top"]))
    lines.append("")
    lines.extend(
        _format_ranking(
            f"食品不利結果比例 Top {top_n}",
            products["rate_top"],
        )
    )
    lines.append("")
    lines.extend(_format_ranking(f"來源國樣本數 Top {top_n}", countries["volume_top"]))
    lines.append("")
    lines.extend(
        _format_ranking(
            f"來源國不利結果比例 Top {top_n}",
            countries["rate_top"],
        )
    )
    lines.extend(
        [
            "",
            "限制：本結果只描述 FDA 目標式抽驗資料內的樣本分布與相對比例，",
            "不代表整體市場違規率，也不能直接判定特定食品或來源國安全與否。",
        ]
    )
    return "\n".join(lines)


def main(
    input_path: Path | None = None,
    *,
    top_n: int = 10,
    min_rate_samples: int = 50,
) -> int:
    """Print the food/country analysis and return a process status code."""
    path = DEFAULT_INPUT_PATH if input_path is None else input_path
    try:
        fieldnames, rows = load_samples(path)
        analysis = build_analysis(
            fieldnames,
            rows,
            top_n=top_n,
            min_rate_samples=min_rate_samples,
        )
        print(format_analysis(analysis))
    except (FoodCountryAnalysisError, OSError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
