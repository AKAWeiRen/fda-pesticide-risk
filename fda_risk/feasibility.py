"""Read and present the saved Week 1 feasibility result."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "_fda_feasibility" / "feasibility-results.json"
EXPECTED_YEARS = tuple(str(year) for year in range(2019, 2024))


class FeasibilityDataError(ValueError):
    """Raised when the saved feasibility result cannot support a summary."""


def load_results(path: Path) -> dict[str, Any]:
    """Load one feasibility JSON object from *path*."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FeasibilityDataError(f"找不到可行性結果檔：{path}") from exc
    except json.JSONDecodeError as exc:
        raise FeasibilityDataError(
            f"無法解析可行性結果檔：第 {exc.lineno} 行"
        ) from exc

    if not isinstance(data, dict):
        raise FeasibilityDataError("可行性結果的最外層必須是 JSON 物件")
    return data


def _required(data: Mapping[str, Any], *keys: str) -> Any:
    value: Any = data
    path: list[str] = []
    for key in keys:
        path.append(key)
        if not isinstance(value, Mapping) or key not in value:
            raise FeasibilityDataError(f"缺少必要欄位：{'.'.join(path)}")
        value = value[key]
    return value


def _integer(data: Mapping[str, Any], *keys: str) -> int:
    value = _required(data, *keys)
    if type(value) is not int:
        raise FeasibilityDataError(f"欄位必須是整數：{'.'.join(keys)}")
    return value


def build_summary(data: Mapping[str, Any]) -> str:
    """Validate *data* and return a human-readable Chinese summary."""
    source_checks = _required(data, "source_file_checks")
    if not isinstance(source_checks, Mapping):
        raise FeasibilityDataError("欄位必須是物件：source_file_checks")

    actual_years = tuple(sorted(source_checks))
    if actual_years != EXPECTED_YEARS:
        raise FeasibilityDataError(
            f"分析年度應為 {EXPECTED_YEARS}，實際為 {actual_years}"
        )

    analytical_rows = sum(
        _integer(data, "source_file_checks", year, "analytical_rows")
        for year in EXPECTED_YEARS
    )
    unique_samples = _integer(data, "all_unique_samples")
    modeling_samples = _integer(data, "modeling_samples")
    test_samples = _integer(data, "temporal_split", "test_samples")
    latest_samples = _integer(
        data, "yearly_modeling_summary", EXPECTED_YEARS[-1], "samples"
    )
    if test_samples != latest_samples:
        raise FeasibilityDataError(
            "測試樣本數不一致："
            f"temporal_split={test_samples}，yearly_modeling_summary={latest_samples}"
        )

    lift = _required(data, "best_nonbaseline_top20_lift")
    if type(lift) not in (int, float):
        raise FeasibilityDataError(
            "欄位必須是數字：best_nonbaseline_top20_lift"
        )
    feasible = _required(data, "model_feasible")
    if type(feasible) is not bool:
        raise FeasibilityDataError("欄位必須是布林值：model_feasible")
    interpretation = _required(data, "interpretation")
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise FeasibilityDataError("欄位必須是非空文字：interpretation")

    conclusion = (
        "資料適合用於 FDA 定向抽樣資料內的相對風險排序"
        if feasible
        else "目前資料不足以支持相對風險排序"
    )
    return "\n".join(
        [
            "FDA農藥殘留專案 第一週可行性摘要",
            "",
            f"分析年度：{EXPECTED_YEARS[0]}–{EXPECTED_YEARS[-1]}",
            f"分析紀錄數：{analytical_rows:,}",
            f"不重複樣本數：{unique_samples:,}",
            f"建模樣本數：{modeling_samples:,}",
            f"測試年度：{EXPECTED_YEARS[-1]}",
            f"測試樣本數：{test_samples:,}",
            f"最佳 Top 20% Lift：{lift:.2f}",
            f"結論：{conclusion}",
            "限制：結果只用於 FDA 定向抽樣資料內的相對排序，"
            "不代表整體市場違規率。",
        ]
    )


def main(results_path: Path | None = None) -> int:
    """Print the feasibility summary and return a process status code."""
    path = DEFAULT_RESULTS_PATH if results_path is None else results_path
    try:
        print(build_summary(load_results(path)))
    except FeasibilityDataError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
