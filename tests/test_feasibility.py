import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from fda_risk import feasibility
from fda_risk.feasibility import (
    DEFAULT_RESULTS_PATH,
    FeasibilityDataError,
    build_summary,
    load_results,
    main,
)


def _valid_results():
    return {
        "source_file_checks": {
            str(year): {"analytical_rows": year - 2000}
            for year in range(2019, 2024)
        },
        "all_unique_samples": 15150,
        "modeling_samples": 14149,
        "temporal_split": {"test_samples": 3577},
        "yearly_modeling_summary": {
            str(year): {"samples": 3577 if year == 2023 else 1000}
            for year in range(2019, 2024)
        },
        "best_nonbaseline_top20_lift": 2.56433867229288,
        "model_feasible": True,
        "interpretation": "Relative prioritization only.",
    }


class FeasibilityCoreTests(unittest.TestCase):
    def test_load_results_reads_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(_valid_results()), encoding="utf-8")
            self.assertEqual(load_results(path)["modeling_samples"], 14149)

    def test_load_results_reports_missing_file(self):
        with self.assertRaisesRegex(FeasibilityDataError, "找不到可行性結果檔"):
            load_results(Path("missing-results.json"))

    def test_load_results_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(FeasibilityDataError, "無法解析"):
                load_results(path)

    def test_build_summary_formats_verified_values(self):
        summary = build_summary(_valid_results())
        self.assertIn("分析年度：2019–2023", summary)
        self.assertIn("分析紀錄數：105", summary)
        self.assertIn("測試樣本數：3,577", summary)
        self.assertIn("最佳 Top 20% Lift：2.56", summary)
        self.assertIn("不代表整體市場違規率", summary)

    def test_build_summary_lists_missing_nested_field(self):
        data = _valid_results()
        del data["temporal_split"]["test_samples"]
        with self.assertRaisesRegex(
            FeasibilityDataError, "temporal_split.test_samples"
        ):
            build_summary(data)

    def test_build_summary_rejects_incomplete_years(self):
        data = _valid_results()
        del data["source_file_checks"]["2020"]
        with self.assertRaisesRegex(FeasibilityDataError, "2019.*2023"):
            build_summary(data)

    def test_build_summary_rejects_string_metric(self):
        data = _valid_results()
        data["all_unique_samples"] = "15150"
        with self.assertRaisesRegex(FeasibilityDataError, "all_unique_samples"):
            build_summary(data)

    def test_build_summary_rejects_mismatched_test_count(self):
        data = _valid_results()
        data["temporal_split"]["test_samples"] = 1
        with self.assertRaisesRegex(FeasibilityDataError, "測試樣本數不一致"):
            build_summary(data)

    def test_build_summary_does_not_claim_success_when_infeasible(self):
        data = _valid_results()
        data["model_feasible"] = False
        summary = build_summary(data)
        self.assertIn("目前資料不足以支持相對風險排序", summary)
        self.assertNotIn("結論：資料適合", summary)


class FeasibilityCommandTests(unittest.TestCase):
    def test_main_prints_summary_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(_valid_results()), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(path)
        self.assertEqual(result, 0)
        self.assertIn("第一週可行性摘要", output.getvalue())

    def test_main_prints_error_and_returns_one(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            result = main(Path("missing-results.json"))
        self.assertEqual(result, 1)
        self.assertIn("錯誤：", errors.getvalue())

    def test_default_path_is_anchored_to_project(self):
        self.assertEqual(
            DEFAULT_RESULTS_PATH,
            Path(feasibility.__file__).resolve().parent.parent
            / "_fda_feasibility"
            / "feasibility-results.json",
        )
