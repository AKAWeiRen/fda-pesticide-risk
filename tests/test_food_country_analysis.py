import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from fda_risk.food_country_analysis import (
    DEFAULT_INPUT_PATH,
    FoodCountryAnalysisError,
    build_analysis,
    format_analysis,
    load_samples,
    main,
)


FIELDNAMES = ("sample_id", "ProdName", "Country", "LabClass")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _row(**changes):
    row = {
        "sample_id": "2019-100",
        "ProdName": "Hot Pepper",
        "Country": "Mexico",
        "LabClass": "1",
    }
    row.update(changes)
    return row


def _write_csv(path: Path, rows, fieldnames=FIELDNAMES):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


class BuildAnalysisTests(unittest.TestCase):
    def test_builds_volume_and_rate_rankings_with_normalized_names(self):
        rows = [
            _row(LabClass="3"),
            _row(sample_id="2019-101", ProdName=" Hot   Pepper "),
            _row(sample_id="2019-102", Country="Canada"),
            _row(
                sample_id="2019-103",
                ProdName="Basmati Rice",
                Country="Canada",
                LabClass="3",
            ),
            _row(
                sample_id="2019-104",
                ProdName="Basmati Rice",
                Country="Canada",
                LabClass="3",
            ),
        ]

        analysis = build_analysis(
            FIELDNAMES,
            rows,
            top_n=2,
            min_rate_samples=2,
        )

        self.assertEqual(analysis["sample_count"], 5)
        self.assertEqual(analysis["products"]["group_count"], 2)
        self.assertEqual(
            analysis["products"]["volume_top"],
            [
                {
                    "name": "Hot Pepper",
                    "samples": 3,
                    "violative": 1,
                    "rate": 1 / 3,
                },
                {
                    "name": "Basmati Rice",
                    "samples": 2,
                    "violative": 2,
                    "rate": 1.0,
                },
            ],
        )
        self.assertEqual(
            [item["name"] for item in analysis["products"]["rate_top"]],
            ["Basmati Rice", "Hot Pepper"],
        )
        self.assertEqual(analysis["countries"]["group_count"], 2)
        self.assertEqual(
            [item["name"] for item in analysis["countries"]["rate_top"]],
            ["Canada", "Mexico"],
        )

    def test_rate_ranking_excludes_groups_below_minimum_sample_count(self):
        rows = [
            _row(ProdName="Small", Country="Smallland", LabClass="3"),
            _row(
                sample_id="2019-101",
                ProdName="Large",
                Country="Largeland",
                LabClass="3",
            ),
            _row(
                sample_id="2019-102",
                ProdName="Large",
                Country="Largeland",
            ),
        ]

        analysis = build_analysis(
            FIELDNAMES,
            rows,
            min_rate_samples=2,
        )

        self.assertEqual(
            [item["name"] for item in analysis["products"]["rate_top"]],
            ["Large"],
        )
        self.assertEqual(
            [item["name"] for item in analysis["countries"]["rate_top"]],
            ["Largeland"],
        )

    def test_rejects_duplicate_sample_id(self):
        with self.assertRaisesRegex(FoodCountryAnalysisError, "sample_id.*重複"):
            build_analysis(FIELDNAMES, [_row(), _row()])

    def test_rejects_empty_data(self):
        with self.assertRaisesRegex(FoodCountryAnalysisError, "沒有任何樣本"):
            build_analysis(FIELDNAMES, [])

    def test_rejects_blank_sample_id(self):
        with self.assertRaisesRegex(FoodCountryAnalysisError, "sample_id.*空白"):
            build_analysis(FIELDNAMES, [_row(sample_id=" ")])

    def test_rejects_blank_dimension_value(self):
        with self.assertRaisesRegex(FoodCountryAnalysisError, "ProdName.*空白"):
            build_analysis(FIELDNAMES, [_row(ProdName=" ")])

    def test_rejects_invalid_labclass(self):
        with self.assertRaisesRegex(FoodCountryAnalysisError, "LabClass.*9"):
            build_analysis(FIELDNAMES, [_row(LabClass="9")])

    def test_rejects_nonpositive_ranking_parameters(self):
        with self.assertRaisesRegex(FoodCountryAnalysisError, "top_n"):
            build_analysis(FIELDNAMES, [_row()], top_n=0)
        with self.assertRaisesRegex(FoodCountryAnalysisError, "min_rate_samples"):
            build_analysis(FIELDNAMES, [_row()], min_rate_samples=0)

    def test_formats_four_rankings_and_scope_warning(self):
        analysis = build_analysis(
            FIELDNAMES,
            [
                _row(LabClass="3"),
                _row(sample_id="2019-101"),
            ],
            top_n=1,
            min_rate_samples=2,
        )

        summary = format_analysis(analysis)

        self.assertIn("食品樣本數 Top 1", summary)
        self.assertIn("食品不利結果比例 Top 1", summary)
        self.assertIn("來源國樣本數 Top 1", summary)
        self.assertIn("來源國不利結果比例 Top 1", summary)
        self.assertIn("Hot Pepper：樣本 2；LabClass 3 為 1；比例 50.00%", summary)
        self.assertIn("不代表整體市場違規率", summary)


class InputAndCommandTests(unittest.TestCase):
    def test_default_path_is_anchored_to_project(self):
        self.assertEqual(
            DEFAULT_INPUT_PATH,
            PROJECT_ROOT / "data" / "processed" / "fda_samples_2019_2023.csv",
        )

    def test_load_samples_rejects_missing_required_column(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "samples.csv"
            _write_csv(path, [_row()], fieldnames=("sample_id", "ProdName"))

            with self.assertRaisesRegex(FoodCountryAnalysisError, "Country.*LabClass"):
                load_samples(path)

    def test_main_prints_summary_and_returns_zero(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "samples.csv"
            _write_csv(path, [_row(LabClass="3")])
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = main(path, top_n=1, min_rate_samples=1)

        self.assertEqual(result, 0)
        self.assertIn("食品與來源國探索分析", output.getvalue())
        self.assertIn("比例 100.00%", output.getvalue())

    def test_main_reports_missing_file(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            errors = io.StringIO()

            with contextlib.redirect_stderr(errors):
                result = main(Path(directory) / "missing.csv")

        self.assertEqual(result, 1)
        self.assertIn("錯誤：", errors.getvalue())


class ProductionAcceptanceTests(unittest.TestCase):
    def test_official_sample_csv_matches_f04_acceptance_numbers(self):
        fieldnames, rows = load_samples(DEFAULT_INPUT_PATH)
        analysis = build_analysis(fieldnames, rows)

        self.assertEqual(analysis["sample_count"], 15150)
        self.assertEqual(analysis["products"]["group_count"], 813)
        self.assertEqual(analysis["countries"]["group_count"], 114)

        product_volume = analysis["products"]["volume_top"][0]
        self.assertEqual(
            (product_volume["name"], product_volume["samples"], product_volume["violative"]),
            ("Pepper, Hot, N.E.C. (Fruit Used as Vegetable)", 273, 65),
        )
        self.assertAlmostEqual(product_volume["rate"], 65 / 273)

        product_rate = analysis["products"]["rate_top"][0]
        self.assertEqual(
            (product_rate["name"], product_rate["samples"], product_rate["violative"]),
            ("Rice, Basmati, Processed (Packaged)", 260, 98),
        )
        self.assertAlmostEqual(product_rate["rate"], 98 / 260)

        country_volume = analysis["countries"]["volume_top"][0]
        self.assertEqual(
            (country_volume["name"], country_volume["samples"], country_volume["violative"]),
            ("United States", 4180, 109),
        )
        self.assertAlmostEqual(country_volume["rate"], 109 / 4180)

        country_rate = analysis["countries"]["rate_top"][0]
        self.assertEqual(
            (country_rate["name"], country_rate["samples"], country_rate["violative"]),
            ("Pakistan", 157, 56),
        )
        self.assertAlmostEqual(country_rate["rate"], 56 / 157)


if __name__ == "__main__":
    unittest.main()
