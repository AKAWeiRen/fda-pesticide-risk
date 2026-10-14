import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from fda_risk.sample_overview import (
    DEFAULT_INPUT_PATH,
    SampleOverviewError,
    build_overview,
    format_overview,
    load_samples,
    main,
)


FIELDNAMES = ("sample_id", "Year", "LabClass", "Country", "Notes")


def _row(**changes):
    row = {
        "sample_id": "2019-100",
        "Year": "2019",
        "LabClass": "1",
        "Country": "Taiwan",
        "Notes": "checked",
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


class BuildOverviewTests(unittest.TestCase):
    def test_counts_blank_and_unknown_separately(self):
        overview = build_overview(
            FIELDNAMES,
            [
                _row(LabClass="3", Country="Unknown", Notes=""),
                _row(sample_id="2019-101"),
                _row(
                    sample_id="2020-102",
                    Year="2020",
                    LabClass="3",
                    Country="",
                    Notes="Unknown",
                ),
            ],
            expected_years=("2019", "2020"),
        )

        self.assertEqual(overview["sample_count"], 3)
        self.assertEqual(overview["missing"]["Country"], 1)
        self.assertEqual(overview["missing"]["Notes"], 1)
        self.assertEqual(overview["unknown"]["Country"], 1)
        self.assertEqual(overview["unknown"]["Notes"], 1)

    def test_calculates_yearly_labclass3_rates(self):
        overview = build_overview(
            FIELDNAMES,
            [
                _row(LabClass="3"),
                _row(sample_id="2019-101"),
                _row(sample_id="2020-102", Year="2020", LabClass="3"),
            ],
            expected_years=("2019", "2020"),
        )

        self.assertEqual(
            overview["yearly"],
            [
                {
                    "year": "2019",
                    "samples": 2,
                    "labclass3": 1,
                    "rate": 0.5,
                },
                {
                    "year": "2020",
                    "samples": 1,
                    "labclass3": 1,
                    "rate": 1.0,
                },
            ],
        )

    def test_rejects_duplicate_sample_id(self):
        with self.assertRaisesRegex(SampleOverviewError, "sample_id.*重複"):
            build_overview(
                FIELDNAMES,
                [_row(), _row()],
                expected_years=("2019",),
            )

    def test_rejects_blank_sample_id(self):
        with self.assertRaisesRegex(SampleOverviewError, "sample_id.*空白"):
            build_overview(
                FIELDNAMES,
                [_row(sample_id=" ")],
                expected_years=("2019",),
            )

    def test_rejects_empty_data(self):
        with self.assertRaisesRegex(SampleOverviewError, "沒有任何樣本"):
            build_overview(FIELDNAMES, [], expected_years=("2019",))

    def test_rejects_missing_expected_year(self):
        with self.assertRaisesRegex(SampleOverviewError, "缺少年度.*2020"):
            build_overview(
                FIELDNAMES,
                [_row()],
                expected_years=("2019", "2020"),
            )

    def test_rejects_unexpected_year(self):
        with self.assertRaisesRegex(SampleOverviewError, "年度.*2018"):
            build_overview(
                FIELDNAMES,
                [_row(Year="2018")],
                expected_years=("2019",),
            )

    def test_rejects_invalid_labclass(self):
        with self.assertRaisesRegex(SampleOverviewError, "LabClass.*9"):
            build_overview(
                FIELDNAMES,
                [_row(LabClass="9")],
                expected_years=("2019",),
            )

    def test_formats_human_readable_summary(self):
        overview = build_overview(
            FIELDNAMES,
            [_row(LabClass="3", Country="Unknown", Notes="")],
            expected_years=("2019",),
        )

        summary = format_overview(overview)

        self.assertIn("總樣本數：1", summary)
        self.assertIn("Country：空白 0；Unknown 1", summary)
        self.assertIn("Notes：空白 1；Unknown 0", summary)
        self.assertIn("2019：樣本 1；LabClass 3 為 1；比例 100.00%", summary)


class InputAndCommandTests(unittest.TestCase):
    def test_default_path_is_anchored_to_project(self):
        project_root = Path(__file__).resolve().parent.parent
        self.assertEqual(
            DEFAULT_INPUT_PATH,
            project_root / "data" / "processed" / "fda_samples_2019_2023.csv",
        )

    def test_load_samples_rejects_missing_required_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            _write_csv(path, [_row()], fieldnames=("sample_id", "Year"))

            with self.assertRaisesRegex(SampleOverviewError, "LabClass"):
                load_samples(path)

    def test_main_reports_error_for_corrupt_utf8_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            path.write_bytes(b"sample_id,Year,LabClass,\xff\n")
            errors = io.StringIO()

            with contextlib.redirect_stderr(errors):
                result = main(path, expected_years=("2019",))

        self.assertEqual(result, 1)
        self.assertIn("錯誤：", errors.getvalue())
        self.assertIn("無法解析樣本層級資料", errors.getvalue())

    def test_main_prints_summary_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            _write_csv(path, [_row(LabClass="3")])
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = main(path, expected_years=("2019",))

        self.assertEqual(result, 0)
        self.assertIn("FDA農藥殘留專案 樣本層級資料品質與年度概況", output.getvalue())
        self.assertIn("比例 100.00%", output.getvalue())

    def test_main_reports_error_for_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            errors = io.StringIO()

            with contextlib.redirect_stderr(errors):
                result = main(path, expected_years=("2019",))

        self.assertEqual(result, 1)
        self.assertIn("錯誤：", errors.getvalue())


class ProductionAcceptanceTests(unittest.TestCase):
    def test_official_sample_csv_matches_f03_acceptance_numbers(self):
        fieldnames, rows = load_samples(DEFAULT_INPUT_PATH)
        overview = build_overview(fieldnames, rows)

        self.assertEqual(overview["sample_count"], 15150)
        self.assertTrue(all(count == 0 for count in overview["missing"].values()))
        self.assertTrue(all(count == 0 for count in overview["unknown"].values()))

        expected = {
            "2019": (4692, 362, 0.07715260017050298),
            "2020": (2180, 217, 0.09954128440366973),
            "2021": (1447, 125, 0.08638562543192743),
            "2022": (3030, 250, 0.08250825082508251),
            "2023": (3801, 382, 0.10049986845566956),
        }
        actual = {item["year"]: item for item in overview["yearly"]}
        self.assertEqual(set(actual), set(expected))
        for year, (samples, labclass3, rate) in expected.items():
            self.assertEqual(actual[year]["samples"], samples)
            self.assertEqual(actual[year]["labclass3"], labclass3)
            self.assertAlmostEqual(actual[year]["rate"], rate)


if __name__ == "__main__":
    unittest.main()
