import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from fda_risk.sample_dataset import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SOURCE_ROOT,
    SampleDataError,
    collapse_records,
    load_records,
    main,
    write_samples,
)


def _row(**changes):
    row = {
        "Year": "2023",
        "SplNo": "100",
        "ProdCode": "20A01",
        "ProdName": "Blackberries",
        "Origin": "I",
        "Country": "Mexico",
        "PAC": "04004A",
        "CollDate": "20230613",
        "LabClass": "1",
        "VioType": "",
    }
    row.update(changes)
    return row


def _write_source(root: Path, year: str, rows, fieldnames=None):
    directory = root / year
    directory.mkdir(parents=True)
    path = directory / f"SampleData{year}.txt"
    names = fieldnames or list(_row().keys())
    with path.open("w", encoding="latin-1", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=names, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


class CollapseRecordsTests(unittest.TestCase):
    def test_collapses_multiple_analytical_rows_into_one_sample(self):
        samples = collapse_records(
            [_row(VioType="", LabClass="3"), _row(VioType=" n ", LabClass="3")]
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sample_id"], "2023-100")
        self.assertEqual(samples[0]["analytical_rows"], 2)
        self.assertEqual(samples[0]["n_or_x_rows"], 1)
        self.assertEqual(samples[0]["violative"], 1)
        self.assertEqual(samples[0]["CollMonth"], "06")

    def test_same_splno_in_different_years_remains_two_samples(self):
        samples = collapse_records([_row(Year="2022"), _row(Year="2023")])

        self.assertEqual(
            [sample["sample_id"] for sample in samples],
            ["2022-100", "2023-100"],
        )

    def test_rejects_conflicting_sample_level_value(self):
        with self.assertRaisesRegex(
            SampleDataError, "2023-100.*ProdName.*Blackberries.*Strawberries"
        ):
            collapse_records([_row(), _row(ProdName="Strawberries")])

    def test_rejects_blank_sample_key(self):
        with self.assertRaisesRegex(SampleDataError, "Year.*SplNo.*空白"):
            collapse_records([_row(SplNo="")])


class SourceAndOutputTests(unittest.TestCase):
    def test_load_records_reads_requested_year_and_validates_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root, "2023", [_row()])

            records = list(load_records(root, years=("2023",)))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ProdName"], "Blackberries")

    def test_load_records_rejects_missing_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [name for name in _row() if name != "LabClass"]
            _write_source(root, "2023", [_row()], names)

            with self.assertRaisesRegex(SampleDataError, "LabClass"):
                list(load_records(root, years=("2023",)))

    def test_load_records_rejects_unexpected_year(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root, "2023", [_row(Year="2022")])

            with self.assertRaisesRegex(SampleDataError, "2023.*2022"):
                list(load_records(root, years=("2023",)))

    def test_write_samples_round_trips_excel_friendly_csv(self):
        samples = collapse_records([_row()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            write_samples(samples, path)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(saved[0]["sample_id"], "2023-100")
        self.assertEqual(saved[0]["analytical_rows"], "1")


class SampleDatasetCommandTests(unittest.TestCase):
    def test_default_paths_are_anchored_to_project(self):
        project_root = Path(__file__).resolve().parent.parent
        self.assertEqual(
            DEFAULT_SOURCE_ROOT, project_root / "_fda_feasibility" / "extracted"
        )
        self.assertEqual(
            DEFAULT_OUTPUT_PATH,
            project_root / "data" / "processed" / "fda_samples_2019_2023.csv",
        )

    def test_main_builds_file_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sources"
            output_path = Path(directory) / "processed" / "samples.csv"
            _write_source(root, "2023", [_row(), _row(VioType="X")])
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = main(root, output_path, years=("2023",))

            self.assertEqual(result, 0)
            self.assertTrue(output_path.exists())
            self.assertIn("來源分析列數：2", output.getvalue())
            self.assertIn("輸出樣本數：1", output.getvalue())

    def test_main_reports_error_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing"
            output_path = Path(directory) / "samples.csv"
            errors = io.StringIO()

            with contextlib.redirect_stderr(errors):
                result = main(root, output_path, years=("2023",))

            self.assertEqual(result, 1)
            self.assertFalse(output_path.exists())
            self.assertIn("錯誤：", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
