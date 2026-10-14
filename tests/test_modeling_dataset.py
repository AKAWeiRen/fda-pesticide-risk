import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from fda_risk.modeling_dataset import (
    DEFAULT_INPUT_PATH, ModelingDataError, build_modeling_samples,
    load_samples, main, validate_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIELDS = ('sample_id', 'Year', 'SplNo', 'ProdCode', 'Country', 'Origin',
          'PAC', 'CollDate', 'LabClass', 'CollMonth', 'violative')


def row(**changes):
    result = dict(zip(FIELDS, ('2019-100', '2019', '100', '20A14',
                              'United States', 'D', '04004A', '20190326',
                              '1', '03', '0')))
    result.update(changes)
    return result


def write_csv(path, rows, fields=FIELDS):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


class ModelingDatasetTests(unittest.TestCase):
    def test_filters_scope_and_exports_only_allowed_roles(self):
        samples, audit = build_modeling_samples(FIELDS, [
            row(LabClass='3', violative='1', n_or_x_rows='5', analytical_rows='9'),
            row(sample_id='2019-101', SplNo='101', PAC='71003A'),
        ])
        self.assertEqual(samples, [{
            'sample_id': '2019-100', 'Year': '2019', 'ProdCode': '20A14',
            'Country': 'United States', 'Origin': 'D', 'CollMonth': '03',
            'violative': 1,
        }])
        self.assertEqual((audit['input_samples'], audit['modeling_samples'],
                          audit['excluded_samples'], audit['violative_samples']),
                         (2, 1, 1, 1))
        self.assertEqual(audit['leakage_feature_count'], 0)

    def test_rejects_label_mismatch_including_excluded_rows(self):
        for pac in ('04004A', '71003A'):
            with self.subTest(pac=pac), self.assertRaisesRegex(ModelingDataError, 'violative'):
                build_modeling_samples(FIELDS, [row(PAC=pac, LabClass='3')])

    def test_rejects_invalid_or_inconsistent_keys(self):
        for changes in ({'sample_id': ''}, {'sample_id': '2019-101'},
                        {'Year': '2024', 'sample_id': '2024-100'}, {'SplNo': ''}):
            with self.subTest(changes=changes), self.assertRaises(ModelingDataError):
                build_modeling_samples(FIELDS, [row(**changes)])
        with self.assertRaisesRegex(ModelingDataError, '重複'):
            build_modeling_samples(FIELDS, [row(), row()])

    def test_rejects_invalid_dates_and_month_mismatch(self):
        for changes in ({'CollDate': '20190230'}, {'CollDate': '2019-03-26'},
                        {'CollMonth': '13'}, {'CollMonth': '04'}):
            with self.subTest(changes=changes), self.assertRaises(ModelingDataError):
                build_modeling_samples(FIELDS, [row(**changes)])

    def test_rejects_invalid_labels_and_origin(self):
        for changes in ({'LabClass': '9'}, {'violative': '2'}, {'Origin': 'Z'}):
            with self.subTest(changes=changes), self.assertRaises(ModelingDataError):
                build_modeling_samples(FIELDS, [row(**changes)])

    def test_rejects_missing_or_unknown_features(self):
        for field in ('ProdCode', 'Country', 'Origin', 'CollMonth'):
            for value in ('', 'Unknown'):
                with self.subTest(field=field, value=value), self.assertRaises(ModelingDataError):
                    build_modeling_samples(FIELDS, [row(**{field: value})])

    def test_rejects_missing_columns_and_empty_scope(self):
        for fields, rows in ((FIELDS[:-1], [row()]), (FIELDS, []),
                             (FIELDS, [row(PAC='71003A')])):
            with self.subTest(fields=fields, rows=rows), self.assertRaises(ModelingDataError):
                build_modeling_samples(fields, rows)

    def test_feature_guard_rejects_outcome_metadata_and_unreviewed_fields(self):
        for field in ('LabClass', 'violative', 'n_or_x_rows', 'analytical_rows',
                      'VioType', 'Found', 'CmplDate', 'sample_id', 'Year',
                      'ProdName', 'new_column'):
            with self.subTest(field=field), self.assertRaises(ModelingDataError):
                validate_features(('ProdCode', field))
        validate_features(('ProdCode', 'Country', 'Origin', 'CollMonth'))


class ModelingCommandTests(unittest.TestCase):
    def test_main_writes_clean_csv_and_summary_without_changing_source(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            source, output = Path(directory) / 'source.csv', Path(directory) / 'out.csv'
            write_csv(source, [row(LabClass='3', violative='1')])
            before = source.read_bytes()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(source, output), 0)
            self.assertEqual(source.read_bytes(), before)
            with output.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                actual = list(reader)
                self.assertEqual(reader.fieldnames, ['sample_id', 'Year', 'ProdCode',
                                 'Country', 'Origin', 'CollMonth', 'violative'])
            self.assertEqual(actual[0]['violative'], '1')
            self.assertIn('洩漏特徵數：0', stream.getvalue())

    def test_invalid_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            source, output = Path(directory) / 'source.csv', Path(directory) / 'out.csv'
            write_csv(source, [row(violative='1')])
            output.write_text('previous output', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(source, output), 1)
            self.assertEqual(output.read_text(encoding='utf-8'), 'previous output')

    def test_missing_file_or_bad_encoding_returns_one(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            source, output = Path(directory) / 'source.csv', Path(directory) / 'out.csv'
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(source, output), 1)
                source.write_bytes(b'sample_id,\xff\n')
                self.assertEqual(main(source, output), 1)
            self.assertFalse(output.exists())

    def test_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            source = Path(directory) / 'source.csv'
            write_csv(source, [row()])
            before = source.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(source, source), 1)
            self.assertEqual(source.read_bytes(), before)

    def test_official_data_acceptance(self):
        fields, rows = load_samples(DEFAULT_INPUT_PATH)
        samples, audit = build_modeling_samples(fields, rows)
        self.assertEqual((audit['input_samples'], len(samples), audit['excluded_samples']),
                         (15150, 14149, 1001))
        self.assertEqual(len({r['sample_id'] for r in samples}), 14149)
        self.assertEqual(audit['violative_samples'], 1309)
        self.assertEqual(audit['yearly_samples'],
                         {'2019': 4327, '2020': 2078, '2021': 1367, '2022': 2800, '2023': 3577})
        self.assertEqual(audit['yearly_violative'],
                         {'2019': 349, '2020': 215, '2021': 124, '2022': 245, '2023': 376})
        self.assertEqual(audit['feature_missing_counts'],
                         {'ProdCode': 0, 'Country': 0, 'Origin': 0, 'CollMonth': 0})
        self.assertEqual(audit['leakage_feature_count'], 0)


if __name__ == '__main__':
    unittest.main()
