import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from fda_risk.temporal_baseline import (
    BaselineDataError, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, build_baseline, main,
)

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ('sample_id', 'Year', 'ProdCode', 'Country', 'Origin', 'CollMonth', 'violative')


def sample(year, number, label=0, **changes):
    result = dict(zip(FIELDS, (f'{year}-{number}', str(year), '20A14',
                              'United States', 'D', '03', str(label))))
    result.update(changes)
    return result


def fixture():
    # Four training samples: one positive. Test: one positive, one negative.
    return [sample(2019, 1, 1), sample(2020, 2), sample(2021, 3),
            sample(2022, 4), sample(2023, 5, 1), sample(2023, 6)]


def write_csv(path, rows, fields=FIELDS):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


class BaselineTests(unittest.TestCase):
    def test_split_and_fitted_probability_use_training_years_only(self):
        result = build_baseline(FIELDS, fixture())
        self.assertEqual(result['temporal_split'], {
            'train_years': [2019, 2020, 2021, 2022], 'test_year': 2023,
            'train_samples': 4, 'train_violative': 1,
            'test_samples': 2, 'test_violative': 1, 'overlap_samples': 0,
        })
        self.assertEqual(result['baseline_probability'], 0.25)

    def test_constant_baseline_metrics_match_hand_calculation(self):
        metrics = build_baseline(FIELDS, fixture())['metrics']
        self.assertEqual(metrics['average_precision'], 0.5)
        self.assertEqual(metrics['roc_auc'], 0.5)
        self.assertEqual(metrics['brier'], 0.3125)
        self.assertEqual(metrics['precision_at_0_5'], 0.0)
        self.assertEqual(metrics['recall_at_0_5'], 0.0)
        self.assertEqual(metrics['top20_samples'], 1)
        self.assertEqual(metrics['top20_expected_precision'], 0.5)
        self.assertEqual(metrics['top20_expected_recall'], 0.5)
        self.assertEqual(metrics['top20_expected_lift'], 1.0)

    def test_test_labels_cannot_change_fitted_probability_or_vocabulary(self):
        before = build_baseline(FIELDS, fixture())
        changed = fixture() + [sample(2023, 7, 1, Country='New Country', ProdCode='NEW')]
        after = build_baseline(FIELDS, changed)
        self.assertEqual(after['baseline_probability'], before['baseline_probability'])
        self.assertEqual(after['train_categories'], before['train_categories'])
        self.assertEqual(after['test_unseen_counts']['Country'], 1)
        self.assertEqual(after['test_unseen_counts']['ProdCode'], 1)
        self.assertNotEqual(after['metrics']['average_precision'], before['metrics']['average_precision'])

    def test_row_order_does_not_create_top20_ranking_ability(self):
        self.assertEqual(build_baseline(FIELDS, fixture()),
                         build_baseline(FIELDS, list(reversed(fixture()))))

    def test_probability_at_or_above_half_predicts_all_positive(self):
        for positive_count in (2, 3):
            rows = fixture()
            for index in range(4):
                rows[index]['violative'] = str(int(index < positive_count))
            with self.subTest(positive_count=positive_count):
                result = build_baseline(FIELDS, rows)
                self.assertEqual(result['baseline_probability'], positive_count / 4)
                self.assertEqual(result['metrics']['precision_at_0_5'], 0.5)
                self.assertEqual(result['metrics']['recall_at_0_5'], 1.0)

    def test_rejects_leakage_columns_missing_columns_and_duplicate_headers(self):
        for fields in (FIELDS + ('LabClass',), FIELDS[:-1], FIELDS + ('Year',)):
            with self.subTest(fields=fields), self.assertRaises(BaselineDataError):
                build_baseline(fields, fixture())

    def test_rejects_bad_key_duplicates_and_extra_cells(self):
        for changes in ({'sample_id': ''}, {'sample_id': '2023-1'}, {None: ['extra']}):
            rows = fixture()
            rows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(BaselineDataError):
                build_baseline(FIELDS, rows)
        with self.assertRaises(BaselineDataError):
            build_baseline(FIELDS, fixture() + [fixture()[0]])

    def test_rejects_bad_year_label_and_feature_values(self):
        for changes in ({'Year': '2024', 'sample_id': '2024-1'},
                        {'violative': '2'}, {'Country': ''}, {'ProdCode': 'Unknown'},
                        {'Origin': 'Z'}, {'CollMonth': '13'}):
            rows = fixture()
            rows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(BaselineDataError):
                build_baseline(FIELDS, rows)

    def test_rejects_empty_data_missing_years_and_single_class_splits(self):
        for rows in ([], fixture()[1:], fixture()[:-2],
                     [dict(row, violative='0') for row in fixture()],
                     [row for row in fixture() if row['sample_id'] != '2023-5']):
            with self.subTest(rows=rows), self.assertRaises(BaselineDataError):
                build_baseline(FIELDS, rows)

    def test_official_data_matches_split_and_historical_baseline_metrics(self):
        with DEFAULT_INPUT_PATH.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        result = build_baseline(reader.fieldnames, rows)
        split = result['temporal_split']
        self.assertEqual((split['train_samples'], split['train_violative'],
                          split['test_samples'], split['test_violative']),
                         (10572, 933, 3577, 376))
        self.assertEqual(split['overlap_samples'], 0)
        self.assertAlmostEqual(result['baseline_probability'], 933 / 10572)
        self.assertAlmostEqual(result['metrics']['average_precision'], 0.10511601901034387)
        self.assertAlmostEqual(result['metrics']['brier'], 0.09435103715434805)
        self.assertEqual(result['metrics']['roc_auc'], 0.5)
        self.assertEqual(result['metrics']['top20_samples'], 716)
        self.assertAlmostEqual(result['metrics']['top20_expected_precision'], 0.10511601901034387)
        self.assertAlmostEqual(result['metrics']['top20_expected_recall'], 0.200167738328208)
        self.assertEqual(result['metrics']['top20_expected_lift'], 1.0)
        self.assertEqual({field: len(values) for field, values in result['train_categories'].items()},
                         {'ProdCode': 645, 'Country': 108, 'Origin': 3, 'CollMonth': 12})
        self.assertEqual(result['test_unseen_counts'],
                         {'ProdCode': 85, 'Country': 8, 'Origin': 0, 'CollMonth': 0})

    def test_saved_official_report_matches_source_and_current_calculation(self):
        source_bytes = DEFAULT_INPUT_PATH.read_bytes()
        reader = csv.DictReader(io.StringIO(source_bytes.decode('utf-8-sig')))
        rows = list(reader)
        saved = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding='utf-8'))
        provenance = saved.pop('source')
        self.assertEqual(provenance['sha256'], hashlib.sha256(source_bytes).hexdigest())
        self.assertEqual(saved, build_baseline(reader.fieldnames, rows))


class BaselineCommandTests(unittest.TestCase):
    def test_main_saves_reproducible_json_without_touching_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, output = Path(directory) / 'input.csv', Path(directory) / 'report.json'
            write_csv(source, fixture())
            original = source.read_bytes()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(source, output), 0)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['baseline_probability'], 0.25)
            self.assertEqual(result['temporal_split']['train_samples'], 4)
            self.assertEqual(source.read_bytes(), original)
            self.assertIn('訓練', stream.getvalue())
            self.assertIn('期望', stream.getvalue())

    def test_bad_input_preserves_existing_report(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, output = Path(directory) / 'input.csv', Path(directory) / 'report.json'
            write_csv(source, fixture(), fields=FIELDS + ('LabClass',))
            output.write_text('previous', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(source, output), 1)
            self.assertEqual(output.read_text(encoding='utf-8'), 'previous')

    def test_missing_corrupt_file_and_source_overwrite_return_one(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, output = Path(directory) / 'input.csv', Path(directory) / 'report.json'
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(source, output), 1)
                source.write_bytes(b'\xff')
                self.assertEqual(main(source, output), 1)
                write_csv(source, fixture())
                self.assertEqual(main(source, source), 1)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
