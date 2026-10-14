import contextlib
import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fda_risk.modeling_dataset import OUTPUT_FIELDS
from fda_risk.risk_ranking import RankingError, build_ranking, rank_samples, save_outputs, main


def fixture():
    rows = []
    for year in range(2019, 2024):
        for i in range(12):
            rows.append(dict(sample_id=f'{year}-{i}', Year=str(year),
                             ProdCode='NEW' if year == 2023 and i == 0 else f'P{i % 2}',
                             Country='TW', Origin='I', CollMonth='06', violative=str(i % 2)))
    return rows


class RankingPolicyTests(unittest.TestCase):
    def test_cutoff_weights_ignore_labels_and_tied_row_order(self):
        samples = [dict(sample_id=str(i), violative=str(i % 2)) for i in range(6)]
        result = rank_samples(samples, [.9, .8, .8, .8, .1, .1])
        self.assertEqual([row['score_rank'] for row in result], [1, 2, 2, 2, 5, 5])
        self.assertEqual([row['top20_weight'] for row in result], [1, 1 / 3, 1 / 3, 1 / 3, 0, 0])
        changed = [dict(row, violative=str(1 - int(row['violative']))) for row in reversed(samples)]
        again = rank_samples(changed, list(reversed([.9, .8, .8, .8, .1, .1])))
        self.assertEqual([(r['sample_id'], r['risk_score'], r['score_rank'], r['top20_weight']) for r in result],
                         [(r['sample_id'], r['risk_score'], r['score_rank'], r['top20_weight']) for r in again])

    def test_constant_scores_allocate_all_weights_equally(self):
        result = rank_samples([dict(sample_id=str(i)) for i in range(5)], [.5] * 5)
        self.assertEqual([r['top20_weight'] for r in result], [.2] * 5)
        self.assertEqual([r['score_rank'] for r in result], [1] * 5)

    def test_rejects_empty_mismatched_duplicate_or_invalid_inputs(self):
        for samples, scores in [([], []), ([dict(sample_id='a')], []),
                                ([dict(sample_id='a'), dict(sample_id='a')], [.1, .2]),
                                ([dict(sample_id='a')], [math.nan]),
                                ([dict(sample_id='a')], [1.1])]:
            with self.subTest(samples=samples, scores=scores), self.assertRaises(RankingError):
                rank_samples(samples, scores)


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranked, cls.report = build_ranking(OUTPUT_FIELDS, fixture())

    def test_four_contributions_plus_intercept_reconstruct_every_score(self):
        for row in self.ranked:
            summed = row['intercept'] + sum(row[f'{field}_contribution']
                                           for field in ('ProdCode', 'Country', 'Origin', 'CollMonth'))
            self.assertAlmostEqual(summed, row['logit'], delta=1e-10)
            self.assertAlmostEqual(1 / (1 + math.exp(-summed)), row['risk_score'], delta=1e-12)
            self.assertAlmostEqual(row['reconstructed_score'], row['risk_score'], delta=1e-12)
        self.assertLessEqual(self.report['max_score_reconstruction_error'], 1e-12)

    def test_known_encoded_category_uses_its_recorded_coefficient(self):
        coefficients = {(r['field'], r['encoded_category']): r['coefficient']
                        for r in self.report['coefficients']}
        for row in self.ranked:
            if row['ProdCode'] in ('P0', 'P1'):
                self.assertEqual(row['ProdCode_contribution'], coefficients[('ProdCode', row['ProdCode'])])

    def test_unseen_category_has_zero_contribution_without_a_rare_bucket(self):
        unseen = next(row for row in self.ranked if row['ProdCode'] == 'NEW')
        self.assertEqual(unseen['ProdCode_contribution'], 0)
        self.assertEqual(unseen['unseen_features'], 'ProdCode')
        self.assertNotIn('NEW', self.report['preprocessing']['train_categories']['ProdCode'])

    def test_unseen_category_uses_the_fitted_rare_bucket_coefficient(self):
        rows = fixture()
        for row in rows:
            if row['Year'] != '2023' and row['sample_id'].endswith('-11'):
                row['ProdCode'] = 'RARE'  # Four occurrences across train: below min_frequency=5.
        ranked, report = build_ranking(OUTPUT_FIELDS, rows)
        coefficient = next(row['coefficient'] for row in report['coefficients']
                           if row['field'] == 'ProdCode' and row['encoded_category'] == 'infrequent_sklearn')
        unseen = next(row for row in ranked if row['ProdCode'] == 'NEW')
        self.assertEqual(unseen['ProdCode_contribution'], coefficient)

    def test_ranked_weights_and_metrics_agree(self):
        self.assertEqual(len(self.ranked), 12)
        self.assertAlmostEqual(sum(row['top20_weight'] for row in self.ranked), 3)
        positives = sum(row['top20_weight'] * int(row['violative']) for row in self.ranked)
        self.assertAlmostEqual(positives, self.report['metrics']['top20_expected_positives'])
        self.assertEqual(self.report['model'], 'logistic_regression')

    def test_test_labels_do_not_change_scores_or_contributions(self):
        rows = fixture()
        for row in rows:
            if row['Year'] == '2023':
                row['violative'] = str(int(int(row['sample_id'].split('-')[1]) < 10))
        changed, _ = build_ranking(OUTPUT_FIELDS, rows)
        fields = ['sample_id', 'risk_score', 'score_rank', 'top20_weight', 'intercept',
                  'ProdCode_contribution', 'Country_contribution', 'Origin_contribution', 'CollMonth_contribution']
        self.assertEqual([[r[key] for key in fields] for r in self.ranked],
                         [[r[key] for key in fields] for r in changed])

    def test_rejects_additional_leakage_field(self):
        with self.assertRaises(ValueError):
            build_ranking((*OUTPUT_FIELDS, 'LabClass'), fixture())


class PublicationTests(unittest.TestCase):
    def test_main_writes_bom_csv_and_matching_json_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.csv'
            with source.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerows(fixture())
            original = source.read_bytes()
            csv_path, json_path = root / 'ranking.csv', root / 'report.json'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(source, csv_path, json_path), 0)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(csv_path.read_bytes().startswith(b'\xef\xbb\xbf'))
            with csv_path.open(encoding='utf-8-sig', newline='') as handle:
                saved = list(csv.DictReader(handle))
            report = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(len(saved), 12)
            self.assertEqual(len({row['sample_id'] for row in saved}), 12)
            self.assertEqual(report['source']['sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(report['ranking_csv']['sha256'], hashlib.sha256(csv_path.read_bytes()).hexdigest())

    def test_output_path_aliases_cannot_overwrite_source_or_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.csv'
            source.write_text('original')
            for csv_path, json_path in [(source, root / 'r.json'), (root / 'r.csv', source),
                                        (root / 'same', root / 'same')]:
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(source, csv_path, json_path), 1)
            self.assertEqual(source.read_text(), 'original')

    def test_publication_failure_restores_both_previous_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, json_path = root / 'ranking.csv', root / 'report.json'
            csv_path.write_text('old csv')
            json_path.write_text('old json')
            original_replace = Path.replace

            def fail_json_once(path, target):
                if Path(target) == json_path and path.suffix == '.tmp':
                    raise PermissionError('controlled publication denial')
                return original_replace(path, target)

            ranked, report = build_ranking(OUTPUT_FIELDS, fixture())
            with patch.object(Path, 'replace', new=fail_json_once):
                with self.assertRaises(OSError):
                    save_outputs(ranked, report, csv_path, json_path)
            self.assertEqual(csv_path.read_text(), 'old csv')
            self.assertEqual(json_path.read_text(), 'old json')
            self.assertFalse(list(root.glob('*.tmp')))

    def test_publication_failure_without_previous_csv_leaves_no_new_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, json_path = root / 'ranking.csv', root / 'report.json'
            json_path.write_text('old json')
            original_replace = Path.replace

            def fail_json_once(path, target):
                if Path(target) == json_path and path.suffix == '.tmp':
                    raise PermissionError('controlled denial')
                return original_replace(path, target)

            ranked, report = build_ranking(OUTPUT_FIELDS, fixture())
            with patch.object(Path, 'replace', new=fail_json_once), self.assertRaises(OSError):
                save_outputs(ranked, report, csv_path, json_path)
            self.assertFalse(csv_path.exists())
            self.assertEqual(json_path.read_text(), 'old json')

    def test_invalid_staged_json_is_rejected_before_either_output_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, json_path = root / 'ranking.csv', root / 'report.json'
            csv_path.write_text('old csv')
            json_path.write_text('old json')
            ranked, report = build_ranking(OUTPUT_FIELDS, fixture())

            def incomplete_dump(value, handle, **kwargs):
                handle.write('{}')

            with patch('fda_risk.risk_ranking.json.dump', side_effect=incomplete_dump):
                with self.assertRaises(RankingError):
                    save_outputs(ranked, report, csv_path, json_path)
            self.assertEqual(csv_path.read_text(), 'old csv')
            self.assertEqual(json_path.read_text(), 'old json')
            self.assertFalse(list(root.glob('*.tmp')))


class RealOutputTests(unittest.TestCase):
    def test_saved_csv_reconstructs_scores_and_matches_report_and_f07(self):
        root = Path(__file__).resolve().parent.parent
        csv_path = root / 'data/results/fda_risk_ranking_2023.csv'
        report = json.loads((root / 'data/results/fda_risk_ranking_report.json').read_text(encoding='utf-8'))
        f07 = json.loads((root / 'data/results/fda_model_comparison.json').read_text(encoding='utf-8'))
        with csv_path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3577)
        self.assertEqual(len({r['sample_id'] for r in rows}), 3577)
        self.assertEqual({r['Year'] for r in rows}, {'2023'})
        self.assertEqual(report['ranking_csv']['sha256'], hashlib.sha256(csv_path.read_bytes()).hexdigest())
        self.assertEqual(report['source']['sha256'], f07['source']['sha256'])
        self.assertAlmostEqual(math.fsum(float(r['top20_weight']) for r in rows), 716)
        self.assertAlmostEqual(math.fsum(float(r['top20_weight']) * int(r['violative']) for r in rows), 188)
        for key, value in f07['models']['logistic_regression']['metrics'].items():
            self.assertEqual(report['metrics'][key], value)
        for previous, row in zip(rows, rows[1:]):
            self.assertGreaterEqual(float(previous['risk_score']), float(row['risk_score']))
            if float(previous['risk_score']) == float(row['risk_score']):
                self.assertEqual(previous['score_rank'], row['score_rank'])
        for row in rows:
            z = float(row['intercept']) + math.fsum(float(row[f'{field}_contribution'])
                                                    for field in ('ProdCode', 'Country', 'Origin', 'CollMonth'))
            self.assertAlmostEqual(z, float(row['logit']), delta=1e-10)
            self.assertAlmostEqual(1 / (1 + math.exp(-z)), float(row['risk_score']), delta=1e-12)


if __name__ == '__main__':
    unittest.main()
