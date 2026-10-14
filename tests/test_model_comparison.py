import contextlib
import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from fda_risk.modeling_dataset import OUTPUT_FIELDS
from fda_risk.model_comparison import ComparisonError, evaluate_scores, fit_comparison, main


def fixture():
    rows = []
    for year in range(2019, 2024):
        for i in range(12):
            rows.append(dict(sample_id=f'{year}-{i}', Year=str(year),
                             ProdCode='TEST_ONLY' if year == 2023 else f'P{i % 2}',
                             Country='TEST_COUNTRY' if year == 2023 else 'TW',
                             Origin='I', CollMonth='06', violative=str(i % 2)))
    return rows


class ScoreTests(unittest.TestCase):
    def test_perfect_ranking_and_threshold_counts(self):
        result = evaluate_scores([1, 0, 1, 0, 0], [.9, .8, .7, .2, .1])
        self.assertEqual(result['confusion_matrix'], [[2, 1], [0, 2]])
        self.assertAlmostEqual(result['precision_at_0_5'], 2 / 3)
        self.assertEqual(result['recall_at_0_5'], 1)
        self.assertAlmostEqual(result['f1_at_0_5'], .8)
        self.assertAlmostEqual(result['average_precision'], 5 / 6)
        self.assertAlmostEqual(result['roc_auc'], 5 / 6)
        self.assertAlmostEqual(result['brier'], .158)
        self.assertEqual(result['top20_samples'], 1)
        self.assertEqual(result['top20_expected_positives'], 1)
        self.assertAlmostEqual(result['top20_expected_lift'], 2.5)

    def test_constant_scores_do_not_gain_from_input_order(self):
        for labels in ([1, 1, 0, 0, 0], [0, 0, 0, 1, 1]):
            result = evaluate_scores(labels, [.1] * 5)
            self.assertAlmostEqual(result['top20_expected_positives'], .4)
            self.assertAlmostEqual(result['top20_expected_lift'], 1)
            self.assertAlmostEqual(result['top20_expected_recall'], .2)
            self.assertEqual(result['roc_auc'], .5)
            self.assertEqual(result['precision_at_0_5'], 0)

    def test_cutoff_ties_receive_fractional_expected_credit(self):
        result = evaluate_scores([1, 0, 1, 0, 0, 0], [.9, .8, .8, .8, .1, .1])
        self.assertEqual(result['top20_samples'], 2)
        self.assertEqual(result['top20_boundary_tied_samples'], 3)
        self.assertEqual(result['top20_boundary_slots'], 1)
        self.assertAlmostEqual(result['top20_expected_positives'], 4 / 3)
        self.assertAlmostEqual(result['top20_expected_lift'], 2)

    def test_threshold_includes_exact_half(self):
        self.assertEqual(evaluate_scores([0, 1], [.5, .5])['confusion_matrix'], [[0, 1], [0, 1]])

    def test_invalid_scores_or_labels_rejected(self):
        cases = [([], []), ([0, 1], [.1]), ([0, 1], [math.nan, .2]),
                 ([0, 1], [math.inf, .2]), ([0, 1], [-.1, .2]),
                 ([0, 1], [.1, 1.1]), ([0, 2], [.1, .2]), ([0, 0], [.1, .2])]
        for labels, scores in cases:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ComparisonError):
                evaluate_scores(labels, scores)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.pipelines = fit_comparison(OUTPUT_FIELDS, fixture())

    def test_all_three_models_and_baseline_share_temporal_split(self):
        self.assertEqual(set(self.report['models']),
                         {'prevalence_baseline', 'logistic_regression', 'decision_tree', 'xgboost'})
        self.assertEqual(self.report['temporal_split']['train_samples'], 48)
        self.assertEqual(self.report['temporal_split']['test_samples'], 12)
        self.assertEqual(self.report['temporal_split']['overlap_samples'], 0)
        self.assertEqual(self.report['baseline_probability'], .5)
        for result in self.report['models'].values():
            self.assertEqual(sum(map(sum, result['metrics']['confusion_matrix'])), 12)
            self.assertEqual(result['metrics']['top20_samples'], 3)

    def test_fitted_encoders_exclude_test_only_categories(self):
        for name, pipeline in self.pipelines.items():
            encoder = pipeline.named_steps['preprocess']
            self.assertNotIn('TEST_ONLY', encoder.categories_[0])
            self.assertNotIn('TEST_COUNTRY', encoder.categories_[1])
            audit = self.report['models'][name]['preprocessing']
            self.assertEqual(audit['test_unseen_counts']['ProdCode'], 12)
            self.assertEqual(audit['test_unseen_counts']['Country'], 12)
            self.assertEqual(audit['train_categories']['ProdCode'], ['P0', 'P1'])

    def test_test_labels_cannot_change_fitted_predictions(self):
        import numpy as np
        rows = fixture()
        for row in rows:
            if row['Year'] == '2023':
                # Change both identity and prevalence (6/12 -> 10/12) without losing a class.
                row['violative'] = str(int(int(row['sample_id'].split('-')[1]) < 10))
        _, altered = fit_comparison(OUTPUT_FIELDS, rows)
        probes = [['P0', 'TW', 'I', '06'], ['P1', 'TW', 'I', '06'],
                  ['TEST_ONLY', 'TEST_COUNTRY', 'I', '06']]
        for name, pipeline in self.pipelines.items():
            np.testing.assert_allclose(pipeline.predict_proba(probes),
                                       altered[name].predict_proba(probes), atol=1e-12)

    def test_refuses_leakage_columns_and_bad_years(self):
        with self.assertRaises(ValueError):
            fit_comparison((*OUTPUT_FIELDS, 'LabClass'), fixture())
        rows = fixture()
        rows[0]['Year'] = '2024'
        with self.assertRaises(ValueError):
            fit_comparison(OUTPUT_FIELDS, rows)

    def test_can_fit_only_the_explanation_model_without_changing_recipe(self):
        report, pipelines = fit_comparison(OUTPUT_FIELDS, fixture(),
                                          model_names=('logistic_regression',))
        self.assertEqual(set(pipelines), {'logistic_regression'})
        self.assertEqual(set(report['models']), {'prevalence_baseline', 'logistic_regression'})
        self.assertEqual(report['models']['logistic_regression'],
                         self.report['models']['logistic_regression'])

    def test_rejects_empty_duplicate_or_unapproved_model_request(self):
        for names in ((), ('unknown',), ('logistic_regression', 'logistic_regression')):
            with self.subTest(names=names), self.assertRaises(ComparisonError):
                fit_comparison(OUTPUT_FIELDS, fixture(), model_names=names)

    def test_missing_input_returns_failure_without_replacing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'report.json'
            output.write_text('previous', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(root / 'missing.csv', output), 1)
            self.assertEqual(output.read_text(), 'previous')

    def test_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.csv'
            source.write_text('original', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(source, source), 1)
            self.assertEqual(source.read_text(), 'original')

    def _source(self, root):
        source = root / 'input.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(fixture())
        return source

    def test_cli_writes_complete_report_and_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            original = source.read_bytes()
            output = root / 'result.json'
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(source, output), 0)
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(saved['source']['sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(saved['temporal_split']['test_samples'], 12)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(set(saved['models']),
                             {'prevalence_baseline', 'logistic_regression', 'decision_tree', 'xgboost'})
            for name in saved['models']:
                self.assertIn(name, stdout.getvalue())
            self.assertFalse(list(root.glob('*.tmp')))
    def test_nonconvergence_preserves_previous_report(self):
        from sklearn.exceptions import ConvergenceWarning

        def nonconverged(*args, **kwargs):
            warnings.warn('controlled nonconvergence', ConvergenceWarning)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            output = root / 'result.json'
            output.write_text('previous', encoding='utf-8')
            # Failure injection at estimator fit: publication and error handling remain real.
            with patch('sklearn.linear_model.LogisticRegression.fit', side_effect=nonconverged):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(source, output), 1)
            self.assertEqual(output.read_text(), 'previous')

    def test_write_failure_preserves_previous_report_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            output = root / 'result.json'
            output.write_text('previous', encoding='utf-8')
            # Simulate an OS denial only at atomic publication, not model/report generation.
            with patch.object(Path, 'replace', side_effect=PermissionError('controlled denial')):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(source, output), 1)
            self.assertEqual(output.read_text(), 'previous')
            self.assertFalse(list(root.glob('*.tmp')))


class PublishedReportTests(unittest.TestCase):
    def test_saved_report_matches_real_input_and_acceptance_counts(self):
        root = Path(__file__).resolve().parent.parent
        report = json.loads((root / 'data/results/fda_model_comparison.json').read_text(encoding='utf-8'))
        source = root / 'data/processed/fda_modeling_2019_2023.csv'
        self.assertEqual(report['source']['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(report['input_samples'], 14149)
        split = report['temporal_split']
        self.assertEqual((split['train_samples'], split['train_violative'],
                          split['test_samples'], split['test_violative'], split['overlap_samples']),
                         (10572, 933, 3577, 376, 0))
        expectations = {
            'prevalence_baseline': (1.0, [[3201, 0], [376, 0]]),
            'logistic_regression': (2.497905, [[2245, 956], [105, 271]]),
            'decision_tree': (1.791895, [[1170, 2031], [31, 345]]),
            'xgboost': (2.563807, [[1919, 1282], [75, 301]]),
        }
        self.assertEqual(set(report['models']), set(expectations))
        for name, (lift, matrix) in expectations.items():
            metrics = report['models'][name]['metrics']
            self.assertAlmostEqual(metrics['top20_expected_lift'], lift, delta=1e-6)
            self.assertEqual(metrics['confusion_matrix'], matrix)
            self.assertEqual(metrics['top20_samples'], 716)
            if name != 'prevalence_baseline':
                audit = report['models'][name]['preprocessing']
                self.assertEqual(audit['fit_samples'], 10572)
                self.assertEqual(audit['test_unseen_counts'],
                                 dict(ProdCode=85, Country=8, Origin=0, CollMonth=0))


if __name__ == '__main__':
    unittest.main()
