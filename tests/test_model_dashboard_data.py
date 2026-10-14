import copy
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fda_risk import model_dashboard_data as data

ROOT = Path(__file__).resolve().parents[1]
PATHS = (ROOT / 'data/results/fda_model_comparison.json',
         ROOT / 'data/results/fda_risk_ranking_report.json',
         ROOT / 'data/results/fda_risk_ranking_2023.csv',
         ROOT / 'data/processed/fda_modeling_2019_2023.csv')


def small_ranked(scores=(.9, .5, .5, .5, .1, .1)):
    rows = []
    for index, score in enumerate(scores, 1):
        logit = math.log(score / (1-score)) if score not in (0, 1) else (1000 if score else -1000)
        rows.append(dict(display_position=index, score_rank=1 if index == 1 else 2 if index <= 4 else 5,
                         sample_id=f'2023-{index}', Year='2023', ProdCode='A', Country='C',
                         Origin='I', CollMonth='01', risk_score=score,
                         top20_weight=1 if index == 1 else 1/3 if index <= 4 else 0,
                         ProdCode_contribution=logit, Country_contribution=0,
                         Origin_contribution=0, CollMonth_contribution=0, intercept=0,
                         logit=logit, reconstructed_score=score, unseen_features='', violative='1' if index <= 3 else '0'))
    return rows


class ModelDashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good = data.load_model_dashboard()

    def test_real_data_contract(self):
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in PATHS]
        result = data.load_model_dashboard()
        c, r, rows = result['comparison'], result['ranking_report'], result['ranked_rows']
        self.assertEqual(c['input_samples'], 14149)
        self.assertEqual(c['temporal_split']['train_samples'], 10572)
        self.assertEqual(c['temporal_split']['train_violative'], 933)
        self.assertEqual(len(rows), 3577)
        self.assertEqual(sum(int(row['violative']) for row in rows), 376)
        self.assertEqual(c['temporal_split']['overlap_samples'], 0)
        self.assertEqual(r['metrics']['confusion_matrix'], [[2245,956],[105,271]])
        self.assertAlmostEqual(math.fsum(row['top20_weight'] for row in rows), 716)
        self.assertAlmostEqual(math.fsum(row['top20_weight'] * int(row['violative']) for row in rows), 188)
        self.assertEqual(r['metrics']['top20_expected_recall'], .5)
        self.assertAlmostEqual(r['metrics']['top20_expected_lift'], 2.4979050279329607)
        for row in rows:
            self.assertAlmostEqual(math.fsum([row['intercept']] + [row[f'{f}_contribution'] for f in data.FEATURES]), row['logit'], delta=1e-10)
            self.assertAlmostEqual(row['risk_score'], row['reconstructed_score'], delta=1e-12)
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in PATHS], before)

    def test_filters_preserve_global_fields(self):
        original = self.good['ranked_rows']
        snapshot = copy.deepcopy(original)
        for kwargs in ({'country':'Vietnam'}, {'product_code':original[0]['ProdCode']},
                       {'top20_only':True}, {'country':'Vietnam','product_code':original[0]['ProdCode'],'top20_only':True}):
            actual = data.filter_ranked_rows(original, **kwargs)
            expected = [r for r in original if (not kwargs.get('country') or r['Country']==kwargs['country'])
                        and (not kwargs.get('product_code') or r['ProdCode']==kwargs['product_code'])
                        and (not kwargs.get('top20_only') or r['top20_weight']>0)]
            self.assertEqual(actual, expected)
        self.assertEqual(data.filter_ranked_rows(original, country='not-present'), [])
        self.assertEqual(original, snapshot)

    def test_fractional_boundary_policy(self):
        rows = small_ranked()
        actual = data.validate_ranking(data.RANKING_FIELDS, rows, intercept=0)
        self.assertEqual([r['score_rank'] for r in actual], [1,2,2,2,5,5])
        self.assertEqual([r['top20_weight'] for r in actual], [1,1/3,1/3,1/3,0,0])
        self.assertEqual(len(data.filter_ranked_rows(actual, top20_only=True)), 4)
        for field, value in [('score_rank',3), ('top20_weight',1), ('display_position',99)]:
            wrong = copy.deepcopy(rows)
            wrong[2][field] = value
            with self.subTest(field=field), self.assertRaises(data.ModelDashboardError):
                data.validate_ranking(data.RANKING_FIELDS, wrong, intercept=0)
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaises(data.ModelDashboardError):
            data.validate_ranking(data.RANKING_FIELDS, rows, intercept=0)

    def test_extreme_logit_and_numeric_guards(self):
        rows = small_ranked((1,.5,.5,.5,0,0))
        self.assertEqual(data.validate_ranking(data.RANKING_FIELDS, rows, intercept=0)[0]['risk_score'], 1)
        for field, value in [('risk_score',float('nan')), ('logit',float('inf')),
                             ('risk_score',1.1), ('top20_weight',-.1),
                             ('score_rank',True), ('display_position','1.0'),
                             ('ProdCode_contribution',.123), ('reconstructed_score',.8),
                             ('intercept',1), ('violative','2'), ('Year','2022')]:
            wrong = copy.deepcopy(rows)
            wrong[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(data.ModelDashboardError):
                data.validate_ranking(data.RANKING_FIELDS, wrong, intercept=0)

    def _with_files(self, mutate):
        # Test-only copies, never change any real result or source.
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/p.name for p in PATHS]
            for source, dest in zip(PATHS, paths):
                dest.write_bytes(source.read_bytes())
            mutate(paths)
            return data.load_model_dashboard(*paths)

    @staticmethod
    def _json_change(paths, index, mutate):
        report = json.loads(paths[index].read_text(encoding='utf-8'))
        mutate(report)
        paths[index].write_text(json.dumps(report), encoding='utf-8')

    def test_input_and_report_corruption(self):
        mutations = [lambda p:p[0].unlink(), lambda p:p[0].write_bytes(b'\xff'),
                     lambda p:p[0].write_text('{',encoding='utf-8'),
                     lambda p:p[0].write_text('[]',encoding='utf-8'),
                     lambda p:p[0].write_text('{"x":1,"x":2}',encoding='utf-8'),
                     lambda p:p[0].write_text('{"x":NaN}',encoding='utf-8'),
                     lambda p:p[2].write_bytes(p[2].read_bytes()+b'\n')]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(data.ModelDashboardError):
                self._with_files(mutate)
        for index, mutate in [
            (0,lambda r:r.update(schema_version=True)), (0,lambda r:r.pop('features')),
            (0,lambda r:r.update(features=['VioType'])),
            (0,lambda r:r['models'].pop('decision_tree')),
            (1,lambda r:r['parameters'].update(C=99)),
            (1,lambda r:r['temporal_split'].update(train_samples=1)),
            (0,lambda r:r['source'].update(sha256='bad')),
            (1,lambda r:r['ranking_csv'].update(fields=['bad']))]:
            with self.subTest(index=index, mutate=mutate), self.assertRaises(data.ModelDashboardError):
                self._with_files(lambda p:self._json_change(p,index,mutate))

    def test_same_counts_different_rows_rejected(self):
        for field, value in [('sample_id','2023-missing'), ('Country','Not original'),
                             ('violative','1'), ('unseen_features','Country')]:
            def mutate(paths):
                reader = csv.DictReader(io.StringIO(paths[2].read_text(encoding='utf-8-sig')))
                names, rows = list(reader.fieldnames), list(reader)
                rows[0][field] = value
                out = io.StringIO()
                writer=csv.DictWriter(out,fieldnames=names)
                writer.writeheader(); writer.writerows(rows)
                paths[2].write_text(out.getvalue(),encoding='utf-8-sig',newline='')
                self._json_change(paths,1,lambda r:r['ranking_csv'].update(sha256=hashlib.sha256(paths[2].read_bytes()).hexdigest()))
            with self.subTest(field=field), self.assertRaises(data.ModelDashboardError):
                self._with_files(mutate)
        rows = small_ranked()
        for names in (data.RANKING_FIELDS[:-1], (*data.RANKING_FIELDS,'extra'),
                      (*data.RANKING_FIELDS[:-1],data.RANKING_FIELDS[0])):
            with self.assertRaises(data.ModelDashboardError):
                data.validate_ranking(names, rows, intercept=0)
        for wrong in (dict(rows[0], extra='value'), {k:v for k,v in rows[0].items() if k!='Origin'}):
            with self.assertRaises(data.ModelDashboardError):
                data.validate_ranking(data.RANKING_FIELDS,[wrong,*rows[1:]],intercept=0)

    def test_metric_consistency(self):
        for mutate in [lambda r:r['models']['decision_tree']['metrics'].update(roc_auc=float('inf')),
                       lambda r:r['models']['xgboost']['metrics'].update(average_precision=2),
                       lambda r:r['models']['decision_tree']['metrics'].update(confusion_matrix=[[True,2031],[31,345]]),
                       lambda r:r['models']['decision_tree']['metrics'].update(top20_expected_lift=-1),
                       lambda r:r['models']['decision_tree']['metrics'].update(top20_samples=715),
                       lambda r:r['models']['decision_tree']['metrics'].update(top20_expected_recall=.1),
                       lambda r:r.update(threshold=.7),
                       lambda r:r.update(confusion_matrix_order='bad'),
                       lambda r:r.update(top20_policy='CSV-order'),
                       lambda r:r['models']['logistic_regression']['metrics'].update(brier=.1)]:
            with self.subTest(mutate=mutate), self.assertRaises(data.ModelDashboardError):
                self._with_files(lambda p:self._json_change(p,0,mutate))

    def test_saved_results_need_no_training_packages(self):
        code = """
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'numpy','scipy','sklearn','xgboost'}:
            raise ImportError('Training dependency forbidden: '+fullname)
sys.meta_path.insert(0,Block())
from fda_risk.model_dashboard_data import load_model_dashboard
print(len(load_model_dashboard()['ranked_rows']))
"""
        with tempfile.TemporaryDirectory() as directory:
            env=os.environ.copy(); env['PYTHONPATH']=str(ROOT)
            result=subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=directory,
                                  env=env,capture_output=True,text=True,encoding='utf-8',timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'3577')
        result=self._with_files(lambda p:self._json_change(p,1,lambda r:r['ranking_csv'].update(path='C:/not-a-real-file.csv')))
        self.assertEqual(len(result['ranked_rows']),3577)

    def test_oversized_integer_reports_friendly_error(self):
        with self.assertRaisesRegex(data.ModelDashboardError,'整數|解析'):
            self._with_files(lambda p:p[0].write_text(
                '{"schema_version":'+('9'*5000)+'}',encoding='utf-8'))

    def test_oversized_csv_integer_reports_friendly_error(self):
        rows=small_ranked()
        rows[0]['display_position']='9'*5000
        with self.assertRaisesRegex(data.ModelDashboardError,'整數|轉換'):
            data.validate_ranking(data.RANKING_FIELDS,rows,intercept=0)


if __name__=='__main__':
    unittest.main()
