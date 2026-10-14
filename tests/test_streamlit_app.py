import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from fda_risk import dashboard_data

APP = Path(__file__).resolve().parents[1] / 'fda_risk' / 'streamlit_app.py'


class StreamlitDashboardTests(unittest.TestCase):
    def app(self):
        self.assertTrue(APP.exists(), 'F09 app must exist')
        return AppTest.from_file(str(APP), default_timeout=20).run()

    def test_initial_render(self):
        at = self.app()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([m.value for m in at.metric], ['15,150', '1,336', '813', '114'])
        self.assertEqual(at.number_input(key='min_rate_samples').value, 50)
        self.assertEqual(len(at.dataframe), 6)
        self.assertEqual(len(at.dataframe[0].value), 5)
        self.assertEqual(len(at.get('vega_lite_chart')), 5)
        self.assertTrue(any('整體市場' in w.value for w in at.warning))

    def test_threshold_rerun(self):
        at = self.app()
        initial_years = at.dataframe[0].value.copy()
        at.number_input(key='min_rate_samples').set_value(100000).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([m.value for m in at.metric], ['15,150', '1,336', '813', '114'])
        self.assertTrue(at.dataframe[0].value.equals(initial_years))
        self.assertEqual(sum('沒有符合條件' in i.value for i in at.info), 2)
        self.assertEqual(len(at.dataframe), 4)
        self.assertEqual(len(at.get('vega_lite_chart')), 3)
        at.number_input(key='min_rate_samples').set_value(50).run()
        self.assertEqual(len(at.dataframe), 6)

    def test_error_removes_previous_results(self):
        at = self.app()
        with patch('fda_risk.dashboard_data.load_dashboard',
                   side_effect=dashboard_data.DashboardDataError('缺少必要欄位：Country')):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(len(at.dataframe), 0)
        self.assertEqual(len(at.get('vega_lite_chart')), 0)
        self.assertIn('Country', at.error[0].value)

    def test_rerun_reloads_source(self):
        source = dashboard_data.DEFAULT_INPUT_PATH
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        calls = []
        real = dashboard_data.load_dashboard
        def read_again(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append(result)
            return result
        with patch('fda_risk.dashboard_data.load_dashboard', side_effect=read_again):
            at = self.app()
            at.number_input(key='min_rate_samples').set_value(100000).run()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['analysis']['min_rate_samples'], 50)
        self.assertEqual(calls[1]['analysis']['products']['rate_top'], [])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_non_root_cwd(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                at = self.app()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(at.metric[0].value, '15,150')
            finally:
                os.chdir(old)

    def test_tables_match_data_contract(self):
        at = self.app()
        tables = [d.value for d in at.dataframe]
        self.assertEqual(tables[0]['樣本數'].tolist(), [4692, 2180, 1447, 3030, 3801])
        self.assertEqual(tables[0]['資料內 LabClass 3 比例'].tolist(),
                         ['7.72%', '9.95%', '8.64%', '8.25%', '10.05%'])
        self.assertEqual(tables[1].iloc[0]['名稱'],
                         'Pepper, Hot, N.E.C. (Fruit Used as Vegetable)')
        self.assertEqual(tables[1].iloc[0]['樣本數'], 273)
        self.assertEqual(tables[2].iloc[0]['資料內 LabClass 3 比例'], '37.69%')
        self.assertEqual(tables[3].iloc[0]['名稱'], 'United States')
        self.assertEqual(tables[3].iloc[0]['樣本數'], 4180)
        self.assertEqual(tables[4].iloc[0]['名稱'], 'Pakistan')
        self.assertEqual(tables[4].iloc[0]['資料內 LabClass 3 比例'], '35.67%')
        self.assertEqual(len(tables[5]), 14)
        self.assertEqual(tables[5]['空白'].sum(), 0)
        self.assertEqual(tables[5]['Unknown'].sum(), 0)

    def test_fresh_process_non_root_startup(self):
        # Parent imports must not accidentally make package discovery pass.
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            # A clean delivery has no vendor copy; retain the caller's dependency path.
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (
                str(APP.parents[1] / '_fda_feasibility' / 'vendor'),
                env.get('PYTHONPATH', ''),
            )))
            code = (
                "import pathlib,tempfile,sys,json; tempfile.tempdir=str(pathlib.Path.cwd()); "
                "from streamlit.testing.v1 import AppTest; "
                "at=AppTest.from_file(sys.argv[1],default_timeout=20).run(); "
                "print(json.dumps({'errors':[e.message for e in at.exception],"
                "'metrics':[m.value for m in at.metric]}))"
            )
            result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(APP)],
                                    cwd=directory, env=env, capture_output=True,
                                    text=True, encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(actual['errors'], [])
        self.assertEqual(actual['metrics'], ['15,150', '1,336', '813', '114'])


if __name__ == '__main__':
    unittest.main()
