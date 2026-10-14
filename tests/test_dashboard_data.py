import csv
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from fda_risk.dashboard_data import (
    DEFAULT_INPUT_PATH, DashboardDataError, build_dashboard, load_dashboard,
)
from fda_risk import sample_overview, food_country_analysis


FIELDS = ('sample_id', 'Year', 'ProdName', 'Country', 'LabClass')


def fixture():
    return [dict(zip(FIELDS, (f'{year}-1', str(year), 'Food A', 'Country A',
                            '3' if year == 2019 else '1')))
            for year in range(2019, 2024)]


class DashboardDataTests(unittest.TestCase):
    def test_real_data_matches_formal_functions(self):
        before = hashlib.sha256(DEFAULT_INPUT_PATH.read_bytes()).hexdigest()
        fields, rows = sample_overview.load_samples(DEFAULT_INPUT_PATH)
        result = load_dashboard()
        self.assertEqual(result['overview'], sample_overview.build_overview(fields, rows))
        self.assertEqual(result['analysis'], food_country_analysis.build_analysis(fields, rows))
        self.assertEqual(result['overview']['sample_count'], 15150)
        self.assertEqual(result['labclass3_count'], 1336)
        self.assertEqual(result['analysis']['products']['group_count'], 813)
        self.assertEqual(result['analysis']['countries']['group_count'], 114)
        self.assertEqual([r['samples'] for r in result['overview']['yearly']],
                         [4692, 2180, 1447, 3030, 3801])
        self.assertEqual([f"{r['rate']:.2%}" for r in result['overview']['yearly']],
                         ['7.72%', '9.95%', '8.64%', '8.25%', '10.05%'])
        products, countries = result['analysis']['products'], result['analysis']['countries']
        self.assertEqual(products['volume_top'][0]['name'],
                         'Pepper, Hot, N.E.C. (Fruit Used as Vegetable)')
        self.assertEqual(products['volume_top'][0]['samples'], 273)
        self.assertEqual(countries['volume_top'][0]['name'], 'United States')
        self.assertEqual(countries['volume_top'][0]['samples'], 4180)
        self.assertEqual(products['rate_top'][0]['name'], 'Rice, Basmati, Processed (Packaged)')
        self.assertEqual(f"{products['rate_top'][0]['rate']:.2%}", '37.69%')
        self.assertEqual(countries['rate_top'][0]['name'], 'Pakistan')
        self.assertEqual(f"{countries['rate_top'][0]['rate']:.2%}", '35.67%')
        self.assertEqual(hashlib.sha256(DEFAULT_INPUT_PATH.read_bytes()).hexdigest(), before)

    def test_threshold_only_changes_rate_top(self):
        first = build_dashboard(FIELDS, fixture(), min_rate_samples=1)
        second = build_dashboard(FIELDS, fixture(), min_rate_samples=50)
        self.assertEqual(first['overview'], second['overview'])
        self.assertEqual(first['labclass3_count'], 1)
        for key in ('products', 'countries'):
            self.assertEqual(first['analysis'][key]['volume_top'], second['analysis'][key]['volume_top'])
            self.assertEqual(first['analysis'][key]['group_count'], 1)
            self.assertEqual(first['analysis'][key]['rate_top'][0]['samples'], 5)
            self.assertEqual(second['analysis'][key]['rate_top'], [])

    def test_empty_rate_top_is_valid(self):
        result = load_dashboard(min_rate_samples=100000)
        for key in ('products', 'countries'):
            self.assertEqual(result['analysis'][key]['rate_top'], [])
            self.assertEqual(len(result['analysis'][key]['volume_top']), 10)

    def test_invalid_inputs_raise_dashboard_error(self):
        cases = [([], FIELDS, '沒有任何樣本'), (fixture(), FIELDS[1:], '缺少必要欄位')]
        for field, value, reason in (
            ('sample_id', '', 'sample_id'), ('Year', '2018', '年度'),
            ('LabClass', '0', 'LabClass'), ('ProdName', '', 'ProdName'),
            ('Country', '', 'Country'),
        ):
            rows = fixture()
            rows[0][field] = value
            cases.append((rows, FIELDS, reason))
        cases.extend([(fixture() + [fixture()[0]], FIELDS, '重複'),
                      (fixture()[:-1], FIELDS, '缺少年度')])
        for rows, fields, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(DashboardDataError, reason):
                build_dashboard(fields, rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            with self.assertRaisesRegex(DashboardDataError, '找不到'):
                load_dashboard(path)
            path.write_bytes(b'\xff')
            with self.assertRaisesRegex(DashboardDataError, '無法解析'):
                load_dashboard(path)

    def test_invalid_threshold_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(DashboardDataError, '正整數'):
                build_dashboard(FIELDS, fixture(), min_rate_samples=value)

    def test_default_path_independent_of_cwd(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(load_dashboard()['overview']['sample_count'], 15150)
            finally:
                os.chdir(old)

    def test_rank_order_preserved(self):
        # Four groups: B/C exact tie; A/D distinct ratios display identically.
        rows = []
        for name, count, positives in [('A', 10001, 5000), ('D', 10000, 5000),
                                       ('C', 2, 1), ('B', 2, 1)]:
            for index in range(count):
                rows.append(dict(zip(FIELDS, (f'{name}-{index}', str(2019 + index % 5),
                                              name, name, '3' if index < positives else '1'))))
        result = build_dashboard(FIELDS, rows, min_rate_samples=1)
        ranking = result['analysis']['products']['rate_top']
        self.assertEqual([r['name'] for r in ranking], ['D', 'B', 'C', 'A'])
        self.assertEqual(f"{ranking[0]['rate']:.2%}", f"{ranking[-1]['rate']:.2%}")
        self.assertEqual(result['analysis'], food_country_analysis.build_analysis(
            FIELDS, rows, min_rate_samples=1))


if __name__ == '__main__':
    unittest.main()
