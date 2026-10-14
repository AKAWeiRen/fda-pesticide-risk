"""Formal release data and entry-point contracts."""
from pathlib import Path
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
SAVED_HASHES = {
    "data/processed/fda_samples_2019_2023.csv": "5539c32ab77bce653bb8e34a7b310435b8559de09c349a8c17c7d92737b42fa9",
    "data/processed/fda_modeling_2019_2023.csv": "237208561f3451e2303b6262cb9bbf9d61a5b6c8be168e2ac8cde904b3d01050",
    "data/results/fda_temporal_baseline.json": "bb79f545252b4f3f1a2cf06943b63be4c5ca114efc1e8e55500646ec9b1fd8ec",
    "data/results/fda_model_comparison.json": "bc52fba8a8e85b0d6728a81c0fc6a32cd5205cb28638691dd1593457b3743c20",
    "data/results/fda_risk_ranking_2023.csv": "3e6e464db6e0d0bac0de0197b2bb473ae0f95ac8597b33269bcfb45bf61e8ab5",
    "data/results/fda_risk_ranking_report.json": "67f1d4ebbb95aee3b5e61dc902406cea808d92e0ce3bdef51a281f920695de72"
}


class DeliveryContractTests(unittest.TestCase):
    def test_saved_fingerprints_match_verified_bytes(self):
        for path, expected in SAVED_HASHES.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), expected)

    def test_reports_match_source_and_ranking_bytes(self):
        source_hash = SAVED_HASHES['data/processed/fda_modeling_2019_2023.csv']
        for name in ('fda_temporal_baseline.json', 'fda_model_comparison.json',
                     'fda_risk_ranking_report.json'):
            report = json.loads((ROOT/'data/results'/name).read_text(encoding='utf-8'))
            self.assertEqual(report['source']['sha256'], source_hash)
        self.assertEqual(report['ranking_csv']['sha256'],
                         SAVED_HASHES['data/results/fda_risk_ranking_2023.csv'])

    def test_release_has_executable_entry_and_dependencies(self):
        for path in ('fda_risk/streamlit_app.py', 'requirements-ui.txt',
                     'requirements-models.txt', '_fda_feasibility/feasibility-results.json',
                     'README.md', 'docs/delivery/local-delivery.md'):
            self.assertTrue((ROOT/path).is_file(), path)

    def test_readme_retains_usage_and_risk_boundaries(self):
        text = (ROOT/'README.md').read_text(encoding='utf-8')
        for expected in ('公開部署狀態：尚未公開部署', '127.0.0.1',
                         'python -m pip install -r requirements-ui.txt',
                         'python -m streamlit run fda_risk/streamlit_app.py',
                         'python -m unittest discover -s tests -v',
                         '目標式抽驗', '不代表整體市場違規率',
                         '不能直接判定食品安全', '未校準',
                         '不是最終部署模型', '不提供新樣本預測'):
            self.assertIn(expected, text)


if __name__ == '__main__':
    unittest.main()
