"""F10 formal interface documentation contract."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class F10ContractTests(unittest.TestCase):
    def test_formal_document_preserves_scope_and_explanation(self):
        html = (ROOT/'docs/features/F10.html').read_text(encoding='utf-8')
        for token in ('3,577', '376', '716', '188', '1e-10', '1e-12',
                      '唯讀', 'AND', '全體', 'logit', '歷史答案',
                      '未見', 'Brier', '未校準'):
            self.assertIn(token, html)


if __name__ == '__main__':
    unittest.main()
