"""F09 formal interface documentation contract."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class F09ContractTests(unittest.TestCase):
    def test_formal_document_preserves_source_and_interactions(self):
        html = (ROOT/'docs/features/F09.html').read_text(encoding='utf-8')
        for token in ('15,150', '1,336', '813', '114', '50', '100000',
                      '空排行', '唯讀', '資料流', '整體市場'):
            self.assertIn(token, html)


if __name__ == '__main__':
    unittest.main()
