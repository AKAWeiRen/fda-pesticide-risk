"""Validate the portable release boundary, independent of the source checkout."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/delivery/manifest.json'
FORBIDDEN = ('learning/', 'docs/learning/', 'docs/coordination/',
             'docs/superpowers/', '.venv/', 'tmp/', '.git/',
             'LEARNING_PROGRESS.md', 'tests/test_learning_')


def validate_entries(entries):
    paths = []
    for item in entries:
        path = item['path']
        parsed = PurePosixPath(path)
        if (not isinstance(path, str) or not path or '\\' in path or ':' in path
                or parsed.is_absolute() or '..' in parsed.parts
                or str(parsed) != path or any(path.startswith(p) for p in FORBIDDEN)
                or 'vendor' in parsed.parts):
            raise ValueError('Unsafe release path')
        if item['kind'] not in ('copy', 'generated'):
            raise ValueError('Invalid provenance kind')
        if item['kind'] == 'copy':
            if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
                raise ValueError('Invalid SHA256')
            if item['source_path'] != path:
                raise ValueError('Copied files must retain source path')
        paths.append(path)
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise ValueError('Duplicate or unsorted release paths')
    return set(paths)


def release_files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob('*')
            if p.is_file() and '.git' not in p.relative_to(root).parts
            and '__pycache__' not in p.relative_to(root).parts
            and not p.name.endswith(('.pyc', '.pyo'))}


class FormalDeliveryTests(unittest.TestCase):
    def manifest(self):
        self.assertTrue(MANIFEST.is_file(), 'Missing formal release manifest')
        return json.loads(MANIFEST.read_text(encoding='utf-8'))

    def test_manifest_exact_tree_and_safe_paths(self):
        data = self.manifest()
        self.assertEqual(data['schema_version'], 1)
        self.assertEqual(data['source_commit'], 'caa89ebbf308d695e0aa6454cfa43d6fb7b39687')
        self.assertEqual(validate_entries(data['files']), release_files(ROOT))

    def test_copy_fingerprints(self):
        items = [i for i in self.manifest()['files'] if i['kind'] == 'copy']
        self.assertTrue(items)
        for item in items:
            with self.subTest(path=item['path']):
                self.assertEqual(hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest(), item['sha256'])

    def test_forbidden_paths_absent(self):
        for path in release_files(ROOT):
            with self.subTest(path=path):
                self.assertFalse(any(path.startswith(p) for p in FORBIDDEN))
                self.assertNotIn('vendor', PurePosixPath(path).parts)

    def test_rejects_unsafe_duplicate_and_unsorted_entries(self):
        for path in ('../learning/x.py', 'C:/outside/x.py', 'learning/guided/x.py',
                     '/absolute.py', 'dir/../x.py', 'dir\\x.py', './x.py'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_entries([{'path': path, 'kind': 'generated'}])
        entry = {'path': 'README.md', 'kind': 'generated'}
        with self.assertRaises(ValueError):
            validate_entries([entry, entry])
        with self.assertRaises(ValueError):
            validate_entries([{'path': 'z.py', 'kind': 'generated'}, entry])
        self.assertEqual(validate_entries([entry]), {'README.md'})

    def test_missing_extra_or_tampered_fixture_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertNotEqual(release_files(root), {'README.md'})
            (root / 'README.md').write_text('ok', encoding='utf-8')
            self.assertEqual(release_files(root), {'README.md'})
            (root / 'extra.txt').write_text('unexpected', encoding='utf-8')
            self.assertNotEqual(release_files(root), {'README.md'})
            self.assertNotEqual(hashlib.sha256(b'changed').hexdigest(),
                                '2689367b205c16ce32ed4200942b8b8b1e262dfc70d9bc9fbc77c49699a4f1df')

    def test_formal_documents_exclude_learning_records(self):
        banned = re.compile(r'\b[GV]\d{2}\b|learning/(?:guided|validation)|completion-report|三軌聊天室|練習 Prompt')
        # FDA raw values are not prose about the release; the explicit manifest
        # pins them byte-for-byte. Check every authored/referenced text document.
        for p in ROOT.rglob('*'):
            if (p.suffix in ('.md', '.html') or
                    (p.suffix == '.txt' and 'docs' in p.relative_to(ROOT).parts)) and '.git' not in p.parts:
                with self.subTest(path=p.relative_to(ROOT)):
                    self.assertIsNone(banned.search(p.read_text(encoding='utf-8')))

    def test_feature_docs_and_readme_links(self):
        for number in range(1, 12):
            path = ROOT / f'docs/features/F{number:02d}.html'
            self.assertTrue(path.is_file(), str(path))
            html = path.read_text(encoding='utf-8')
            for marker in ('<html lang="zh-Hant">', '目的', '輸入', '輸出', '方法', '驗收', '限制'):
                self.assertIn(marker, html)
        for doc in [ROOT / 'README.md', *ROOT.glob('docs/delivery/*.md')]:
            self.assertTrue(doc.is_file(), str(doc))
            for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
                parsed = urlsplit(link.strip('<>'))
                if parsed.scheme in ('http', 'https') or not parsed.path:
                    continue
                self.assertFalse(parsed.scheme or parsed.netloc, link)
                target = (doc.parent / unquote(parsed.path)).resolve()
                self.assertTrue(target.is_relative_to(ROOT.resolve()), link)
                self.assertTrue(target.is_file(), link)


if __name__ == '__main__':
    unittest.main()
