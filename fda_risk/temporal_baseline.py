"""Evaluate a training-prevalence baseline on the held-out FDA year."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from fda_risk.modeling_dataset import (
    FEATURES, OUTPUT_FIELDS, ModelingDataError, load_samples, validate_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / 'data/processed/fda_modeling_2019_2023.csv'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'data/results/fda_temporal_baseline.json'
TRAIN_YEARS = ('2019', '2020', '2021', '2022')
TEST_YEAR = '2023'
TOP_FRACTION = 0.20


class BaselineDataError(ValueError):
    """Raised when the modeling CSV cannot support a temporal evaluation."""


def split_samples(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Validate the F05 schema and split by fiscal year without shuffling."""
    if len(set(fieldnames)) != len(fieldnames) or set(fieldnames) != set(OUTPUT_FIELDS):
        raise BaselineDataError('輸入須為 F05 的七個欄位，不可缺欄、重複或夾帶其他欄位')
    validate_features(FEATURES)
    seen: set[str] = set()
    years: set[str] = set()
    train, test = [], []
    for line, row in enumerate(rows, start=2):
        if set(row) != set(OUTPUT_FIELDS):
            raise BaselineDataError(f'第 {line} 列的 CSV 儲存格數量不符')
        values = {field: '' if row[field] is None else str(row[field]).strip()
                  for field in OUTPUT_FIELDS}
        year, sample_id = values['Year'], values['sample_id']
        if year not in (*TRAIN_YEARS, TEST_YEAR):
            raise BaselineDataError(f'第 {line} 列的年度無效：{year}')
        if not sample_id.startswith(f'{year}-') or not sample_id[len(year) + 1:]:
            raise BaselineDataError(f'第 {line} 列的 sample_id 與 Year 不一致')
        if sample_id in seen:
            raise BaselineDataError(f'sample_id 重複：{sample_id}')
        seen.add(sample_id)
        years.add(year)
        if values['violative'] not in {'0', '1'}:
            raise BaselineDataError(f'樣本 {sample_id} 的 violative 無效')
        for field in FEATURES:
            if not values[field] or values[field].casefold() == 'unknown':
                raise BaselineDataError(f'樣本 {sample_id} 的 {field} 缺漏')
        if values['Origin'] not in {'D', 'I', 'R'}:
            raise BaselineDataError(f'樣本 {sample_id} 的 Origin 無效')
        if values['CollMonth'] not in {f'{month:02d}' for month in range(1, 13)}:
            raise BaselineDataError(f'樣本 {sample_id} 的 CollMonth 無效')
        (test if year == TEST_YEAR else train).append(values)
    missing_years = sorted(set((*TRAIN_YEARS, TEST_YEAR)) - years)
    if missing_years:
        raise BaselineDataError(f'缺少分析年度：{", ".join(missing_years)}')
    for name, samples in (('訓練集', train), ('測試集', test)):
        if {sample['violative'] for sample in samples} != {'0', '1'}:
            raise BaselineDataError(f'{name} 必須同時含有 0 與 1，才能計算本版完整評估')
    if {row['sample_id'] for row in train} & {row['sample_id'] for row in test}:
        raise BaselineDataError('訓練集與測試集樣本重疊')
    return train, test


def build_baseline(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fit only on train, then evaluate constant scores and random-top-k expectations."""
    train, test = split_samples(fieldnames, rows)
    train_positive = sum(int(row['violative']) for row in train)
    test_positive = sum(int(row['violative']) for row in test)
    probability = train_positive / len(train)
    test_rate = test_positive / len(test)
    categories = {field: sorted({row[field] for row in train}) for field in FEATURES}
    # Test-only categories are counted for audit, never merged into fitted state.
    known = {field: set(categories[field]) for field in FEATURES}
    unseen = {field: sum(row[field] not in known[field] for row in test)
              for field in FEATURES}
    n_top = math.ceil(len(test) * TOP_FRACTION)
    # For constant scores, AP is test prevalence and pairwise ROC-AUC is 0.5.
    # Top-k uses random selection expectation, not arbitrary CSV tie ordering.
    metrics = {
        'average_precision': test_rate,
        'roc_auc': 0.5,
        'brier': sum((int(row['violative']) - probability) ** 2 for row in test) / len(test),
        'precision_at_0_5': test_rate if probability >= 0.5 else 0.0,
        'recall_at_0_5': 1.0 if probability >= 0.5 else 0.0,
        'top20_samples': n_top,
        'top20_expected_precision': test_rate,
        'top20_expected_recall': n_top / len(test),
        'top20_expected_lift': 1.0,
    }
    return {
        'schema_version': 1,
        'model': 'training_prevalence_baseline',
        'input_samples': len(rows),
        'features': list(FEATURES),
        'temporal_split': {
            'train_years': [int(year) for year in TRAIN_YEARS],
            'test_year': int(TEST_YEAR),
            'train_samples': len(train), 'train_violative': train_positive,
            'test_samples': len(test), 'test_violative': test_positive,
            'overlap_samples': 0,
        },
        'baseline_probability': probability,
        'train_categories': categories,
        'test_unseen_counts': unseen,
        'metrics': metrics,
        'top20_policy': 'All scores tie. Report expected random selection of ceil(0.20*n_test) samples.',
        'preprocessing': 'Category vocabulary uses train only; actual model encoder fitting is verified in F07.',
        'interpretation': 'FDA targeted monitoring samples only; not population-wide market violation prevalence or a food safety determination.',
    }


def save_report(report: Mapping[str, object], path: Path) -> None:
    """Replace the report only after a complete JSON write succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, suffix='.tmp', delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def format_summary(report: Mapping[str, object], path: Path) -> str:
    split, metrics = report['temporal_split'], report['metrics']
    lines = [
        'FDA農藥殘留專案 時間切分與基準模型',
        f"訓練集 FY2019–FY2022：{split['train_samples']:,} 個樣本；不利結果 {split['train_violative']:,}",
        f"測試集 FY2023：{split['test_samples']:,} 個樣本；不利結果 {split['test_violative']:,}",
        f"訓練／測試樣本重疊：{split['overlap_samples']}",
        f"基準固定預測分數（訓練比例）：{report['baseline_probability']:.6f}",
        f"PR-AUC（Average Precision）：{metrics['average_precision']:.6f}",
        f"ROC-AUC：{metrics['roc_auc']:.6f}",
        f"Brier score：{metrics['brier']:.6f}",
        f"0.5 門檻 Precision／Recall：{metrics['precision_at_0_5']:.6f}／{metrics['recall_at_0_5']:.6f}",
        f"Top 20% 樣本數（向上取整）：{metrics['top20_samples']:,}",
        f"Top 20% 隨機期望 Precision：{metrics['top20_expected_precision']:.6f}",
        f"Top 20% 隨機期望 Recall：{metrics['top20_expected_recall']:.6f}",
        f"Top 20% 隨機期望 Lift：{metrics['top20_expected_lift']:.6f}",
    ]
    for field in FEATURES:
        lines.append(f"{field}：訓練類別 {len(report['train_categories'][field]):,}；測試未見類別樣本 {report['test_unseen_counts'][field]:,}")
    lines.extend([
        '類別字典只使用訓練集；實際模型編碼器的擬合由 F07 驗證。',
        '同分基準沒有排序能力；Top 20% 是隨機抽取的期望，不是實際排名成效。',
        'FDA 目標式抽驗結果不代表整體市場違規率，也不能直接判定食品安全。',
        f'輸出位置：{path}',
    ])
    return '\n'.join(lines)


def main(input_path: Path | None = None, output_path: Path | None = None) -> int:
    source = DEFAULT_INPUT_PATH if input_path is None else input_path
    output = DEFAULT_OUTPUT_PATH if output_path is None else output_path
    try:
        if source.resolve() == output.resolve() or (
            source.exists() and output.exists() and source.samefile(output)
        ):
            raise BaselineDataError('輸出不可覆寫輸入資料')
        fields, rows = load_samples(source)
        report = build_baseline(fields, rows)
        report['source'] = {
            'path': str(source.resolve()),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        save_report(report, output)
        print(format_summary(report, output))
    except (BaselineDataError, ModelingDataError, OSError) as exc:
        print(f'錯誤：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
