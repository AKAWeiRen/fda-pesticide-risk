"""Prepare an explicit, audited feature contract for FDA risk modeling."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / 'data/processed/fda_samples_2019_2023.csv'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'data/processed/fda_modeling_2019_2023.csv'
FEATURES = ('ProdCode', 'Country', 'Origin', 'CollMonth')
METADATA_FIELDS = ('sample_id', 'Year')
OUTPUT_FIELDS = (*METADATA_FIELDS, *FEATURES, 'violative')
REQUIRED_COLUMNS = (*OUTPUT_FIELDS, 'SplNo', 'PAC', 'CollDate', 'LabClass')
EXPECTED_YEARS = tuple(str(year) for year in range(2019, 2024))
MODELING_PAC = '04004A'


class ModelingDataError(ValueError):
    """Raised when a dataset cannot satisfy the reviewed modeling contract."""


def _clean(value: object) -> str:
    return '' if value is None else str(value).strip()


def validate_features(features: Sequence[str]) -> None:
    """Only reviewed pre-analysis fields may enter the model's X matrix."""
    if not features or len(set(features)) != len(features):
        raise ModelingDataError('模型特徵不可空白或重複')
    unapproved = sorted(set(features) - set(FEATURES))
    if unapproved:
        raise ModelingDataError(f'禁止或尚未核准的模型特徵：{", ".join(unapproved)}')


def load_samples(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ModelingDataError(f'無法讀取樣本 CSV：{path}；{exc}') from exc
    return fields, rows


def build_modeling_samples(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, str | int]], dict[str, object]]:
    """Validate all source samples, filter PAC, and project a fixed schema."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
    if missing:
        raise ModelingDataError(f'缺少必要欄位：{", ".join(missing)}')
    if len(set(fieldnames)) != len(fieldnames):
        raise ModelingDataError('CSV 欄位名稱重複')
    if not rows:
        raise ModelingDataError('樣本 CSV 沒有任何樣本')
    validate_features(FEATURES)
    seen: set[str] = set()
    modeling: list[dict[str, str | int]] = []
    for line, row in enumerate(rows, start=2):
        if None in row:
            raise ModelingDataError(f'第 {line} 列有多餘的 CSV 儲存格')
        values = {field: _clean(row.get(field)) for field in REQUIRED_COLUMNS}
        sample_id, year, number = values['sample_id'], values['Year'], values['SplNo']
        if year not in EXPECTED_YEARS or not number or sample_id != f'{year}-{number}':
            raise ModelingDataError(f'第 {line} 列的 sample_id、Year 或 SplNo 無效')
        if sample_id in seen:
            raise ModelingDataError(f'sample_id 重複：{sample_id}')
        seen.add(sample_id)
        if values['LabClass'] not in {'1', '2', '3', '4', '5'}:
            raise ModelingDataError(f'樣本 {sample_id} 的 LabClass 無效')
        label = values['violative']
        if label not in {'0', '1'} or int(label) != int(values['LabClass'] == '3'):
            raise ModelingDataError(f'樣本 {sample_id} 的 violative 與 LabClass 不一致')
        date = values['CollDate']
        try:
            if len(date) != 8 or not date.isascii() or not date.isdigit():
                raise ValueError('採樣日期必須為 YYYYMMDD')
            datetime.strptime(date, '%Y%m%d')
        except ValueError as exc:
            raise ModelingDataError(f'樣本 {sample_id} 的 CollDate 無效：{date}') from exc
        if values['CollMonth'] != date[4:6]:
            raise ModelingDataError(f'樣本 {sample_id} 的 CollMonth 與 CollDate 不一致')
        for field in FEATURES:
            if not values[field] or values[field].casefold() == 'unknown':
                raise ModelingDataError(f'樣本 {sample_id} 的 {field} 缺漏')
        if values['Origin'] not in {'D', 'I', 'R'}:
            raise ModelingDataError(f'樣本 {sample_id} 的 Origin 無效')
        if not values['PAC'] or values['PAC'].casefold() == 'unknown':
            raise ModelingDataError(f'樣本 {sample_id} 的 PAC 缺漏')
        if values['PAC'] == MODELING_PAC:
            sample: dict[str, str | int] = {field: values[field] for field in OUTPUT_FIELDS}
            sample['violative'] = int(label)
            modeling.append(sample)
    if not modeling:
        raise ModelingDataError('PAC 04004A 範圍內沒有建模樣本')
    yearly = Counter(str(sample['Year']) for sample in modeling)
    positive = Counter(str(sample['Year']) for sample in modeling if sample['violative'] == 1)
    audit: dict[str, object] = {
        'input_samples': len(rows),
        'modeling_samples': len(modeling),
        'excluded_samples': len(rows) - len(modeling),
        'violative_samples': sum(int(sample['violative']) for sample in modeling),
        'yearly_samples': dict(sorted(yearly.items())),
        'yearly_violative': {year: positive[year] for year in sorted(yearly)},
        'feature_missing_counts': {field: 0 for field in FEATURES},
        'leakage_feature_count': 0,
    }
    return modeling, audit


def write_samples(samples: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(samples)


def format_audit(audit: Mapping[str, object], path: Path) -> str:
    lines = [
        'FDA農藥殘留專案 建模資料集與資料洩漏檢查',
        f"輸入樣本數：{audit['input_samples']:,}",
        f"建模樣本數（PAC 04004A）：{audit['modeling_samples']:,}",
        f"排除其他 PAC 樣本數：{audit['excluded_samples']:,}",
        f"不利結果樣本數：{audit['violative_samples']:,}",
        f"洩漏特徵數：{audit['leakage_feature_count']}",
        f"模型特徵：{', '.join(FEATURES)}",
        '識別／時間欄位：sample_id、Year（不作為模型特徵）',
        '答案欄位：violative（不作為模型特徵）',
    ]
    for year, count in audit['yearly_samples'].items():
        lines.append(f"FY{year}：樣本 {count:,}；不利結果 {audit['yearly_violative'][year]:,}")
    for field, count in audit['feature_missing_counts'].items():
        lines.append(f'{field} 缺漏數：{count}')
    lines.extend([
        f'輸出位置：{path}',
        '限制：本檢查確認核准特徵契約，不能證明所有潛在洩漏均不存在。',
        'F06 驗證時間切分與訓練類別字典；實際模型編碼器只在訓練集擬合，由 F07 接續驗證。',
        'FDA 目標式抽驗結果不代表整體市場違規率，也不能直接判定食品安全。',
    ])
    return '\n'.join(lines)


def main(input_path: Path | None = None, output_path: Path | None = None) -> int:
    source = DEFAULT_INPUT_PATH if input_path is None else input_path
    output = DEFAULT_OUTPUT_PATH if output_path is None else output_path
    try:
        if source.resolve() == output.resolve():
            raise ModelingDataError('輸出路徑不可覆寫輸入 CSV')
        fields, rows = load_samples(source)
        samples, audit = build_modeling_samples(fields, rows)
        write_samples(samples, output)
        print(format_audit(audit, output))
    except (ModelingDataError, OSError) as exc:
        print(f'錯誤：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
