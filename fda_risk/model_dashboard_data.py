"""Validate saved F07/F08 results without loading any training dependencies."""

import csv
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fda_risk.modeling_dataset import FEATURES
from fda_risk.temporal_baseline import BaselineDataError, split_samples

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_PATH = ROOT / 'data/results/fda_model_comparison.json'
DEFAULT_RANKING_REPORT_PATH = ROOT / 'data/results/fda_risk_ranking_report.json'
DEFAULT_RANKING_CSV_PATH = ROOT / 'data/results/fda_risk_ranking_2023.csv'
DEFAULT_SOURCE_PATH = ROOT / 'data/processed/fda_modeling_2019_2023.csv'
MODEL_NAMES = ('prevalence_baseline', 'logistic_regression', 'decision_tree', 'xgboost')
CONTRIBUTIONS = tuple(f'{f}_contribution' for f in FEATURES)
RANKING_FIELDS = ('display_position', 'score_rank', 'sample_id', 'Year', *FEATURES,
                  'risk_score', 'top20_weight', *CONTRIBUTIONS, 'intercept', 'logit',
                  'reconstructed_score', 'unseen_features', 'violative')
TOP_POLICY = (
    'k=ceil(0.20*n_test). Include scores above the kth score; remaining slots '
    'are uniformly random within the exact-score boundary tie. Report expected '
    'positives, precision, recall and lift; no CSV-order tie breaking.'
)
LR_PARAMETERS = dict(class_weight='balanced', C=.5, max_iter=2000,
                     solver='liblinear', random_state=42)


class ModelDashboardError(ValueError):
    """A saved result cannot support a trustworthy complete display."""


def _need(obj: Mapping, key: str) -> Any:
    if key not in obj:
        raise ModelDashboardError(f'缺少必要欄位：{key}')
    return obj[key]


def _object(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ModelDashboardError(f'{name} 須為物件')
    return value


def _number(value: Any, name: str, *, csv_value: bool = False) -> float:
    if type(value) not in (int, float) and not (csv_value and isinstance(value, str)):
        raise ModelDashboardError(f'{name} 須為有限數值，不可為布林值')
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ModelDashboardError(f'{name} 須為有限數值') from exc
    if not math.isfinite(number):
        raise ModelDashboardError(f'{name} 須為有限數值')
    return number


def _integer(value: Any, name: str, *, csv_value: bool = False) -> int:
    if type(value) is int and value >= 0:
        return value
    if csv_value and isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
        try:
            return int(value)
        except ValueError as exc:
            raise ModelDashboardError(f'{name} 整數超出可解析範圍') from exc
    raise ModelDashboardError(f'{name} 須為非負整數，不可為布林值')


def _close(actual: float, expected: float, name: str, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise ModelDashboardError(f'{name} 不一致：{actual}／{expected}')


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelDashboardError(f'JSON 欄位重複：{key}')
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ModelDashboardError(f'JSON 非有限數值：{value}')


def _json(raw: bytes, name: str) -> Mapping:
    try:
        return _object(json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_pairs,
                                  parse_constant=_invalid_constant), name)
    except (UnicodeError, ValueError) as exc:
        raise ModelDashboardError(f'{name} 無法解析：{exc}') from exc


def _csv(raw: bytes, name: str) -> tuple[tuple[str, ...], list[dict]]:
    try:
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
        fields, rows = tuple(reader.fieldnames or ()), list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise ModelDashboardError(f'{name} CSV 無法解析：{exc}') from exc
    if len(fields) != len(set(fields)):
        raise ModelDashboardError(f'{name} CSV 欄名重複')
    if any(set(row) != set(fields) or any(v is None for v in row.values()) for row in rows):
        raise ModelDashboardError(f'{name} CSV 儲存格數量不符')
    return fields, rows


def validate_ranking(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]], *, intercept: float,
) -> list[dict[str, Any]]:
    """Normalize a complete ranking; verify ties and additive explanations."""
    if tuple(fieldnames) != RANKING_FIELDS or not rows:
        raise ModelDashboardError('排序 CSV 須為固定 19 欄、順序正確且非空')
    reference_intercept = _number(intercept, '報告 intercept')
    normalized, seen = [], set()
    for source in rows:
        if set(source) != set(RANKING_FIELDS):
            raise ModelDashboardError('排序 CSV 儲存格數量不符')
        row = dict(source)
        for field in ('sample_id', 'Year', *FEATURES, 'violative', 'unseen_features'):
            if not isinstance(row[field], str):
                raise ModelDashboardError(f'排序 {field} 須為文字')
        key = row['sample_id']
        if not key.startswith('2023-') or not key[5:] or key != key.strip() or key in seen:
            raise ModelDashboardError(f'排序 sample_id 空白、年度不符或重複：{key}')
        seen.add(key)
        if row['Year'] != '2023' or row['violative'] not in {'0', '1'}:
            raise ModelDashboardError(f'{key} 年度或答案無效')
        if any(not row[f].strip() or row[f].strip() != row[f] for f in FEATURES):
            raise ModelDashboardError(f'{key} 特徵空白或未正規化')
        unseen = row['unseen_features'].split('|') if row['unseen_features'] else []
        if len(unseen) != len(set(unseen)) or any(f not in FEATURES for f in unseen):
            raise ModelDashboardError(f'{key} 未見類別標記無效')
        for field in ('display_position', 'score_rank'):
            row[field] = _integer(row[field], f'{key} {field}', csv_value=True)
        for field in ('risk_score', 'top20_weight', *CONTRIBUTIONS,
                      'intercept', 'logit', 'reconstructed_score'):
            row[field] = _number(row[field], f'{key} {field}', csv_value=True)
        if any(not 0 <= row[f] <= 1 for f in ('risk_score','top20_weight','reconstructed_score')):
            raise ModelDashboardError(f'{key} 分數或權重須在 0–1')
        _close(row['intercept'], reference_intercept, f'{key} intercept')
        try:
            summed = math.fsum([row['intercept']] + [row[f] for f in CONTRIBUTIONS])
        except (OverflowError, ValueError) as exc:
            raise ModelDashboardError(f'{key} 貢獻加總超出有限範圍') from exc
        _close(summed, row['logit'], f'{key} logit 還原', 1e-10)
        exponent = math.exp(-abs(row['logit']))
        sigmoid = 1 / (1 + exponent) if row['logit'] >= 0 else exponent / (1 + exponent)
        _close(sigmoid, row['risk_score'], f'{key} 分數還原')
        _close(sigmoid, row['reconstructed_score'], f'{key} reconstructed_score')
        normalized.append(row)
    keys = [(-r['risk_score'], r['sample_id']) for r in normalized]
    if keys != sorted(keys):
        raise ModelDashboardError('排序分數或同分顯示順序不一致')
    k = math.ceil(.2 * len(normalized))
    cutoff = normalized[k-1]['risk_score']
    above = sum(r['risk_score'] > cutoff for r in normalized)
    tied = sum(r['risk_score'] == cutoff for r in normalized)
    previous, rank = None, 0
    for position, row in enumerate(normalized, 1):
        if row['risk_score'] != previous:
            rank = position
        if row['display_position'] != position or row['score_rank'] != rank:
            raise ModelDashboardError(f"{row['sample_id']} 名次或顯示序號不一致")
        weight = 1 if row['risk_score'] > cutoff else (k-above)/tied if row['risk_score']==cutoff else 0
        _close(row['top20_weight'], weight, '全體 Top20 同分權重')
        previous = row['risk_score']
    return normalized


def filter_ranked_rows(
    rows: Sequence[Mapping[str, Any]], *, country: str | None = None,
    product_code: str | None = None, top20_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter display only, preserving global ordering, ranks and weights."""
    return [dict(row) for row in rows if (country is None or row['Country'] == country)
            and (product_code is None or row['ProdCode'] == product_code)
            and (not top20_only or row['top20_weight'] > 0)]


def _metrics(metrics: Mapping, n: int, positives: int) -> None:
    k = math.ceil(.2*n)
    bounded = ('average_precision','roc_auc','brier','precision_at_0_5','recall_at_0_5',
               'f1_at_0_5','top20_cutoff_score','top20_expected_precision','top20_expected_recall')
    for name in bounded:
        value = _number(_need(metrics,name), name)
        if not 0 <= value <= 1:
            raise ModelDashboardError(f'{name} 須在 0–1')
    matrix = _need(metrics, 'confusion_matrix')
    if not isinstance(matrix,list) or len(matrix)!=2 or any(not isinstance(r,list) or len(r)!=2 for r in matrix):
        raise ModelDashboardError('confusion_matrix 須為 2×2')
    tn,fp,fn,tp = [_integer(value,'confusion_matrix') for row in matrix for value in row]
    if tn+fp != n-positives or fn+tp != positives:
        raise ModelDashboardError('confusion_matrix 樣本／分類計數不一致')
    precision, recall = tp/(tp+fp) if tp+fp else 0, tp/positives
    for name, expected in (('precision_at_0_5',precision), ('recall_at_0_5',recall),
                           ('f1_at_0_5',2*precision*recall/(precision+recall) if precision+recall else 0)):
        _close(metrics[name],expected,name)
    if _integer(_need(metrics,'top20_samples'),'top20_samples') != k:
        raise ModelDashboardError('top20_samples 不一致')
    tied = _integer(_need(metrics,'top20_boundary_tied_samples'),'邊界同分數')
    slots = _integer(_need(metrics,'top20_boundary_slots'),'邊界名額')
    if not 1 <= slots <= min(k,tied) or tied > n:
        raise ModelDashboardError('Top20 邊界名額或同分數無效')
    expected = _number(_need(metrics,'top20_expected_positives'),'Top20 期望不利數')
    if not max(0,k-(n-positives)) <= expected <= min(k,positives):
        raise ModelDashboardError('Top20 期望不利數無效')
    for name,value in (('top20_expected_precision',expected/k),
                       ('top20_expected_recall',expected/positives),
                       ('top20_expected_lift',(expected/k)/(positives/n))):
        actual = _number(_need(metrics,name),name)
        _close(actual,value,name)


def load_model_dashboard(
    comparison_path: Path = DEFAULT_COMPARISON_PATH,
    ranking_report_path: Path = DEFAULT_RANKING_REPORT_PATH,
    ranking_csv_path: Path = DEFAULT_RANKING_CSV_PATH,
    source_path: Path = DEFAULT_SOURCE_PATH,
) -> dict[str, Any]:
    """Read each file once, checking the same bytes that are parsed and displayed."""
    paths = dict(comparison=comparison_path, ranking_report=ranking_report_path,
                 ranking_csv=ranking_csv_path, source=source_path)
    raw = {}
    for name,path in paths.items():
        try:
            raw[name] = path.read_bytes()
        except OSError as exc:
            raise ModelDashboardError(f'{name} 無法讀取：{path}：{exc}') from exc
    comparison = _json(raw['comparison'],'F07 比較報告')
    report = _json(raw['ranking_report'],'F08 排序報告')
    for obj in (comparison,report):
        if _integer(_need(obj,'schema_version'),'schema_version') != 1:
            raise ModelDashboardError('schema_version 不支援')
        if _need(obj,'features') != list(FEATURES):
            raise ModelDashboardError('模型特徵白名單不一致')
        source = _object(_need(obj,'source'),'source')
        if _need(source,'sha256') != hashlib.sha256(raw['source']).hexdigest():
            raise ModelDashboardError('來源 SHA256 不一致')
        if _need(obj,'top20_policy') != TOP_POLICY:
            raise ModelDashboardError('Top20 政策不一致')
    source_fields, source_rows = _csv(raw['source'],'來源')
    try:
        train,test = split_samples(source_fields,source_rows)
    except BaselineDataError as exc:
        raise ModelDashboardError(str(exc)) from exc
    split = dict(train_years=[2019,2020,2021,2022], test_year=2023,
                 train_samples=len(train),train_violative=sum(int(r['violative']) for r in train),
                 test_samples=len(test),test_violative=sum(int(r['violative']) for r in test),overlap_samples=0)
    for obj in (comparison,report):
        saved = _object(_need(obj,'temporal_split'),'temporal_split')
        if saved.get('train_years') != split['train_years'] or any(type(y) is not int for y in saved['train_years']):
            raise ModelDashboardError('訓練年度不一致')
        for key,value in split.items():
            if key!='train_years' and _integer(_need(saved,key),key)!=value:
                raise ModelDashboardError(f'時間切分 {key} 不一致')
        if _integer(_need(obj,'input_samples'),'input_samples')!=len(source_rows):
            raise ModelDashboardError('建模樣本數不一致')
    if _number(_need(comparison,'threshold'),'threshold') != .5 or _need(comparison,'confusion_matrix_order')!='[[TN, FP], [FN, TP]]':
        raise ModelDashboardError('門檻或混淆矩陣定義不一致')
    models = _object(_need(comparison,'models'),'models')
    if set(models) != set(MODEL_NAMES):
        raise ModelDashboardError('四種模型清單不一致')
    for name in MODEL_NAMES:
        item = _object(models[name],name)
        _metrics(_object(_need(item,'metrics'),f'{name} metrics'),len(test),split['test_violative'])
    lr = _object(models['logistic_regression'],'LR')
    for params in (_need(lr,'parameters'),_need(report,'parameters')):
        if params != LR_PARAMETERS or any(type(params[k]) is not type(v) for k,v in LR_PARAMETERS.items()):
            raise ModelDashboardError('LR 固定參數不一致')
    if _need(report,'model') != 'logistic_regression':
        raise ModelDashboardError('排序展示模型須為 LR')
    lr_metrics = _object(_need(report,'metrics'),'LR metrics')
    _metrics(lr_metrics,len(test),split['test_violative'])
    if set(lr_metrics)!=set(lr['metrics']):
        raise ModelDashboardError('F07/F08 LR 指標欄位不一致')
    for name,value in lr_metrics.items():
        other=lr['metrics'][name]
        if isinstance(value,list):
            if value!=other:
                raise ModelDashboardError(f'F07/F08 {name} 不一致')
        else:
            _close(_number(value,name),_number(other,name),f'F07/F08 {name}')
    csv_meta = _object(_need(report,'ranking_csv'),'ranking_csv')
    if _need(csv_meta,'sha256') != hashlib.sha256(raw['ranking_csv']).hexdigest():
        raise ModelDashboardError('排序 CSV SHA256 不一致')
    fields, csv_rows = _csv(raw['ranking_csv'],'排序')
    if _need(csv_meta,'fields') != list(RANKING_FIELDS):
        raise ModelDashboardError('排序報告欄位不一致')
    for obj,key in ((csv_meta,'rows'),(report,'ranking_samples')):
        if _integer(_need(obj,key),key)!=len(test):
            raise ModelDashboardError('排序報告列數不一致')
    ranked = validate_ranking(fields,csv_rows,intercept=_need(report,'intercept'))
    source_by_id={r['sample_id']:r for r in test}
    if {r['sample_id'] for r in ranked}!=set(source_by_id):
        raise ModelDashboardError('排序樣本鍵集合與來源 FY2023 不一致')
    categories={f:{r[f] for r in train} for f in FEATURES}
    for row in ranked:
        original=source_by_id[row['sample_id']]
        if any(row[f]!=original[f] for f in ('Year',*FEATURES,'violative')):
            raise ModelDashboardError(f"{row['sample_id']} 特徵或歷史答案與來源不一致")
        if row['unseen_features']!='|'.join(f for f in FEATURES if row[f] not in categories[f]):
            raise ModelDashboardError(f"{row['sample_id']} 未見類別標記與訓練資料不一致")
    weight_sum=math.fsum(r['top20_weight'] for r in ranked)
    positive_sum=math.fsum(r['top20_weight']*int(r['violative']) for r in ranked)
    # Accumulated weights allow small summation differences while each row stays strict.
    _close(weight_sum,_number(_need(report,'top20_weight_sum'),'top20_weight_sum'),'權重和',1e-9)
    _close(positive_sum,_number(_need(report,'top20_positive_weight_sum'),'top20_positive_weight_sum'),'答案權重和',1e-9)
    _close(positive_sum,lr_metrics['top20_expected_positives'],'全體期望不利數',1e-9)
    cutoff=ranked[math.ceil(.2*len(ranked))-1]['risk_score']
    tied=sum(r['risk_score']==cutoff for r in ranked)
    slots=math.ceil(.2*len(ranked))-sum(r['risk_score']>cutoff for r in ranked)
    if lr_metrics['top20_boundary_tied_samples']!=tied or lr_metrics['top20_boundary_slots']!=slots:
        raise ModelDashboardError('LR 邊界同分統計不一致')
    _close(cutoff,lr_metrics['top20_cutoff_score'],'LR 邊界分數')
    matrix=[[sum(r['violative']=='0' and r['risk_score']<.5 for r in ranked),
             sum(r['violative']=='0' and r['risk_score']>=.5 for r in ranked)],
            [sum(r['violative']=='1' and r['risk_score']<.5 for r in ranked),
             sum(r['violative']=='1' and r['risk_score']>=.5 for r in ranked)]]
    if matrix!=lr_metrics['confusion_matrix']:
        raise ModelDashboardError('LR 混淆矩陣與排序 CSV 不一致')
    _close(math.fsum((int(r['violative'])-r['risk_score'])**2 for r in ranked)/len(ranked),lr_metrics['brier'],'LR Brier')
    baseline_probability=split['train_violative']/len(train)
    _close(_number(_need(comparison,'baseline_probability'),'baseline_probability'),baseline_probability,'基準比例')
    return dict(comparison=comparison,ranking_report=report,ranked_rows=ranked,
                countries=sorted({r['Country'] for r in ranked}),
                product_codes=sorted({r['ProdCode'] for r in ranked}),
                source_paths={name:str(path.resolve()) for name,path in paths.items()})
