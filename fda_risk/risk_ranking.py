"""Display held-out logistic scores with exact additive model explanations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from fda_risk.model_comparison import TOP_POLICY, fit_comparison
from fda_risk.modeling_dataset import FEATURES, load_samples
from fda_risk.temporal_baseline import DEFAULT_INPUT_PATH, TOP_FRACTION, split_samples

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / 'data/results/fda_risk_ranking_2023.csv'
DEFAULT_JSON_PATH = PROJECT_ROOT / 'data/results/fda_risk_ranking_report.json'
CONTRIBUTION_FIELDS = tuple(f'{field}_contribution' for field in FEATURES)
RANKING_FIELDS = ('display_position', 'score_rank', 'sample_id', 'Year', *FEATURES,
                  'risk_score', 'top20_weight', *CONTRIBUTION_FIELDS,
                  'intercept', 'logit', 'reconstructed_score', 'unseen_features', 'violative')


class RankingError(ValueError):
    """Raised when ranking, explanation or publication checks fail."""


def rank_samples(samples: Sequence[Mapping[str, object]], scores: Sequence[float]) -> list[dict]:
    """Rank by score only; sample_id orders the display of ties, not their credit."""
    p = [float(value) for value in scores]
    if not samples or len(samples) != len(p):
        raise RankingError('樣本與分數須等長且非空')
    ids = [str(row.get('sample_id', '')).strip() for row in samples]
    if any(not key for key in ids) or len(set(ids)) != len(ids):
        raise RankingError('排序樣本鍵須非空且唯一')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p):
        raise RankingError('排序分數須為 0–1 的有限數值')
    k = math.ceil(TOP_FRACTION * len(p))
    cutoff = sorted(p, reverse=True)[k - 1]
    above = sum(value > cutoff for value in p)
    tied = sum(value == cutoff for value in p)
    ranked = []
    previous, score_rank = None, 0
    order = sorted(range(len(p)), key=lambda i: (-p[i], ids[i]))
    for position, i in enumerate(order, start=1):
        if p[i] != previous:
            score_rank = position
        weight = 1.0 if p[i] > cutoff else (k - above) / tied if p[i] == cutoff else 0.0
        ranked.append(dict(samples[i], display_position=position, score_rank=score_rank,
                           risk_score=p[i], top20_weight=weight))
        previous = p[i]
    return ranked


def build_ranking(fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> tuple[list[dict], dict]:
    """Reuse the F07 logistic recipe and decompose its trained linear predictor."""
    import numpy as np
    from scipy.special import expit

    _, test = split_samples(fieldnames, rows)
    comparison, pipelines = fit_comparison(fieldnames, rows, model_names=('logistic_regression',))
    pipeline = pipelines['logistic_regression']
    encoder, model = pipeline.named_steps['preprocess'], pipeline.named_steps['model']
    x = [[row[field] for field in FEATURES] for row in test]
    encoded = encoder.transform(x)
    names = list(map(str, encoder.get_feature_names_out(FEATURES)))
    weights = model.coef_[0]
    intercept = float(model.intercept_[0])
    indices = {field: [] for field in FEATURES}
    coefficients = []
    for i, name in enumerate(names):
        field = next((field for field in FEATURES if name.startswith(field + '_')), None)
        if field is None:
            raise RankingError('無法將編碼欄位對應至四個核准特徵')
        indices[field].append(i)
        coefficients.append(dict(field=field, encoded_category=name[len(field) + 1:],
                                 coefficient=float(weights[i])))
    if len(names) != len(weights) or any(not indices[field] for field in FEATURES):
        raise RankingError('編碼欄位與模型係數不一致')
    contributions = {field: np.asarray(encoded[:, columns] @ weights[columns]).ravel()
                     for field, columns in indices.items()}
    summed = intercept + sum(contributions.values())
    logits = np.asarray(pipeline.decision_function(x))
    scores = pipeline.predict_proba(x)[:, 1]
    reconstructed = expit(summed)
    logit_error = float(np.max(np.abs(summed - logits)))
    score_error = float(np.max(np.abs(reconstructed - scores)))
    if not np.all(np.isfinite(summed)) or logit_error > 1e-10 or score_error > 1e-12:
        raise RankingError('貢獻加總無法還原模型分數，停止輸出')
    fitted = comparison['models']['logistic_regression']
    categories = {field: set(values) for field, values in fitted['preprocessing']['train_categories'].items()}
    explained = []
    for i, row in enumerate(test):
        explained.append(dict(row, **{f'{field}_contribution': float(contributions[field][i])
                                       for field in FEATURES},
                              intercept=intercept, logit=float(logits[i]),
                              reconstructed_score=float(reconstructed[i]),
                              unseen_features='|'.join(field for field in FEATURES if row[field] not in categories[field])))
    ranked = rank_samples(explained, scores)
    weight_sum = math.fsum(row['top20_weight'] for row in ranked)
    expected_positives = math.fsum(row['top20_weight'] * int(row['violative']) for row in ranked)
    metrics = fitted['metrics']
    if not math.isclose(weight_sum, metrics['top20_samples'], abs_tol=1e-9, rel_tol=0) or not math.isclose(
        expected_positives, metrics['top20_expected_positives'], abs_tol=1e-9, rel_tol=0,
    ):
        raise RankingError('排序權重與 F07 評估政策不一致')
    positive = sorted((row for row in coefficients if row['coefficient'] > 0),
                      key=lambda row: (-row['coefficient'], row['field'], row['encoded_category']))[:10]
    negative = sorted((row for row in coefficients if row['coefficient'] < 0),
                      key=lambda row: (row['coefficient'], row['field'], row['encoded_category']))[:10]
    report = {
        'schema_version': 1, 'model': 'logistic_regression',
        'model_role': 'Interpretability demonstration, not a selected final deployment model.',
        'score_interpretation': 'Class-weighted model scores are not calibrated market violation probabilities or food safety determinations.',
        'input_samples': comparison['input_samples'], 'ranking_samples': len(ranked),
        'features': list(FEATURES), 'temporal_split': comparison['temporal_split'],
        'parameters': fitted['parameters'], 'preprocessing': fitted['preprocessing'],
        'versions': comparison['versions'], 'metrics': metrics,
        'intercept': intercept, 'coefficients': coefficients,
        'largest_positive_coefficients': positive, 'largest_negative_coefficients': negative,
        'max_logit_reconstruction_error': logit_error,
        'max_score_reconstruction_error': score_error,
        'top20_weight_sum': weight_sum, 'top20_positive_weight_sum': expected_positives,
        'top20_policy': TOP_POLICY,
        'display_policy': 'Descending full-precision score, then sample_id for tied display only. score_rank is competition rank; tied samples share it. top20_weight is inclusion probability, not a realized selection.',
        'explanation_policy': 'intercept + four field contributions = logit; expit(logit) reconstructs score. Contributions and coefficients are logit units, not percentage points or causal effects. Full one-hot coefficients are model-specific, not contrasts against a dropped reference category.',
        'selection_policy': comparison['selection_policy'],
        'evaluation_history': comparison['evaluation_history'],
        'interpretation': comparison['interpretation'],
    }
    return ranked, report


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def save_outputs(ranked: Sequence[Mapping], report: Mapping, csv_path: Path, json_path: Path) -> None:
    """Stage and verify both files; restore the CSV if JSON publication fails.

    Each replace is atomic, not the pair. The JSON hash detects interrupted runs;
    ordinary publication errors roll back, but process/power loss cannot do so.
    """
    if _same_path(csv_path, json_path):
        raise RankingError('CSV 與 JSON 輸出位置不可相同')
    for path in (csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    rollback = None
    published_csv = False
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                         dir=csv_path.parent, suffix='.tmp', delete=False) as handle:
            staged_csv = Path(handle.name)
            temporary.append(staged_csv)
            writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
            writer.writeheader()
            writer.writerows(ranked)
        with staged_csv.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RANKING_FIELDS:
                raise RankingError('CSV 讀回欄位順序不一致')
            reread = list(reader)
        if len(reread) != report['ranking_samples'] or len({row['sample_id'] for row in reread}) != len(reread):
            raise RankingError('CSV 讀回列數或唯一鍵不一致')
        if not math.isclose(math.fsum(float(row['top20_weight']) for row in reread),
                            report['top20_weight_sum'], abs_tol=1e-9, rel_tol=0):
            raise RankingError('CSV 讀回 Top20 權重不一致')
        saved_report = dict(report, ranking_csv={
            'path': str(csv_path.resolve()), 'sha256': hashlib.sha256(staged_csv.read_bytes()).hexdigest(),
            'rows': len(reread), 'fields': list(RANKING_FIELDS),
        })
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=json_path.parent,
                                         suffix='.tmp', delete=False) as handle:
            staged_json = Path(handle.name)
            temporary.append(staged_json)
            json.dump(saved_report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        with staged_json.open(encoding='utf-8') as handle:
            if json.load(handle) != saved_report:
                raise RankingError('JSON 讀回內容與預期報告不一致')
        if csv_path.exists():
            with tempfile.NamedTemporaryFile(dir=csv_path.parent, suffix='.rollback', delete=False) as handle:
                rollback = Path(handle.name)
                handle.write(csv_path.read_bytes())
        staged_csv.replace(csv_path)
        published_csv = True
        staged_json.replace(json_path)
    except OSError:
        if published_csv:
            if rollback is None:
                csv_path.unlink()
            else:
                try:
                    rollback.replace(csv_path)
                except OSError as exc:
                    preserved = rollback
                    rollback = None  # Keep the recovery file when the OS also denies rollback.
                    raise RankingError(f'輸出失敗且無法回復 CSV；舊資料保留於 {preserved}') from exc
        raise
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()
        if rollback is not None and rollback.exists():
            rollback.unlink()


def format_summary(report: Mapping, csv_path: Path, json_path: Path) -> str:
    metrics = report['metrics']
    lines = ['FDA農藥殘留專案 可解釋風險排序展示（Logistic Regression）',
             f"FY2023 排序樣本：{report['ranking_samples']:,}",
             f"Top 20% 名額／權重合計：{metrics['top20_samples']}／{report['top20_weight_sum']:.6f}",
             f"Top 20% 期望不利樣本／Recall／Lift：{report['top20_positive_weight_sum']:.6f}／{metrics['top20_expected_recall']:.6f}／{metrics['top20_expected_lift']:.6f}",
             f"貢獻還原最大誤差（logit／分數）：{report['max_logit_reconstruction_error']:.3e}／{report['max_score_reconstruction_error']:.3e}"]
    for title, key in (('正向係數 Top 10', 'largest_positive_coefficients'),
                       ('負向係數 Top 10', 'largest_negative_coefficients')):
        lines.append(title + '（logit 單位，不是因果或百分點）：')
        for item in report[key]:
            lines.append(f"  {item['field']}={item['encoded_category']}：{item['coefficient']:.6f}")
    lines += ['同分共用 score_rank；sample_id 僅決定顯示順序，top20_weight 不是實際選取旗標。',
              '這是解釋展示模型，不是最終部署模型；分數不是已校準市場機率。',
              'FDA 目標式抽驗結果不代表整體市場違規率，也不能直接判定食品安全。',
              f'排序 CSV：{csv_path}', f'說明 JSON：{json_path}']
    return '\n'.join(lines)


def main(input_path: Path | None = None, csv_path: Path | None = None,
         json_path: Path | None = None) -> int:
    source = DEFAULT_INPUT_PATH if input_path is None else input_path
    csv_output = DEFAULT_CSV_PATH if csv_path is None else csv_path
    json_output = DEFAULT_JSON_PATH if json_path is None else json_path
    try:
        if any(_same_path(a, b) for a, b in ((source, csv_output), (source, json_output), (csv_output, json_output))):
            raise RankingError('輸入、CSV 與 JSON 位置不可互相覆寫')
        original = source.read_bytes()
        fields, rows = load_samples(source)
        if source.read_bytes() != original:
            raise RankingError('讀取期間來源資料變更，請重新執行')
        ranked, report = build_ranking(fields, rows)
        report['source'] = {'path': str(source.resolve()), 'sha256': hashlib.sha256(original).hexdigest()}
        save_outputs(ranked, report, csv_output, json_output)
        print(format_summary(report, csv_output, json_output))
    except ImportError as exc:
        print(f'錯誤：請依 requirements-models.txt 準備模型套件；{exc}', file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f'錯誤：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
