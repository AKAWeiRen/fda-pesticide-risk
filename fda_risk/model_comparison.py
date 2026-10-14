"""Compare fixed models on the F05 dataset with a held-out fiscal year."""

from __future__ import annotations

import hashlib
import math
import platform
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

from fda_risk.modeling_dataset import FEATURES, ModelingDataError, load_samples
from fda_risk.temporal_baseline import (
    DEFAULT_INPUT_PATH, BaselineDataError, TOP_FRACTION,
    build_baseline, save_report, split_samples,
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / 'data/results/fda_model_comparison.json'
MODEL_NAMES = ('logistic_regression', 'decision_tree', 'xgboost')
TOP_POLICY = (
    'k=ceil(0.20*n_test). Include scores above the kth score; remaining slots '
    'are uniformly random within the exact-score boundary tie. Report expected '
    'positives, precision, recall and lift; no CSV-order tie breaking.'
)


class ComparisonError(ValueError):
    """Raised when predictions or fitting cannot support a complete comparison."""


def evaluate_scores(labels: Sequence[int], scores: Sequence[float]) -> dict[str, object]:
    """Evaluate scores, giving ties random-selection expectation at the cutoff."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = list(labels), [float(score) for score in scores]
    if not y or len(y) != len(p) or set(y) != {0, 1}:
        raise ComparisonError('答案與分數須等長且非空，答案須同時含有 0 與 1')
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in p):
        raise ComparisonError('預測分數須為 0–1 的有限數值')
    tn = sum(label == 0 and score < .5 for label, score in zip(y, p))
    fp = sum(label == 0 and score >= .5 for label, score in zip(y, p))
    fn = sum(label == 1 and score < .5 for label, score in zip(y, p))
    tp = sum(label == 1 and score >= .5 for label, score in zip(y, p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    k = math.ceil(len(y) * TOP_FRACTION)
    cutoff = sorted(p, reverse=True)[k - 1]
    above = [(label, score) for label, score in zip(y, p) if score > cutoff]
    boundary = [label for label, score in zip(y, p) if score == cutoff]
    slots = k - len(above)
    expected = sum(label for label, _ in above) + slots * sum(boundary) / len(boundary)
    return {
        'average_precision': float(average_precision_score(y, p)),
        'roc_auc': float(roc_auc_score(y, p)),
        'brier': sum((label - score) ** 2 for label, score in zip(y, p)) / len(y),
        'precision_at_0_5': precision, 'recall_at_0_5': recall,
        'f1_at_0_5': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'top20_samples': k, 'top20_cutoff_score': cutoff,
        'top20_boundary_tied_samples': len(boundary), 'top20_boundary_slots': slots,
        'top20_expected_positives': expected,
        'top20_expected_precision': expected / k,
        'top20_expected_recall': expected / sum(y),
        'top20_expected_lift': (expected / k) / (sum(y) / len(y)),
    }


def fit_comparison(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]],
    *, model_names: Sequence[str] = MODEL_NAMES,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit each pipeline on train only; return a report and the fitted pipelines."""
    import numpy as np
    import scipy
    import sklearn
    import xgboost
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    if not model_names or len(set(model_names)) != len(model_names) or set(model_names) - set(MODEL_NAMES):
        raise ComparisonError('模型清單不可空白、重複或包含未核准模型')
    train, test = split_samples(fieldnames, rows)
    baseline = build_baseline(fieldnames, rows)
    x_train = [[row[field] for field in FEATURES] for row in train]
    x_test = [[row[field] for field in FEATURES] for row in test]
    y_train = np.array([int(row['violative']) for row in train])
    y_test = [int(row['violative']) for row in test]
    positive = int(y_train.sum())
    negative = len(y_train) - positive
    settings = {
        'logistic_regression': dict(class_weight='balanced', max_iter=2000,
                                    C=.5, solver='liblinear', random_state=42),
        'decision_tree': dict(class_weight='balanced', max_depth=6,
                              min_samples_leaf=30, random_state=42),
        'xgboost': dict(n_estimators=250, max_depth=4, learning_rate=.05,
                        subsample=.8, colsample_bytree=.8, min_child_weight=10,
                        reg_lambda=2.0, objective='binary:logistic', eval_metric='logloss',
                        scale_pos_weight=negative / positive, random_state=42,
                        n_jobs=2, tree_method='hist'),
    }
    constructors = dict(zip(MODEL_NAMES, (LogisticRegression, DecisionTreeClassifier, XGBClassifier)))
    pipelines = {}
    results = {'prevalence_baseline': {
        'parameters': {'probability': baseline['baseline_probability']},
        'metrics': evaluate_scores(y_test, [baseline['baseline_probability']] * len(test)),
    }}
    for name in model_names:
        constructor = constructors[name]
        # Independent fitted encoders; no imputation because F05 rejects missing features.
        encoder = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=5,
                                sparse_output=True, dtype=np.float64)
        pipeline = Pipeline([('preprocess', encoder), ('model', constructor(**settings[name]))])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                pipeline.fit(x_train, y_train)
        except ConvergenceWarning as exc:
            raise ComparisonError(f'{name} 未收斂，不保存比較報告') from exc
        scores = pipeline.predict_proba(x_test)[:, 1]
        categories = {field: list(map(str, values))
                      for field, values in zip(FEATURES, encoder.categories_)}
        if categories != baseline['train_categories']:
            raise ComparisonError(f'{name} 的編碼器類別與訓練資料不一致')
        results[name] = {
            'parameters': settings[name], 'metrics': evaluate_scores(y_test, scores),
            'preprocessing': {
                'fit_samples': len(train), 'train_categories': categories,
                'test_unseen_counts': baseline['test_unseen_counts'],
                'encoded_columns': int(encoder.transform(x_train[:1]).shape[1]),
                'min_frequency': 5, 'handle_unknown': 'infrequent_if_exist',
                'sparse_output': True,
            },
        }
        pipelines[name] = pipeline
    report = {
        'schema_version': 1, 'input_samples': len(rows), 'features': list(FEATURES),
        'temporal_split': baseline['temporal_split'],
        'baseline_probability': baseline['baseline_probability'],
        'models': results, 'top20_policy': TOP_POLICY,
        'threshold': .5, 'confusion_matrix_order': '[[TN, FP], [FN, TP]]',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
                     'xgboost': xgboost.__version__},
        'selection_policy': 'Fixed settings before evaluation; no tuning, early stopping, calibration or final model selection using FY2023.',
        'evaluation_history': 'FY2023 was previously used for feasibility analysis; this is a fixed-setting demonstration, not a newly untouched test set.',
        'interpretation': baseline['interpretation'],
    }
    return report, pipelines


def format_summary(report: Mapping[str, object], output: Path) -> str:
    split = report['temporal_split']
    lines = ['FDA農藥殘留專案 三種模型固定設定比較',
             f"訓練 FY2019–FY2022：{split['train_samples']:,}／{split['train_violative']:,}",
             f"測試 FY2023：{split['test_samples']:,}／{split['test_violative']:,}；重疊 {split['overlap_samples']}",
             '模型 | AP | ROC-AUC | Brier | Precision(0.5) | Recall(0.5) | F1(0.5) | Top20期望Lift']
    for name, result in report['models'].items():
        metrics = result['metrics']
        keys = ('average_precision', 'roc_auc', 'brier', 'precision_at_0_5',
                'recall_at_0_5', 'f1_at_0_5', 'top20_expected_lift')
        lines.append(name + ' | ' + ' | '.join(f'{metrics[key]:.6f}' for key in keys))
        lines.append(f"  混淆矩陣 [[TN, FP], [FN, TP]]：{metrics['confusion_matrix']}；Top20樣本數 {metrics['top20_samples']}")
    lines += ['AP 是 Average Precision，不是梯形積分的 PR 曲線面積。',
              'Top20 邊界同分採隨機抽取期望，不使用列順序；不以測試集調參或選定最終模型。',
              '類別編碼器僅在訓練集擬合；類別權重分數不代表已校準的市場機率。',
              'FDA 目標式抽驗結果不代表整體市場違規率，也不能直接判定食品安全。',
              f'輸出位置：{output}']
    return '\n'.join(lines)


def main(input_path: Path | None = None, output_path: Path | None = None) -> int:
    source = DEFAULT_INPUT_PATH if input_path is None else input_path
    output = DEFAULT_OUTPUT_PATH if output_path is None else output_path
    try:
        if source.resolve() == output.resolve() or (
            source.exists() and output.exists() and source.samefile(output)
        ):
            raise ComparisonError('輸出不可覆寫輸入資料')
        source_bytes = source.read_bytes()
        fields, rows = load_samples(source)
        if source.read_bytes() != source_bytes:
            raise ComparisonError('讀取期間來源資料變更，請重新執行')
        report, _ = fit_comparison(fields, rows)
        report['source'] = {'path': str(source.resolve()),
                            'sha256': hashlib.sha256(source_bytes).hexdigest()}
        save_report(report, output)
        print(format_summary(report, output))
    except ImportError as exc:
        print(f'錯誤：缺少模型套件，請依 requirements-models.txt 準備環境；{exc}', file=sys.stderr)
        return 1
    except (ComparisonError, BaselineDataError, ModelingDataError, OSError, ValueError) as exc:
        print(f'錯誤：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
