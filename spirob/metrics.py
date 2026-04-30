from __future__ import annotations

import math
import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(y_true, y_prob, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        'accuracy': _safe_div(tp + tn, len(y_true)),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': _binary_auc(y_true, y_prob),
    }


def multiclass_metrics(y_true, logits) -> dict:
    y_true = np.asarray(y_true).astype(int)
    logits = np.asarray(logits).astype(float)
    if logits.ndim == 1:
        logits = logits[:, None]
    y_pred = np.argmax(logits, axis=1)
    acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    n_cls = int(max(logits.shape[1], int(y_true.max()) + 1 if len(y_true) else 1))
    recalls = []
    for c in range(n_cls):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float(np.mean(y_pred[mask] == c)))
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    return {'accuracy': acc, 'balanced_accuracy': bal_acc}


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else 0.0
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred)))) if len(y_true) else 0.0
    if len(y_true) == 0:
        r2 = 0.0
    else:
        ss_res = float(np.sum(np.square(y_true - y_pred)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = 1.0 - _safe_div(ss_res, ss_tot)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def trust_metrics(y_true, y_pred) -> dict:
    base = regression_metrics(y_true, y_pred)
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    base['corr'] = float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1]) if y_true.size > 1 else 0.0
    return base


def robustness_breakdown(corruption_types, y_true, y_prob) -> dict:
    corruption_types = list(corruption_types)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    report = {}
    for name in sorted(set(corruption_types)):
        idx = [i for i, c in enumerate(corruption_types) if c == name]
        if not idx:
            continue
        report[name] = binary_metrics(y_true[idx], y_prob[idx])
    return report
