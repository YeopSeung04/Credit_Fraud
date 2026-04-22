"""
evaluate.py — 불균형 분류 평가 전담 모듈
핵심: 불균형 데이터에서 AUC만 보면 안 됨 → Average Precision이 주 지표
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from src.utils import get_logger

logger = get_logger(__name__)


def find_best_threshold(y_true, y_prob, strategy: str = "f1") -> float:
    """
    최적 분류 임계값 탐색
    strategy:
      'f1'       - F1 최대화
      'precision' - Precision 0.9 이상에서 최대 Recall
      'youden'   - Youden's J (TPR - FPR 최대)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if strategy == "f1":
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        idx = np.argmax(f1s[:-1])
    elif strategy == "precision":
        mask = precisions[:-1] >= 0.90
        if mask.any():
            idx = np.where(mask)[0][np.argmax(recalls[:-1][mask])]
        else:
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
            idx = np.argmax(f1s[:-1])
    elif strategy == "youden":
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        j = tpr - fpr
        best_thresh = roc_thresh[np.argmax(j)]
        return float(best_thresh)
    else:
        raise ValueError(f"알 수 없는 strategy: {strategy}")

    return float(thresholds[idx])


def evaluate(
    y_true,
    y_prob,
    threshold: float = None,
    strategy: str = "f1",
    verbose: bool = True,
) -> dict:
    """
    전체 평가 지표 계산
    Returns:
        dict with all metrics
    """
    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob, strategy)

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc": round(roc_auc, 6),
        "average_precision": round(avg_prec, 6),
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "threshold": round(threshold, 6),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # 비즈니스 지표
        "fraud_caught_rate": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "false_alarm_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 6),
    }

    if verbose:
        logger.info("─" * 50)
        logger.info(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        logger.info(f"  Average Precision: {metrics['average_precision']:.4f}  ← 주 지표")
        logger.info(f"  F1 Score:          {metrics['f1']:.4f}")
        logger.info(f"  Precision:         {metrics['precision']:.4f}")
        logger.info(f"  Recall:            {metrics['recall']:.4f}")
        logger.info(f"  Threshold:         {metrics['threshold']:.4f}")
        logger.info(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        logger.info(f"  사기 탐지율:        {metrics['fraud_caught_rate']:.2%}")
        logger.info(f"  오탐율 (정상→사기): {metrics['false_alarm_rate']:.4%}")
        logger.info("─" * 50)

    return metrics


def get_curve_data(y_true, y_prob) -> dict:
    """ROC / PR 커브 데이터 반환 (시각화용)"""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)

    # 포인트 수 줄이기 (노트북 렌더링용)
    n = 300
    roc_idx = np.linspace(0, len(fpr) - 1, min(n, len(fpr)), dtype=int)
    pr_idx  = np.linspace(0, len(prec) - 1, min(n, len(prec)), dtype=int)

    return {
        "roc": {"fpr": fpr[roc_idx].tolist(), "tpr": tpr[roc_idx].tolist()},
        "pr":  {"precision": prec[pr_idx].tolist(), "recall": rec[pr_idx].tolist()},
    }


def compare_models(results: dict) -> pd.DataFrame:
    """여러 모델 결과를 DataFrame으로 비교"""
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "ROC-AUC": m["roc_auc"],
            "Avg Precision": m["average_precision"],
            "F1": m["f1"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
        })
    df = pd.DataFrame(rows).sort_values("Avg Precision", ascending=False)
    return df
