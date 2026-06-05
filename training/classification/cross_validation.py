"""
"""

import logging
from typing import Callable, List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

logger = logging.getLogger(__name__)


def run_stratified_kfold(
    train_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    """
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        logger.info(f"Fold {fold}/{k}: train={len(train_idx)}, val={len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_result = train_fn(X_train, X_val, y_train, y_val)
        fold_result["fold"] = fold
        results.append(fold_result)

    return results


def run_kfold(
    train_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    """
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        logger.info(f"Fold {fold}/{k}: train={len(train_idx)}, val={len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_train, y_val = y[train_idx], y[val_idx]

        fold_result = train_fn(X_train, X_val, y_train, y_val)
        fold_result["fold"] = fold
        results.append(fold_result)

    return results


def summarize_cv_results(
    results: List[Dict[str, Any]],
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    """
    if not results:
        return {}

    #
    if metric_names is None:
        metric_names = []
        for key, val in results[0].items():
            if key == "fold":
                continue
            if isinstance(val, (int, float, np.floating)):
                metric_names.append(key)

    summary = {}
    for metric in metric_names:
        values = [r.get(metric, 0.0) for r in results]
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"std_{metric}"] = float(np.std(values))
        summary[f"min_{metric}"] = float(np.min(values))
        summary[f"max_{metric}"] = float(np.max(values))

    summary["n_folds"] = len(results)
    return summary


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
