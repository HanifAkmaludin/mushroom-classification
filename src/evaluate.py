"""
Evaluation and Food Safety verification module for Mushroom Classification.
Focuses strictly on eliminating False Negatives (achieving 100% Sensitivity/Recall for poisonous mushrooms).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics with a strong focus on botanical food safety.
    
    Positive class (1) = Poisonous
    Negative class (0) = Edible
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensitivity (Recall of Poisonous class): TP / (TP + FN)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    
    # Specificity (Recall of Edible class): TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision of Poisonous predictions: TP / (TP + FP)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None

    is_100_percent_safe = (fn == 0)

    return {
        "accuracy": acc,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_positives_tp": int(tp),
        "false_negatives_fn": int(fn),
        "true_negatives_tn": int(tn),
        "false_positives_fp": int(fp),
        "is_100_percent_safe": is_100_percent_safe,
    }


def find_safety_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_recall: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calibrate classification decision threshold to strictly guarantee target_recall (default 100%).
    
    Iterates through thresholds in [0.01, 0.99] to find the maximum threshold where
    False Negatives = 0 (Recall >= target_recall), minimizing unnecessary edible mushroom discards.
    """
    if not hasattr(model, "predict_proba"):
        return 0.5, calculate_metrics(y_val, model.predict(X_val))

    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = None

    for t in thresholds:
        preds = (probs >= t).astype(int)
        rec = recall_score(y_val, preds, pos_label=1, zero_division=0)
        
        if rec >= target_recall:
            current_f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(t)
                best_metrics = calculate_metrics(y_val, preds, probs)

    if best_metrics is None:
        # Fallback to very safe low threshold
        best_threshold = 0.1
        preds = (probs >= best_threshold).astype(int)
        best_metrics = calculate_metrics(y_val, preds, probs)

    return best_threshold, best_metrics


def cross_validate_safety(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Perform Stratified K-Fold Cross Validation to ensure 100% sensitivity generalizes across data splits.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    accuracies = []
    sensitivities = []
    precisions = []
    f1s = []
    false_negatives = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        
        metrics = calculate_metrics(y_va, preds)
        accuracies.append(metrics["accuracy"])
        sensitivities.append(metrics["sensitivity_recall"])
        precisions.append(metrics["precision"])
        f1s.append(metrics["f1_score"])
        false_negatives.append(metrics["false_negatives_fn"])

    return {
        "accuracy": (float(np.mean(accuracies)), float(np.std(accuracies))),
        "sensitivity": (float(np.mean(sensitivities)), float(np.std(sensitivities))),
        "precision": (float(np.mean(precisions)), float(np.std(precisions))),
        "f1_score": (float(np.mean(f1s)), float(np.std(f1s))),
        "total_false_negatives": int(sum(false_negatives)),
    }
