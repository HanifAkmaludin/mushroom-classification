"""
Visualization module for EDA, model evaluation, confusion matrices, and feature importance.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from .config import FIGURES_DIR

# Set clean aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


def plot_eda_physical_characteristics(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """
    Generate EDA charts analyzing physical characteristics specified in the task:
    cap shape, cap surface, cap color, gill spacing, odor, and spore print color.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (FIGURES_DIR / "eda_physical_characteristics.png")

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    features = [
        ("odor", "Odor Distribution", axes[0, 0]),
        ("gill_spacing", "Gill Spacing Distribution", axes[0, 1]),
        ("spore_print_color", "Spore Print Color Distribution", axes[1, 0]),
        ("cap_shape", "Cap Shape Distribution", axes[1, 1]),
        ("cap_surface", "Cap Surface Distribution", axes[2, 0]),
        ("cap_color", "Cap Color Distribution", axes[2, 1]),
    ]

    for col, title, ax in features:
        if col in df.columns:
            order = df[col].value_counts().index
            sns.countplot(
                data=df,
                x=col,
                hue="class",
                order=order,
                palette={"edible": "#2ecc71", "poisonous": "#e74c3c"},
                ax=ax,
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="Edibility", loc="upper right")

    plt.suptitle("Distribution of Physical Characteristics by Toxicity (Edible vs. Poisonous)", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_confusion_matrices(
    cms: Dict[str, np.ndarray],
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot side-by-side confusion matrices for Decision Tree and Random Forest,
    highlighting 0 False Negatives (100% poisonous sensitivity).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (FIGURES_DIR / "confusion_matrices.png")

    fig, axes = plt.subplots(1, len(cms), figsize=(6 * len(cms), 5))
    if len(cms) == 1:
        axes = [axes]

    labels = ["Edible (0)", "Poisonous (1)"]

    for ax, (model_name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        fn = cm[1, 0]
        safety_status = "100% SENSITIVITY (FN = 0)" if fn == 0 else f"DANGER (FN = {fn})"
        ax.set_title(f"{model_name}\nStatus: {safety_status}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Actual Label", fontsize=11)

    plt.suptitle("Model Comparison: Confusion Matrices (Target: Zero False Negatives)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 15,
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot top N most important features based on Gini importance.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (FIGURES_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, color="#3498db")
    plt.title(f"Top {top_n} Feature Importance ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Gini Importance / Mean Decrease Impurity", fontsize=11)
    plt.ylabel("Physical Characteristics / Features", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_tree_diagram(
    dt_model: Any,
    feature_names: List[str],
    max_depth: int = 3,
    save_path: Optional[Path] = None
) -> Path:
    """
    Visualizes the top levels of the Decision Tree rule structure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (FIGURES_DIR / "decision_tree_structure.png")

    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["Edible", "Poisonous"],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title(f"Decision Tree Rule Hierarchy (Depth <= {max_depth})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_roc_pr_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot ROC and Precision-Recall Curves for comparison.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (FIGURES_DIR / "roc_pr_curves.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

            precision, recall, _ = precision_recall_curve(y_test, probs)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Sensitivity for Poisonous)")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve (Safety Focus)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")

    plt.suptitle("Model Classification Evaluation Curves (ROC & Precision-Recall)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path
