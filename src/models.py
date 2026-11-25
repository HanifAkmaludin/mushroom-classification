"""
Model training and persistence module for Decision Tree and Random Forest.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .config import MODELS_DIR


def build_decision_tree(
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Optional[Any] = "balanced",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Constructs a DecisionTreeClassifier instance.
    Uses 'balanced' class weights by default to emphasize penalizing poisonous false negatives.
    """
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )


def build_random_forest(
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Optional[Any] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Constructs a RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def tune_decision_tree(
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "recall",
    random_state: int = 42
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Tune Decision Tree hyperparameters with 5-Fold Stratified CV,
    optimizing explicitly for 'recall' (sensitivity for poisonous class).
    """
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 5}],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest(
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "recall",
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Tune Random Forest hyperparameters with 5-Fold Stratified CV,
    optimizing explicitly for 'recall' (sensitivity for poisonous class).
    """
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "class_weight": ["balanced", None],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def save_model(model: Any, filename: str) -> Path:
    """
    Save trained model artifact using joblib.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / filename
    joblib.dump(model, save_path)
    return save_path


def load_model(filename: str) -> Any:
    """
    Load saved model artifact.
    """
    load_path = MODELS_DIR / filename
    return joblib.load(load_path)
