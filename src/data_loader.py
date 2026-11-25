"""
Data loading and preprocessing module for Mushroom Classification.
"""
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from .config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    COLUMN_NAMES,
    VALUE_MAPPINGS,
)


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw UCI mushroom data file.
    """
    path = filepath or RAW_DATA_PATH
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def map_to_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map single-letter abbreviations to human-readable botanical terms.
    """
    df_mapped = df.copy()
    for col, mapping in VALUE_MAPPINGS.items():
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].map(mapping).fillna(df_mapped[col])
    return df_mapped


def prepare_dataset(
    filepath: Optional[str] = None,
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads raw data, transforms abbreviations to descriptive terms,
    handles constant features, and prepares features X and target y.
    
    Returns:
        df_clean: DataFrame with descriptive human-readable values.
        X: Feature DataFrame (with zero-variance columns dropped).
        y: Binary target Series (1 = poisonous, 0 = edible).
    """
    df_raw = load_raw_data(filepath)
    df_clean = map_to_descriptive(df_raw)

    # Save processed human-readable dataset for EDA & inspection
    if save_processed:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

    # Target variable: 1 for poisonous (positive class for recall/sensitivity), 0 for edible
    y = (df_clean["class"] == "poisonous").astype(int)

    # Drop target column
    X = df_clean.drop(columns=["class"])

    # veil_type has only one constant value ('partial') across all 8124 rows (zero variance)
    if "veil_type" in X.columns and X["veil_type"].nunique() <= 1:
        X = X.drop(columns=["veil_type"])

    return df_clean, X, y


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    One-hot encodes categorical features and performs stratified train/test split.
    """
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
