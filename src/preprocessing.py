"""Text cleaning and dataset splitting for Indonesian news titles."""

import re

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/extra whitespace from a title."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str, text_col: str = "title", label_col: str = "category") -> pd.DataFrame:
    """Load the raw CSV and drop missing/duplicate rows."""
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col])
    return df


def split_dataset(
    df: pd.DataFrame,
    label_col: str = "category",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split."""
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, stratify=df[label_col], random_state=random_state
    )
    relative_test_size = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, stratify=temp_df[label_col], random_state=random_state
    )
    return train_df, val_df, test_df
