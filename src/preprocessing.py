"""Text cleaning and dataset splitting for Indonesian news titles."""

import re

import pandas as pd
from sklearn.model_selection import train_test_split

# Informal/journalistic filler words that Sastrawi's default Indonesian
# stopword list misses but that dominate news-title word clouds without
# carrying topic signal.
EXTRA_STOPWORDS = {
    "tak", "jadi", "bikin", "soal", "hingga", "usai", "mau", "cara", "buat",
    "kini", "sebut", "bakal", "lewat", "sempat", "kata", "punya",
}


def get_stopwords() -> set[str]:
    """Sastrawi's default Indonesian stopwords plus EXTRA_STOPWORDS."""
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    return set(StopWordRemoverFactory().get_stop_words()) | EXTRA_STOPWORDS


def clean_text(text: str, stopwords: set[str] | None = None) -> str:
    """Lowercase, strip URLs/punctuation/extra whitespace from a title.

    If `stopwords` is given, also drops any token present in that set.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if stopwords:
        text = " ".join(w for w in text.split() if w not in stopwords)
    return text


def light_clean_text(text: str) -> str:
    """Minimal cleaning for transformer input: strip URLs and extra whitespace only.

    Keeps punctuation, casing, and stopwords intact — IndoBERT's own subword
    tokenizer and self-attention rely on full sentence structure, so the
    aggressive stripping used for the TF-IDF baseline (clean_text) would
    throw away signal here rather than help.
    """
    text = re.sub(r"http\S+|www\.\S+", " ", text)
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
