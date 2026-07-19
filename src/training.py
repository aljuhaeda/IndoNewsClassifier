"""Training routines for the TF-IDF baseline and IndoBERT fine-tune."""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def train_baseline(
    train_df: pd.DataFrame,
    text_col: str = "title_clean",
    label_col: str = "category",
    max_features: int = 20000,
) -> Pipeline:
    """Fit a TF-IDF + Logistic Regression pipeline."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(train_df[text_col], train_df[label_col])
    return pipeline


def evaluate_baseline(pipeline: Pipeline, df: pd.DataFrame, text_col: str = "title_clean", label_col: str = "category"):
    """Return classification report dict and confusion matrix array."""
    preds = pipeline.predict(df[text_col])
    report = classification_report(df[label_col], preds, output_dict=True)
    labels = sorted(df[label_col].unique())
    cm = confusion_matrix(df[label_col], preds, labels=labels)
    return report, cm, labels


def save_baseline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_baseline(path: str) -> Pipeline:
    return joblib.load(path)


def build_hf_dataset(df: pd.DataFrame, tokenizer, label2id: dict, text_col: str = "title_light_clean", label_col: str = "category"):
    """Tokenize a dataframe into a HF Dataset. Shared by train/val/test so encoding stays identical."""
    from datasets import Dataset

    ds = Dataset.from_pandas(
        pd.DataFrame(
            {
                "text": df[text_col].tolist(),
                "label": df[label_col].map(label2id).tolist(),
            }
        )
    )
    return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64), batched=True)


def train_indobert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "title_light_clean",
    label_col: str = "category",
    model_name: str = "indobenchmark/indobert-base-p1",
    output_dir: str = "models/indobert-news",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
):
    """Fine-tune IndoBERT for sequence classification. Intended to run on Colab GPU.

    Import of transformers/torch is deferred to keep this module importable
    (for the baseline path / Streamlit app) even where torch isn't installed.

    Returns (trainer, tokenizer, label2id, id2label).
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    labels = sorted(train_df[label_col].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = build_hf_dataset(train_df, tokenizer, label2id, text_col, label_col)
    val_ds = build_hf_dataset(val_df, tokenizer, label2id, text_col, label_col)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, tokenizer, label2id, id2label


def evaluate_indobert(trainer, test_df: pd.DataFrame, tokenizer, label2id: dict, id2label: dict, text_col: str = "title_light_clean", label_col: str = "category"):
    """Return classification report dict, confusion matrix array, and sorted label list for a fine-tuned IndoBERT trainer."""
    test_ds = build_hf_dataset(test_df, tokenizer, label2id, text_col, label_col)
    pred_output = trainer.predict(test_ds)
    preds = pred_output.predictions.argmax(axis=-1)

    y_true = [id2label[i] for i in pred_output.label_ids]
    y_pred = [id2label[i] for i in preds]

    labels = sorted(label2id.keys())
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return report, cm, labels
