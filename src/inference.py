"""Prediction helpers shared by the notebook and the Streamlit app."""

from src.preprocessing import clean_text
from src.training import load_baseline


def predict_baseline(pipeline, text: str) -> tuple[str, dict[str, float]]:
    """Return (predicted_label, {label: probability}) for one raw title."""
    cleaned = clean_text(text)
    label = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    scores = dict(zip(pipeline.classes_, proba))
    return label, scores


def predict_indobert(model, tokenizer, text: str) -> tuple[str, dict[str, float]]:
    """Return (predicted_label, {label: probability}) for one raw title using a fine-tuned IndoBERT model."""
    import torch

    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    scores = {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    label = max(scores, key=scores.get)
    return label, scores
