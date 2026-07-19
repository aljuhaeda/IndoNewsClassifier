# IndoNewsClassifier

*Work in progress — this README will be rewritten with real results once the notebook is complete. Every number that lands here will come from a cell in `notebooks/`.*

## Problem

Classify Indonesian-language news titles into one of 9 topic categories (finance, food, health, hot, inet, news, oto, sport, travel), comparing a classical baseline against a fine-tuned transformer.

## Dataset

[Indonesian News Title](https://www.kaggle.com/datasets/ibamibrahim/indonesian-news-title) (also mirrored on [GitHub](https://github.com/ibamibrahim/dataset-judul-berita-indonesia)) — 90k+ titles scraped from detik.com, single-label category per title.

## Repo structure

```
IndoNewsClassifier/
├── notebooks/     EDA → preprocessing → baseline → transformer → evaluation
├── src/           reusable modules (preprocessing, training, inference)
├── models/        committed baseline checkpoint (transformer weights on HF Hub instead)
├── docs/          confusion matrices, charts for the README
├── app.py         Streamlit inference app
└── requirements.txt
```

## Status

- [ ] EDA
- [ ] TF-IDF + Logistic Regression baseline
- [ ] IndoBERT fine-tune
- [ ] Streamlit app
- [ ] Deployed to HuggingFace Spaces
- [ ] Results filled into this README

## License

MIT — see [LICENSE](LICENSE).
