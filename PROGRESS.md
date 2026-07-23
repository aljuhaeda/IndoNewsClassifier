# IndoNewsClassifier — Progress

## Status
Deployed, stable. Live at https://indonewsclassifier.streamlit.app
(Streamlit Community Cloud free tier — sleeps after inactivity, wakes
on first visit within ~30s.)

## Done
- EDA, TF-IDF+LogReg baseline, fine-tuned IndoBERT, side-by-side
  comparison — all in `notebooks/indonewsclassifier.ipynb`.
- Real results reported honestly with both accuracy and macro-F1 (not
  just the flattering number): TF-IDF 82.79% acc / 0.80 F1, IndoBERT
  89.66% acc / 0.8705 F1, no per-class regressions from the transformer.
- Streamlit inference app (`app.py`) deployed and live.
- README documents dataset source, reproduction steps, and limitations
  honestly (single time window overlapping COVID onset, 13.3x class
  imbalance, `news` as a catch-all overlapping category).
- Was already the strongest repo in the portfolio audit — no fixes
  needed.

## In progress
- Nothing currently active.

## Known issues / honest limitations
- Free-tier Streamlit hosting sleeps after inactivity — first visitor
  after a dormant period gets a ~20-30s cold start, not an error.
- IndoBERT weights aren't committed to the repo (documented in README;
  reproducible via the notebook, optionally pushed to HF Hub).
- Model trained on a single 6-month window (Jan-Jun 2020) from one
  outlet — vocabulary skewed toward COVID-era news, may not generalize
  to other periods/sources (documented in README, not hidden).

## Verification log
- 2026-07-23: git working tree clean, no pending diff.
  `/security-review` skill checked — N/A, diff-based and nothing to
  review. Live demo woken from sleep and confirmed booting correctly:
  all core widget JS assets (TextArea, Button, input handling) returned
  200; UI renders inside an iframe the browser tooling couldn't
  visually confirm past network-level verification, but no failed
  requests or console errors.

## Next up
- Nothing scheduled.
