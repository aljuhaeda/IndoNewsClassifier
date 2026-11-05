# IndoNewsClassifier — Progress

## Status
Deployed, stable. Live at https://indonewsclassifier.streamlit.app
(Streamlit Community Cloud free tier — sleeps after inactivity, wakes
on first visit within ~30s.)

## Done
- EDA, TF-IDF+LogReg baseline, fine-tuned IndoBERT, side-by-side
  comparison — all in `notebooks/indonewsclassifier.ipynb`.
- Real results reported honestly with both accuracy and macro-F1 (not
  just the flattering number): TF-IDF 82.82% acc / 0.8008 F1 (re-run
  from the repo's code 2026-11-03; the original notebook run recorded
  82.79% / 0.8000, likely a library-version effect), IndoBERT 89.66% acc
  / 0.8705 F1 and no per-class regressions from the transformer (reported
  from the notebook's saved output, not re-reproduced).
- Streamlit inference app (`app.py`) deployed and live.
- README documents dataset source, reproduction steps, and limitations
  honestly (single time window overlapping COVID onset, 13.3x class
  imbalance, `news` as a catch-all overlapping category).
- Fixed a real empty-input prediction bug (2026-07-28) — see
  Verification log (continued) below.

## In progress
- Nothing currently active.

## Known issues / honest limitations
- Free-tier Streamlit hosting sleeps after inactivity — first visitor
  after a dormant period gets a slow cold start, not an error. Slower
  than previously documented: a live re-check (2026-07-29) was still
  mid-boot past 35s, not the ~20-30s this used to say.
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
- `requirements.txt` pins are live (`b299a93`, Cloud runs Python 3.14). Checked live 2026-11-05: page renders and a test sports title returned `sport`, no exception. The `RuntimeError: Expected ASGI message` seen in the Cloud log on 2026-11-03 was not re-read and its cause stays unknown; the slow load that day did not recur.
- Nothing scheduled.

## Verification log (continued)
- 2026-07-28: fresh audit found a real logic bug — `predict_baseline`
  only checked the raw title for emptiness, but predicts on cleaned text
  (`clean_text` strips non-Latin/punctuation-only input down to an empty
  string). A garbage title silently produced a confident-looking fake
  prediction instead of an error. Fixed: `predict_baseline` now raises
  `ValueError` when cleaning empties the input, and the app shows a
  warning instead. Verified directly against the deployed model
  (empty-after-cleaning input now raises; a normal title still predicts
  correctly).
- 2026-08-11: reviewed commits since last check (README screenshot only)
  — no app/model code changed, status unaffected.
- 2026-11-03 (metrics re-reproduced from code on 2026-11-03, TF-IDF
  baseline only): ran notebook cells 4/5/10/12 as a scratch script using
  the repo's `src.preprocessing` / `src.training` unchanged. Repo `.venv`
  = Python 3.14.7, scikit-learn 1.9.0. Data:
  `data/raw/detik_news_title.csv`, 91,017 rows → 90,891 after dropping
  126 duplicate titles. Split: `split_dataset()`, stratified 70/15/15,
  `random_state=42`, giving 63,623 / 13,634 / 13,634 rows (identical to
  the on-disk `processed_*.csv`). Model:
  `TfidfVectorizer(max_features=20000, ngram_range=(1,2))` +
  `LogisticRegression(max_iter=1000, class_weight="balanced")` in one
  Pipeline fit on train only. Result: **test acc 0.8282, macro-F1
  0.8008** (val 0.8267 / 0.8022); 180,009 params. The committed
  `models/tfidf_logreg.joblib` (the deployed model) gives the same
  0.8282 / 0.8008 on this test split. **This does not exactly match the
  claimed 82.79% / 0.8000**, which is the notebook's saved output. The gap
  is +0.03pp acc and +0.0008 F1, and per-class F1 is also slightly off
  (health 0.691 vs claimed 0.692, travel 0.739 vs claimed 0.738). A
  scikit-learn version change since the notebook run is a likely cause,
  but that's a guess and wasn't confirmed.
  - **Leak check.** The vectorizer is inside the Pipeline and fit on train
    only, and cleaning is stateless, so that's clean. Model selection is on
    val, not test: IndoBERT early-stops on val loss and the baseline isn't
    tuned. Exact-title duplicates are dropped before the split, but 162/13,634
    test rows have cleaned text identical to a train row (3 identical
    after lowercasing only). Dropping those 162 gives acc 0.8269 /
    macro-F1 0.7983, so the effect is negligible. The split is random over a
    Jan–Jun 2020 dated dataset, so the numbers are within-period, not
    forward-in-time.
  - **Not re-reproduced:** the IndoBERT 89.66% / 0.8705 (GPU fine-tune not
    run; weights aren't in the repo). It rests only on the notebook's
    saved output, so the "+0.0705 macro-F1" delta and per-class gains
    are unverified too.
