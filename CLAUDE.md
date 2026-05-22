# AML Trading Anomaly Detection — Claude Code Notes

## What this project is

An end-to-end Python framework for AML surveillance of stock trading activity. It engineers behavioral features from trade-level data, trains unsupervised anomaly detection models (Isolation Forest + Local Outlier Factor (LOF)), and produces plain-English analyst reports.

See [README.md](README.md) for full project documentation and model explanations.
See [USING_YOUR_OWN_DATA.md](USING_YOUR_OWN_DATA.md) for the practitioner guide.

---

## Dual-audience design

This repo serves two distinct audiences. Always write code, docs, and explanations with both in mind.

**Audience 1 — Learners and evaluators**
Want to run the full end-to-end system with synthetic data. They use `data_gen/`, `run_all.py`, and all six notebooks. The synthetic data and injected anomalies exist entirely for their benefit — to make the system runnable and validatable without needing a real dataset.

**Audience 2 — Practitioners with real data**
Want to plug in their own accounts, securities, and transaction data. They skip `data_gen/` entirely and adapt the feature engineering modules for their own schema. The modeling and scoring pipeline is schema-agnostic — it works with any feature matrix.

**Key principle:** `data_gen/` is demo scaffolding. `features/`, `models/`, and `reporting/` are the production-ready components. When building features, keep them general enough that a practitioner can replace or extend them without touching the modeling layer.

---

## Unit of analysis and operational context

- **Unit of analysis is the account**, not the individual transaction. Transactions are aggregated into account-level behavioral features over rolling windows (7-day, 30-day).
- **Production cadence is nightly batch scoring** — all accounts that traded that day get re-scored against the already-fitted model. The model is retrained periodically (weekly or monthly), not on every scoring run.
- **New accounts** with fewer than 30 days of history will have sparse self-baseline features (`value_zscore_vs_self`, `velocity_ratio_7d_vs_30d`). These default to 0. New accounts are a known limitation — consider a rule-based overlay or lower threshold for accounts under 90 days old.

---

## Package structure

The importable package is `aml_anomaly` inside `src/`. Sub-packages map to pipeline stages:

- `data_gen/` — **demo only** — synthetic data generation (securities → accounts → trades, in that order)
- `features/` — feature engineering; `pipeline.py` is the single entry point that calls all feature modules
- `models/` — `train.py` fits and saves; `score.py` loads and scores — never refit on new data
- `reporting/` — `flags.py` owns the plain-English feature label dictionary used by both narratives and ranked tables

---

## Critical design constraints

- **Generate data in order: securities → accounts → trades.** Trades reference both accounts and securities.
- **Holdout split is random (not time-based).** Injected anomaly trades are concentrated in recent dates; a time-based split would push them all into holdout, leaving the training features with no anomalous behavior.
- **`train.py` saves; `score.py` loads — never the reverse.** The scoring pipeline must never refit the scaler, PCA, or models.
- **`injected_anomalies.csv` is validation-only.** No model or feature module should ever read it. It exists solely for notebook-level qualitative validation.
- **Population stats (`models/population_stats.json`) are written at train time.** The reporting module reads this to compute how far a flagged account deviates from the population mean per feature.
- **Plain-English feature labels live in one place: `reporting/flags.py`.** All narratives and ranked tables pull from this central dictionary — never duplicate label strings elsewhere.
- **LOF is always written as "Local Outlier Factor (LOF)" on first use in any file.** After first use in a given file, LOF alone is fine.

---

## File path conventions

- All file paths use `pathlib.Path` — no string concatenation with `os.path`
- Data outputs: `data/raw/`, `data/features/`, `data/holdout/`
- Model artifacts: `models/` (`.pkl` files + JSON config files)
- Analyst outputs: `outputs/`

---

## Random state

`random_state=42` throughout — all numpy, sklearn, and Faker generators. This ensures the synthetic dataset is reproducible.

---

## Dependencies and commands

```bash
uv sync --extra dev              # install all deps including dev tools
uv run python run_all.py         # full pipeline end-to-end (demo path)
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run mypy src/                 # type check
uv run pytest tests/unit/ -v     # unit tests

# Practitioner path — score your own data against fitted models
uv run python -m aml_anomaly.models.score \
    --trades path/to/trades.csv \
    --output outputs/scored_accounts.csv
```

---

## Notebooks (numbered in pipeline order)

Notebooks 01-02 are demo-specific (explore synthetic data). Notebooks 03-06 work with any feature matrix and score file.

1. `01_data_exploration.ipynb` — sanity checks on synthetic data *(demo only)*
2. `02_feature_engineering.ipynb` — feature distributions, correlations, null rates *(demo only)*
3. `03_pca_analysis.ipynb` — scree plot, variance explained, component loadings *(universal)*
4. `04_model_training.ipynb` — model score distributions, contamination tuning *(universal)*
5. `05_scoring_holdout.ipynb` — score holdout, confirm pipeline works end-to-end *(universal)*
6. `06_results_review.ipynb` — analyst-facing output: narratives, ranked features, trade detail *(universal)*

---

## What "done" looks like for each module

- `data_gen/`: running `accounts.py`, `securities.py`, `trades.py` writes CSVs to `data/raw/`
- `features/pipeline.py`: `build_feature_matrix(trades, accounts, securities)` returns one clean DataFrame — one row per account, all feature columns populated
- `models/train.py`: produces `.pkl` files + JSON config files in `models/`
- `models/score.py`: accepts a trades CSV path and output path via CLI; scores without touching `.pkl` files other than to load them
- `reporting/`: `flags.py` produces a narrative string and ranked DataFrame for a given account ID; `export.py` writes the output files
