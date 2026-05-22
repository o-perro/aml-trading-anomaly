# AML Trading Anomaly Detection — Claude Code Notes

## What this project is

An end-to-end Python framework for AML surveillance of stock trading activity. It generates synthetic trade data, engineers behavioral features, trains unsupervised anomaly detection models (Isolation Forest + Local Outlier Factor (LOF)), and produces plain-English analyst reports.

See [README.md](README.md) for full project documentation including model explanations.

## Package structure

The importable package is `aml_anomaly` inside `src/`. Four sub-packages map to the four pipeline stages:

- `data_gen/` — synthetic data generation (securities → accounts → trades, in that order)
- `features/` — feature engineering; `pipeline.py` is the single entry point that calls all feature modules
- `models/` — `train.py` fits and saves; `score.py` loads and scores — never refit on new data
- `reporting/` — `flags.py` owns the plain-English feature label dictionary used by both narratives and ranked tables

## Critical design constraints

- **Generate data in order: securities → accounts → trades.** Trades reference both accounts and securities.
- **Set aside the holdout split before any feature engineering or training.** Save to `data/holdout/new_trades.csv`.
- **`train.py` saves; `score.py` loads — never the reverse.** The scoring pipeline must never refit the scaler, PCA, or models.
- **`injected_anomalies.csv` is validation-only.** No model or feature module should read it. It exists solely for notebook-level qualitative validation.
- **Population stats (`models/population_stats.json`) are written at train time.** The reporting module reads this to compute how far a flagged account deviates from the population mean per feature.
- **Plain-English feature labels live in one place: `reporting/flags.py`.** All narratives and ranked tables pull from this central dictionary — never duplicate label strings elsewhere.

## File path conventions

- All file paths use `pathlib.Path` — no string concatenation with `os.path`
- Data outputs go to `data/raw/`, `data/features/`, `data/holdout/`
- Model artifacts go to `models/` (`.pkl` files + `population_stats.json`)
- Analyst outputs go to `outputs/`

## Random state

`random_state=42` throughout — all numpy, sklearn, and Faker generators. This ensures the synthetic dataset is reproducible.

## Dependencies and commands

```bash
uv sync --extra dev              # install all deps including dev tools
uv run python run_all.py         # full pipeline end-to-end
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run mypy src/                 # type check
uv run pytest tests/unit/ -v     # unit tests
```

## Notebooks (numbered in pipeline order)

1. `01_data_exploration.ipynb` — sanity checks on generated synthetic data
2. `02_feature_engineering.ipynb` — feature distributions, correlations, null rates
3. `03_pca_analysis.ipynb` — scree plot, variance explained, component loadings
4. `04_model_training.ipynb` — run training, tune contamination parameter
5. `05_scoring_holdout.ipynb` — score holdout without retraining; confirm pipeline works
6. `06_results_review.ipynb` — top flagged accounts, validate injected anomalies were caught

## What "done" looks like for each module

- `data_gen/`: running `accounts.py`, `securities.py`, `trades.py` writes CSVs to `data/raw/` with the schema from the PRD
- `features/pipeline.py`: calling `build_feature_matrix(trades, accounts, securities)` returns a single clean DataFrame with one row per account and all feature columns populated
- `models/train.py`: running it produces `.pkl` files in `models/` and `population_stats.json`
- `models/score.py`: accepts a trades CSV path and output path via CLI; scores without touching `.pkl` files other than to load them
- `reporting/`: `flags.py` produces a narrative string and a ranked DataFrame for a given account ID; `export.py` writes the two output files
