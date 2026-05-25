# Using This Framework With Your Own Data

This guide is for practitioners who have real brokerage data and want to deploy the anomaly detection framework against it. If you want to understand the full system first using synthetic data, see the [main README](README.md).

---

## What to skip

The `src/aml_anomaly/data_gen/` package and `run_all.py` exist solely to generate synthetic data for demonstration and validation purposes. You do not need them.

**Skip entirely:**
- `src/aml_anomaly/data_gen/` — synthetic account, security, and trade generation
- `run_all.py` — the end-to-end demo pipeline
- Notebooks `01_data_exploration.ipynb` and `02_feature_engineering.ipynb` as written — replace the data loading cells with your own data sources

Everything else — `features/`, `models/`, `reporting/`, and notebooks `03` through `06` — is designed to work with any data that goes through the feature engineering step.

---

## What the framework actually needs

The modeling layer is completely agnostic to your raw data schema. The only contract it requires is a **feature matrix**: a DataFrame with one row per account and numeric feature columns. It doesn't care what those features are named or how they were computed.

The feature engineering layer is where your data schema matters — but it is a **template to adapt**, not a rigid requirement. The four feature modules (`velocity.py`, `concentration.py`, `behavioral.py`, `network.py`) are examples of the kinds of behavioral signals that matter for AML. You will likely want to replace, extend, or modify them based on what data you have and what patterns are relevant to your institution.

---

## Recommended approach

### Step 1 — Understand the feature template

Read through the four feature modules and the feature reference in the README. Understand what each feature captures and why. Then map your own data against them:

- Which features can you compute directly from your transaction data?
- Which features require data you don't have (e.g., counterparty account IDs may not be available for market orders)?
- What additional features does your data support that aren't in the template (e.g., product type, channel, geographic data, customer relationship data)?

The goal is not to reproduce our exact 35 features — it is to engineer a set of behavioral features that capture the patterns you care about detecting.

### Step 2 — Replace or extend the feature modules

Each feature module is an independent Python file that takes DataFrames as input and returns a DataFrame with one row per account. You can:

- **Replace a module entirely** — write your own `velocity.py` using your transaction schema
- **Extend a module** — add additional features to an existing module
- **Add a new module** — create a new feature file and add it to `pipeline.py`

The `pipeline.py` entry point calls each module and merges the results. Adding your own module is as simple as importing it and calling it in `build_feature_matrix()`.

### Step 3 — Prepare your data

Your data needs to be loadable as pandas DataFrames before being passed to `build_feature_matrix()`. The function signature is:

```python
from aml_anomaly.features.pipeline import build_feature_matrix

features = build_feature_matrix(
    trades=trades_df,       # your transaction-level data
    accounts=accounts_df,   # your account/customer-level data
    securities=securities_df  # your instrument/security-level data
)
```

The column names your feature modules reference are up to you — they're defined inside the feature modules themselves, not in any central schema file. If you rename a column in your data, update the corresponding reference in the feature module.

### Step 4 — Train the models

Once you have a feature matrix, training is a single function call:

```python
from aml_anomaly.models.train import train

scores = train(
    trades=trades_df,
    accounts=accounts_df,
    securities=securities_df,
    models_dir=Path("models/"),
    contamination=0.04,  # expected anomaly rate — adjust for your population
)
```

This fits the scaler, PCA, Isolation Forest, and Local Outlier Factor (LOF), and saves all fitted objects to `models/`. The `contamination` parameter controls what fraction of accounts get flagged — 0.04 means the top 4% by anomaly score. Adjust this based on your AML team's review capacity and your institution's risk appetite.

### Step 5 — Score new data on a schedule

For production, run scoring on a schedule (nightly is typical). The correct flow is:

1. Identify all accounts with at least one transaction today
2. For each in-scope account, pull the last **6 months (180 days)** of transaction history
3. Pass that history through the feature engineering pipeline and score

```python
from aml_anomaly.models.score import score_trades

scores = score_trades(
    trades=last_180_days_df,  # 6-month lookback per account — not a random sample
    accounts=accounts_df,
    securities=securities_df,
    models_dir=Path("models/"),
)
```

Or via CLI:

```bash
uv run python -m aml_anomaly.models.score \
    --trades data/new_trades.csv \
    --output outputs/scored_accounts.csv
```

**Why 6 months?** The feature engineering uses 30-day rolling windows, but self-baseline features (`value_zscore_vs_self`, `velocity_zscore_vs_self`) compare recent activity against everything older than 30 days. 30 days is the hard minimum for features to be non-null; 6 months gives those self-baseline features a meaningful historical reference and ensures low-frequency accounts have enough history. Always pull a **time-contiguous** window — a random sample of transactions destroys temporal signals like `velocity_ratio_7d_vs_30d`.

The scoring pipeline loads the fitted model objects and applies them to the new data without retraining. The model's definition of "normal" stays constant between scoring runs. Retrain the model periodically (weekly or monthly) as your account population evolves.

---

## The contamination parameter

`contamination=0.04` means "flag the top 4% of accounts by anomaly score." This is not a statistical threshold — it is a capacity setting. Set it based on how many cases your AML team can realistically review in a given period.

At 50,000 accounts with `contamination=0.04`, you would generate 2,000 flagged accounts per run. If your team can review 50 cases per day, either lower the threshold or implement a tiered review process (automated triage for lower scores, manual review for top scores).

---

## Adapting the notebooks

Notebooks `03` through `06` are designed to work with any feature matrix and score file, not just the synthetic data. Replace the data loading cells at the top of each notebook with your own data sources and they will work as-is.

The notebooks are organized around four questions:
- **Notebook 03** — does PCA compress your features sensibly? Do anomalous accounts visually separate in PCA space?
- **Notebook 04** — what do the raw model score distributions look like? Is the contamination parameter right for your data?
- **Notebook 05** — does the scoring pipeline work correctly on new data?
- **Notebook 06** — what do the top-flagged accounts look like? Do the narratives make sense to an analyst?

If you have a labeled dataset (known SAR filings, confirmed cases, prior investigations), you can use it in notebooks 04 and 06 as a ground-truth validation — the same way we use the injected anomaly accounts in the demo.

**Notebook 06 is a developer and validation tool**, not a production analyst interface. In a real deployment, the scored output feeds into a purpose-built case management system — a commercial AML platform, an internal web application, or a BI tool like Tableau or Power BI. The Excel export (`outputs/flagged_accounts.xlsx`) is the pragmatic bridge for teams without a downstream system yet in place.

---

## Operational considerations

### Account-level scoring

This framework scores accounts, not individual transactions. Transactions are the raw material — they get aggregated into behavioral features that describe the account's pattern of activity over rolling windows (7-day, 30-day). The account is the unit the AML analyst reviews.

### New accounts

Accounts with limited history will have sparse or zero values on features that require a historical baseline (`value_zscore_vs_self`, `velocity_ratio_7d_vs_30d`). These features default to zero when history is insufficient — meaning new accounts may be under-scored on those dimensions.

Recommended mitigations:
- Add a separate low-threshold scoring tier for accounts under 90 days old
- Apply rule-based overlays for brand new accounts (any large trade, any off-hours trade, any micro-cap concentration) in addition to the model score
- Consider `account_age_days` as a feature — the model will learn to weight recent activity in young accounts differently

### Retraining cadence

Retrain the model when the account population changes significantly — new customer segments, regulatory changes, seasonal patterns. For a stable brokerage population, monthly retraining is typically sufficient. The `train.py` module handles retraining cleanly; all fitted objects are overwritten in `models/` and scoring automatically picks up the new versions.

---

## Known limitations

- **No supervised signal** — the model learns what normal looks like but has no ground truth for what suspicious looks like. Precision improves significantly if you have labeled historical cases to validate against.
- **New account handling** — see above. Self-baseline features are uninformative for accounts with fewer than 30 days of history.
- **Feature engineering is a template** — the 35 features in this repo reflect one interpretation of AML-relevant behavior in equity trading. Your institution's risk typologies, product mix, and customer base will require a different or extended feature set.
- **Counterparty data** — several network features (`circular_trade_flag`, `top_counterparty_concentration_pct`) require counterparty account IDs on transactions. Market orders typically don't carry this data. These features will be zero for institutions without counterparty-level transaction data.
