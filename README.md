# AML Stock Trading Anomaly Detection

An end-to-end Python framework for AML surveillance of stock trading activity. It generates realistic synthetic trade data, engineers behavioral features, trains two complementary anomaly detection models, and produces analyst-ready reports — including plain-English explanations for every flagged account.

---

## What this project does

Traditional AML surveillance relies on hard-coded rule thresholds ("flag any account with more than X trades per day"). Rules are easy to understand but easy to game, and they generate enormous volumes of false positives.

This framework takes a different approach: **unsupervised anomaly detection**. Instead of applying fixed rules, it learns what normal trading behavior looks like across the entire account population, then surfaces accounts that deviate meaningfully from that baseline. Every flag includes:

1. A plain-English narrative explaining *why* the account was flagged
2. A ranked table of the specific features that drove the score
3. The individual trades that contributed to those features

The system is designed for a human analyst to review, not to make autonomous decisions.

---

## Project philosophy

- **Simplicity over sophistication** — standard Python packages only, no GPU required
- **Interpretability** — every anomaly flag has a plain-English explanation; no black-box scores
- **Modularity** — data generation, feature engineering, modeling, and reporting are independent stages; any one can be swapped out
- **Speed to value** — a working pilot first, refinements second

The framework is **unsupervised**, meaning it does not require pre-labeled suspicious activity. The feature set is also fully compatible with supervised models if labeled data becomes available later.

---

## Repository structure

```
aml-trading-anomaly/
│
├── src/
│   └── aml_anomaly/
│       ├── data_gen/          # Synthetic data generation
│       │   ├── accounts.py    # 2,000 synthetic customer accounts
│       │   ├── securities.py  # 200 synthetic stocks/tickers
│       │   └── trades.py      # 100,000 synthetic trade transactions
│       │
│       ├── features/          # Feature engineering (one row per account)
│       │   ├── velocity.py    # Trade speed and frequency
│       │   ├── concentration.py  # Ticker and counterparty concentration
│       │   ├── behavioral.py  # Self-baseline deviation
│       │   ├── network.py     # Counterparty relationship features
│       │   └── pipeline.py    # Assembles all features into one DataFrame
│       │
│       ├── models/            # Anomaly detection models
│       │   ├── dimensionality.py    # Preprocessing and PCA
│       │   ├── isolation_forest.py  # Primary model
│       │   ├── lof.py               # Secondary model
│       │   ├── ensemble.py          # Combines both scores
│       │   ├── train.py             # Fits and saves all model objects
│       │   └── score.py             # Scores new data against fitted models
│       │
│       └── reporting/         # Analyst output
│           ├── flags.py       # Plain-English narratives + ranked feature tables
│           └── export.py      # Writes CSV and Excel outputs
│
├── notebooks/                 # Exploratory and results notebooks (numbered in order)
├── data/
│   ├── raw/                   # Generated synthetic CSVs (gitignored)
│   ├── features/              # Engineered feature tables (gitignored)
│   └── holdout/               # 20% trade split used for scoring validation (gitignored)
├── models/                    # Saved model artifacts: .pkl files (gitignored)
├── outputs/                   # Anomaly scores and analyst Excel reports (gitignored)
├── tests/                     # Unit tests
├── run_all.py                 # Runs the full pipeline end-to-end
└── pyproject.toml             # Dependencies and project metadata
```

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management — it is significantly faster than `pip` and produces more reproducible installs.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo and install dependencies
git clone https://github.com/o-perro/aml-trading-anomaly.git
cd aml-trading-anomaly
uv sync --extra dev
```

---

## Running the full pipeline

```bash
uv run python run_all.py
```

This executes every stage in order:
1. Generate synthetic data (accounts → securities → trades)
2. Set aside 20% of trades as a holdout set (never seen during training)
3. Engineer features for the training population
4. Fit the StandardScaler, PCA, Isolation Forest, and Local Outlier Factor (LOF) — save all to `models/`
5. Score the holdout trades against the fitted models (no retraining)
6. Export results to `outputs/`

Alternatively, run stages individually:

```bash
uv run python -m aml_anomaly.models.train     # train only
uv run python -m aml_anomaly.models.score --trades data/holdout/new_trades.csv --output outputs/scored_holdout.xlsx
```

---

## How realistic trade simulation works

Generating synthetic data that looks like real brokerage activity requires matching the statistical distributions of real trades:

**Quantities and values** use log-normal distributions. Real trade sizes aren't uniformly random — most are modest but a small number are very large. The log-normal distribution naturally produces that long right tail.

**Prices** are generated near each security's `avg_price_usd` with a small random noise term, reflecting how real executions work: you trade near the market price, not at an arbitrary number.

**Trade timing** is weighted heavily toward market hours (9:30am–4:00pm ET, weekdays). Roughly 5% of trades are allowed to fall outside those hours, because some legitimate after-hours trading does occur. This makes the `off_hours_trade_pct` feature meaningful — unusual but not impossible.

**Account behavior** varies by account type and a randomly assigned activity level. Some accounts trade daily, others monthly. Each account's trades are generated at a rate consistent with that level, which is what gives the self-baseline features (like `value_zscore_vs_self`) their statistical power.

**Referential integrity** is maintained throughout: trades always reference valid account IDs and ticker symbols, generated in the correct order (securities first, then accounts, then trades).

---

## How anomaly injection works

Approximately 3–5% of accounts (60–100 out of 2,000) have known suspicious patterns deliberately injected into their trading behavior. These accounts are recorded in `data/raw/injected_anomalies.csv` — a reference file for validation that is **never read by any model or feature module**.

Each injected pattern is designed to produce extreme values on specific features, which is what the anomaly models detect:

| Pattern | What is injected | Features that become extreme |
|---|---|---|
| **Wash trading** | Account buys and sells the same ticker repeatedly with one counterparty in rapid succession | `same_day_reversal_count` spikes; `buy_sell_ratio_30d` → ~1.0; `top_counterparty_concentration_pct` → 90%+ |
| **Velocity spike** | Normal account suddenly does 10–20× its usual trade count over 7 days | `velocity_ratio_7d_vs_30d` → 10–20; `value_zscore_vs_self` → 5–8σ above own average |
| **Smurfing** | 10–20 trades per day all just below $10,000 (deliberate structuring to avoid reporting thresholds) | `round_value_trade_pct` → 80%+; `burst_event_count` spikes; `trades_per_day_30d` elevated |
| **Illiquid concentration** | More than 50% of monthly trade volume directed into a single micro-cap stock | `illiquid_stock_trade_pct` → 90%+; `top_ticker_concentration_pct` → extreme; `trade_size_vs_adv_max` → extreme |
| **Off-hours clustering** | 70–80% of trades placed outside normal market hours | `off_hours_trade_pct` → 0.7–0.8 vs population average of ~0.05 |

In the results notebook (`06_results_review.ipynb`), we validate the model by checking: of the injected anomaly accounts, what fraction landed in the top 4% anomaly score list?

---

## The two anomaly detection models

### Isolation Forest (primary model, 60% weight)

Isolation Forest works by randomly partitioning the feature space. The key insight: **outliers are easy to isolate**. A normal account sits in a dense region of the feature space — it takes many random splits to separate it from its neighbors. An anomalous account sits alone in a sparse region — it gets isolated in very few splits.

Each account is run through 200 decision trees. The average number of splits needed to isolate that account across all trees becomes its anomaly score. Accounts that are isolated quickly (low score) are flagged as anomalous.

Parameters used:
- `n_estimators = 200` — number of trees
- `contamination = 0.04` — tells the model to expect ~4% anomalies
- `random_state = 42` — ensures reproducibility

### Local Outlier Factor (secondary model, 40% weight)

LOF asks a different question: **is this account in a sparser region of the feature space than its neighbors?**

It computes a local density for each account — roughly, how tightly packed are the accounts around it in feature space. Then it compares that density to the densities of its nearest neighbors. If your neighbors are all in dense clusters but you're floating in a sparse region, your LOF score is high.

This makes LOF especially powerful for detecting **local outliers** — accounts that are unusual *relative to their peer group* but not necessarily unusual globally. For example, a retail investor who suddenly trades 20 times per week might not look extreme next to institutional traders, but LOF compares them to the retail cluster they actually belong to.

Parameters used:
- `n_neighbors = 20` — the 20 nearest accounts are used to compute local density
- `contamination = 0.04` — matches the Isolation Forest setting

### Why use both?

The two models have complementary blind spots:

| | Isolation Forest | LOF |
|---|---|---|
| Good at | Global outliers — accounts unusual overall | Local outliers — accounts unusual *for their peer group* |
| Weakness | May miss outliers that cluster together | Struggles with very high-dimensional data |

Combining both (60% IF + 40% LOF weight) means an account must look anomalous from multiple angles to rank at the top of the list. This reduces false positives.

### Ensemble score

```
anomaly_score = (normalized_IF_score × 0.6) + (normalized_LOF_score × 0.4)
```

Both raw scores are first normalized to [0, 1] before combining. The final output for each account:
- `anomaly_score` — continuous composite score (higher = more anomalous)
- `anomaly_rank` — rank from most to least anomalous across all accounts
- `anomaly_flag` — binary: 1 if account is in the top 4% by anomaly score

---

## Dimensionality reduction with PCA

Before running either model, all features are passed through **Principal Component Analysis (PCA)**.

PCA is a technique that compresses many correlated features into a smaller set of uncorrelated dimensions (called principal components) while retaining most of the information. We keep enough components to explain 95% of the cumulative variance in the data.

Why this matters:
- Many of our features are correlated (e.g. `trades_per_day_30d` and `trade_value_per_day_30d` tend to move together). PCA removes that redundancy.
- Isolation Forest and LOF both perform better in lower-dimensional spaces — with 30+ features, the "curse of dimensionality" can dilute distance measures. PCA keeps the geometry meaningful.

The PCA object is fitted on the training data and saved to `models/pca.pkl`. When scoring new data, the same fitted PCA is applied — we never refit it on new data.

---

## Train vs. score split

This distinction is critical for production-representative behavior.

**Training mode** (`train.py`) — run once on the full synthetic dataset. Fits all model objects (scaler, PCA, Isolation Forest, LOF) and saves them to disk. Also saves population feature statistics (mean and standard deviation per feature) to `models/population_stats.json` — used by the reporting module to compute how far each flagged account deviates from the population.

**Scoring mode** (`score.py`) — loads all fitted objects and scores a new batch of accounts *without retraining*. This is the production flow: new trades come in, get featurized using the same pipeline, and are scored against the already-fitted model. The model's definition of "normal" never changes based on incoming data.

The holdout dataset (20% of trades, set aside before training) is used to test the scoring pipeline end-to-end.

---

## Analyst output

Every flagged account produces three layers of output:

**Layer 1 — Plain-English narrative.** A 2–4 sentence description of why the account was flagged, written without variable names or statistical jargon. Example:

> *"This account's trading activity surged to 12 times its normal pace over the past 7 days, with nearly all of that activity concentrated in two thinly-traded stocks. The account also executed 6 same-day buy/sell reversals — a pattern not seen in its prior history."*

**Layer 2 — Ranked feature contributions.** A table of the top 5 features driving the score, showing account value vs. population average vs. deviation in standard deviations (σ).

**Layer 3 — Supporting trades.** The 10–20 individual transactions that drove the anomalous features, with a `flag_reason` column linking each trade to the pattern it contributed to.

All three layers are exported to a multi-tab Excel workbook (`outputs/flagged_accounts.xlsx`) designed for AML analysts.

---

## Feature reference

### Velocity & frequency

| Feature | What it measures |
|---|---|
| `trades_per_day_30d` | Average trades per active day over 30 days |
| `trades_per_day_7d` | Average trades per active day over 7 days |
| `velocity_ratio_7d_vs_30d` | 7-day trade pace ÷ 30-day trade pace — detects sudden spikes |
| `max_trades_in_1hr` | Most trades placed in any single hour |
| `burst_event_count` | Times the account placed 5+ trades within a 30-minute window |
| `avg_time_between_trades_sec` | Average seconds between consecutive trades |
| `min_time_between_trades_sec` | Shortest gap between any two trades |
| `trade_value_per_day_30d` | Average daily dollar value traded over 30 days |
| `trade_value_per_day_7d` | Average daily dollar value traded over 7 days |

### Concentration

| Feature | What it measures |
|---|---|
| `top_ticker_concentration_pct` | % of total trade value in the single most-traded stock |
| `top_3_ticker_concentration_pct` | % of total trade value in the top 3 stocks |
| `illiquid_stock_trade_pct` | % of trade value in thinly-traded stocks |
| `trade_size_vs_adv_max` | Largest single trade as % of that stock's average daily volume |
| `trade_size_vs_adv_avg` | Average trade size as % of average daily volume |
| `micro_cap_trade_pct` | % of trade value in micro-cap stocks |
| `buy_sell_ratio_30d` | Buy value ÷ sell value over 30 days (near 1.0 may indicate wash trading) |
| `round_value_trade_pct` | % of trades with suspiciously round dollar values |

### Behavioral self-baseline

| Feature | What it measures |
|---|---|
| `value_zscore_vs_self` | This month's trade value vs. account's own historical average (in σ) |
| `velocity_zscore_vs_self` | This month's trade frequency vs. account's own historical average (in σ) |
| `new_ticker_count_30d` | Stocks traded for the first time in the past 30 days |
| `new_ticker_pct_30d` | % of recent trades involving stocks never traded before |
| `off_hours_trade_pct` | % of trades placed outside 9:30am–4:00pm ET |
| `avg_holding_period_minutes` | Average time between a buy and next sell of the same ticker |
| `min_holding_period_minutes` | Shortest buy-to-sell turnaround for any ticker |
| `weekend_trade_pct` | % of trades placed on weekends |

### Network & counterparty

| Feature | What it measures |
|---|---|
| `unique_counterparties_30d` | Distinct accounts traded against in the past 30 days |
| `top_counterparty_concentration_pct` | % of trades directed at the single most frequent counterparty |
| `new_counterparty_count_30d` | New counterparties traded with for the first time this month |
| `same_day_reversal_count` | Times the account bought and sold the same stock on the same day |
| `circular_trade_flag` | Whether the account appears in a circular trading chain (A→B→C→A) |
| `shared_counterparty_ticker_count` | Tickers where the account repeatedly trades with the same counterparty |

---

## Development

```bash
# Run all quality checks (required before every commit)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/

# Run tests
uv run pytest tests/unit/ -v
uv run pytest --cov=src --cov-report=term-missing
```

---

## License

MIT
