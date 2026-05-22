# AML Stock Trading Anomaly Detection

### Project Specification & Build Guide

> **Purpose:** End-to-end specification for building a modular Python anomaly detection framework for AML surveillance of stock trading activity. This document is intended to guide implementation — including synthetic data generation, feature engineering, modeling, and reporting.

-----

## 1. Project Philosophy

This project is a working pilot, not a research paper. The goal is a functional, interpretable anomaly detection system that the AML team can use, review, and iterate on. The approach favors:

- **Simplicity over sophistication** — standard Python packages only, no GPU requirements
- **Interpretability** — every anomaly flag has a plain-English explanation
- **Modularity** — each component (data, features, model, reporting) is independent and replaceable
- **Speed to value** — a working pilot first, refinements second

The framework is **unsupervised**: it does not require pre-labeled suspicious activity. It learns what normal looks like and surfaces statistical outliers for human review. The feature set is also fully compatible with supervised models if labeled data becomes available later.

-----

## 2. Repository Structure

```
aml-trading-anomaly/
│
├── README.md                          # Project overview, setup instructions, data dictionary
├── requirements.txt                   # All Python dependencies
├── .gitignore
│
├── src/
│   └── aml_anomaly/
│       ├── __init__.py
│       ├── data_gen/
│       │   ├── __init__.py
│       │   ├── accounts.py            # Synthetic customer account generation
│       │   ├── securities.py          # Synthetic stock/ticker generation
│       │   └── trades.py              # Synthetic trade transaction generation
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── velocity.py            # Trade speed and frequency features
│       │   ├── concentration.py       # Ticker and counterparty concentration features
│       │   ├── behavioral.py          # Self-baseline deviation features
│       │   ├── network.py             # Counterparty relationship features
│       │   └── pipeline.py            # Assembles all features into one modeling DataFrame
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── dimensionality.py      # PCA reduction
│       │   ├── isolation_forest.py    # Primary anomaly detection model
│       │   ├── lof.py                 # Local Outlier Factor (secondary model)
│       │   └── ensemble.py            # Combines model scores into a final anomaly score
│       │
│       └── reporting/
│           ├── __init__.py
│           ├── flags.py               # Generates plain-English flag descriptions per account
│           └── export.py              # Exports results to CSV and Excel
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Exploratory data analysis on synthetic data
│   ├── 02_feature_engineering.ipynb   # Feature distributions, correlations, null checks
│   ├── 03_pca_analysis.ipynb          # Variance explained, component loadings
│   ├── 04_model_training.ipynb        # Model runs, contamination parameter tuning
│   └── 05_results_review.ipynb        # Anomaly scores, flagged accounts, visualizations
│
├── data/
│   ├── raw/                           # Generated synthetic CSVs (accounts, securities, trades)
│   └── features/                      # Engineered feature tables ready for modeling
│
└── outputs/
    ├── anomaly_scores.csv             # All accounts with anomaly scores
    └── flagged_accounts.xlsx          # Analyst-ready report of top flagged accounts
```

-----

## 3. Dependencies

```text
pandas
numpy
scikit-learn
faker
scipy
matplotlib
seaborn
openpyxl
jupyter
```

All installable via:

```bash
pip install -r requirements.txt
```

-----

## 4. Synthetic Data Generation

### Design Principles

- Data must look realistic: plausible names, ticker symbols, price ranges, and trading volumes
- Inject approximately 3–5% anomalous accounts with known behavioral patterns for qualitative validation
- Maintain referential integrity: all trades reference valid account IDs and ticker symbols
- Use `Faker` for names and addresses, `numpy` random distributions for numeric fields, `pandas` for assembly
- Seed all random generators for reproducibility (`random_state=42`)

-----

### 4.1 Accounts Table

**Target:** ~2,000 rows | **File:** `src/aml_anomaly/data_gen/accounts.py` | **Output:** `data/raw/accounts.csv`

|Column                     |Type       |Plain-English Description                                 |Generation Notes                                |
|---------------------------|-----------|----------------------------------------------------------|------------------------------------------------|
|`account_id`               |string     |Unique account identifier                                 |Format: `ACC-00001` through `ACC-02000`         |
|`customer_name`            |string     |Full name of the account holder                           |Faker `name()`                                  |
|`account_type`             |categorical|Whether account is retail, institutional, or broker-dealer|70% retail, 20% institutional, 10% broker-dealer|
|`account_open_date`        |date       |Date the account was opened                               |Random date 1–15 years ago                      |
|`account_age_days`         |int        |Number of days since the account was opened               |Derived from open date                          |
|`country_of_origin`        |string     |Country of the account holder                             |80% US, 20% distributed across other countries  |
|`state`                    |string     |US state (domestic accounts only)                         |Random US state                                 |
|`risk_tier`                |int (1–5)  |KYC-assigned risk rating at account opening               |Right-skewed: most accounts are tier 1 or 2     |
|`is_pep`                   |bool       |Whether the customer is a Politically Exposed Person      |~2% true                                        |
|`is_high_risk_jurisdiction`|bool       |Whether the country of origin is on a financial watchlist |Derived from country field                      |
|`annual_income_usd`        |float      |Stated annual income at account opening                   |Log-normal distribution, range $30K–$5M         |
|`net_worth_usd`            |float      |Stated net worth at account opening                       |Correlated with income                          |

-----

### 4.2 Securities Table

**Target:** ~200 rows | **File:** `src/aml_anomaly/data_gen/securities.py` | **Output:** `data/raw/securities.csv`

|Column                |Type       |Plain-English Description                            |Generation Notes                                     |
|----------------------|-----------|-----------------------------------------------------|-----------------------------------------------------|
|`ticker`              |string     |Stock ticker symbol                                  |3–4 uppercase letters, e.g. `AXLM`, `BNVT`           |
|`company_name`        |string     |Fictitious company name                              |Faker `company()`                                    |
|`sector`              |categorical|Industry sector                                      |11 GICS sectors, evenly distributed                  |
|`market_cap_tier`     |categorical|Size classification of the company                   |30% large, 30% mid, 25% small, 15% micro             |
|`avg_daily_volume`    |int        |Typical number of shares traded per day              |Log-normal; micro-cap much lower                     |
|`avg_price_usd`       |float      |Typical stock price                                  |Log-normal, range $1–$500                            |
|`bid_ask_spread_pct`  |float      |Difference between buy and sell price as a percentage|Inversely correlated with volume; wider for micro-cap|
|`price_volatility_30d`|float      |How much the price typically moves over 30 days      |Higher for small/micro-cap                           |
|`exchange`            |categorical|Exchange where the stock is listed                   |NYSE, NASDAQ, OTC                                    |
|`is_illiquid`         |bool       |Flag for thinly-traded stocks                        |True when avg_daily_volume is in bottom 20%          |

-----

### 4.3 Trades Table

**Target:** ~100,000 rows | **File:** `src/aml_anomaly/data_gen/trades.py` | **Output:** `data/raw/trades.csv`

|Column                   |Type       |Plain-English Description                                          |Generation Notes                                     |
|-------------------------|-----------|-------------------------------------------------------------------|-----------------------------------------------------|
|`trade_id`               |string     |Unique trade identifier                                            |Format: `TRD-0000001`                                |
|`account_id`             |string     |Account that placed the trade                                      |Foreign key to accounts table                        |
|`counterparty_account_id`|string     |Account on the other side of the trade                             |Foreign key to accounts table; null for market orders|
|`ticker`                 |string     |Stock that was traded                                              |Foreign key to securities table                      |
|`trade_date`             |date       |Date the trade was executed                                        |Random across past 2 years                           |
|`trade_time`             |time       |Time of day the trade was executed                                 |Weighted toward market hours; small % off-hours      |
|`trade_direction`        |categorical|Whether the account was buying or selling                          |`BUY` or `SELL`                                      |
|`quantity`               |int        |Number of shares traded                                            |Log-normal; capped relative to avg_daily_volume      |
|`price_usd`              |float      |Price per share at execution                                       |Near avg_price_usd with small random noise           |
|`trade_value_usd`        |float      |Total dollar value of the trade (quantity × price)                 |Derived                                              |
|`order_type`             |categorical|How the order was placed                                           |`MARKET`, `LIMIT`, `STOP`                            |
|`time_to_execution_ms`   |int        |Milliseconds from order placement to fill                          |Log-normal; algorithmic trades cluster near zero     |
|`is_off_hours`           |bool       |Whether the trade occurred outside normal market hours             |Derived from trade_time                              |
|`is_round_value`         |bool       |Whether the trade value is suspiciously round (e.g. exactly $9,500)|Derived; used as smurfing signal                     |

#### Anomaly Injection (~3–5% of accounts)

Inject known behavioral patterns into a labeled subset for validation. These accounts are flagged in a separate `injected_anomalies.csv` reference file (not used by the model):

|Pattern               |Description                                                                                 |
|----------------------|--------------------------------------------------------------------------------------------|
|Wash trading          |Account buys and sells the same ticker in rapid succession, often with a linked counterparty|
|Velocity spike        |Account suddenly trades 10–20× its historical average over a short window                   |
|Smurfing              |Multiple trades just below $10,000 in the same day across the same or similar tickers       |
|Illiquid concentration|Account directs >50% of monthly volume into a single micro-cap stock                        |
|Off-hours clustering  |Majority of trades placed outside normal market hours                                       |

-----

## 5. Feature Engineering

All features are computed at the **account level** — one row per account — and assembled into a single modeling DataFrame by `src/aml_anomaly/features/pipeline.py`.

Rolling windows used throughout: **1-day, 7-day, 30-day**.

-----

### 5.1 Velocity & Frequency Features

**File:** `src/aml_anomaly/features/velocity.py`

|Feature Name                 |Plain-English Description                                                    |
|-----------------------------|-----------------------------------------------------------------------------|
|`trades_per_day_30d`         |Average number of trades per active day over the past 30 days                |
|`trades_per_day_7d`          |Average number of trades per active day over the past 7 days                 |
|`max_trades_in_1hr`          |The highest number of trades placed within any single hour                   |
|`avg_time_between_trades_sec`|Average number of seconds between consecutive trades                         |
|`min_time_between_trades_sec`|Shortest gap between any two consecutive trades (catches rapid-fire bursts)  |
|`trade_value_per_day_30d`    |Average total dollar value traded per day over 30 days                       |
|`trade_value_per_day_7d`     |Average total dollar value traded per day over 7 days                        |
|`burst_event_count`          |Number of times the account placed 5 or more trades within a 30-minute window|
|`velocity_ratio_7d_vs_30d`   |Ratio of 7-day trade pace to 30-day trade pace (detects sudden spikes)       |

-----

### 5.2 Concentration Features

**File:** `src/aml_anomaly/features/concentration.py`

|Feature Name                    |Plain-English Description                                                         |
|--------------------------------|----------------------------------------------------------------------------------|
|`top_ticker_concentration_pct`  |Percentage of total trade value directed at the single most-traded ticker         |
|`top_3_ticker_concentration_pct`|Percentage of total trade value in the top 3 tickers                              |
|`illiquid_stock_trade_pct`      |Percentage of trade value in stocks flagged as illiquid                           |
|`trade_size_vs_adv_max`         |Largest single trade as a percentage of that stock’s average daily volume         |
|`trade_size_vs_adv_avg`         |Average trade size as a percentage of the stock’s average daily volume            |
|`micro_cap_trade_pct`           |Percentage of total trade value in micro-cap stocks                               |
|`buy_sell_ratio_30d`            |Ratio of buy value to sell value over 30 days (near 1.0 may indicate wash trading)|
|`round_value_trade_pct`         |Percentage of trades where the dollar value is suspiciously round                 |

-----

### 5.3 Behavioral Self-Baseline Features

**File:** `src/aml_anomaly/features/behavioral.py`

These features compare each account to its **own** historical behavior, making them especially powerful for unsupervised detection.

|Feature Name                |Plain-English Description                                                                             |
|----------------------------|------------------------------------------------------------------------------------------------------|
|`value_zscore_vs_self`      |How far this month’s trade value is from the account’s own historical average (in standard deviations)|
|`velocity_zscore_vs_self`   |How far this month’s trade frequency is from the account’s own historical average                     |
|`new_ticker_count_30d`      |Number of stocks the account traded for the first time in the past 30 days                            |
|`new_ticker_pct_30d`        |Percentage of recent trades involving stocks the account has never traded before                      |
|`off_hours_trade_pct`       |Percentage of trades placed outside normal market hours (9:30am–4:00pm ET)                            |
|`avg_holding_period_minutes`|Average time between a buy and the next sell of the same ticker                                       |
|`min_holding_period_minutes`|Shortest buy-to-sell turnaround for any ticker (very short = in-and-out trading)                      |
|`weekend_trade_pct`         |Percentage of trades placed on weekends                                                               |

-----

### 5.4 Network & Counterparty Features

**File:** `src/aml_anomaly/features/network.py`

|Feature Name                        |Plain-English Description                                                        |
|------------------------------------|---------------------------------------------------------------------------------|
|`unique_counterparties_30d`         |Number of distinct accounts this account traded against in the past 30 days      |
|`top_counterparty_concentration_pct`|Percentage of trades directed at the single most frequent counterparty           |
|`new_counterparty_count_30d`        |Number of counterparties this account traded with for the first time this month  |
|`same_day_reversal_count`           |Number of times the account bought and sold the same ticker on the same day      |
|`circular_trade_flag`               |Whether the account appears in a circular trading chain (A → B → C → A pattern)  |
|`shared_counterparty_ticker_count`  |Number of tickers where this account repeatedly trades with the same counterparty|

-----

### 5.5 Account Profile Features

**File:** Used as context in `pipeline.py` — included in feature matrix as encoded categoricals

|Feature Name               |Plain-English Description                                                |
|---------------------------|-------------------------------------------------------------------------|
|`account_age_days`         |How long the account has been open                                       |
|`risk_tier`                |KYC risk rating assigned at account opening                              |
|`is_pep`                   |Whether the account holder is a Politically Exposed Person               |
|`is_high_risk_jurisdiction`|Whether the account holder’s country is on a financial watchlist         |
|`account_type_encoded`     |Numeric encoding of account type (retail / institutional / broker-dealer)|

-----

## 6. Modeling Pipeline

### 6.1 Preprocessing

**File:** `src/aml_anomaly/models/dimensionality.py`

1. Drop any features with >20% null values
1. Impute remaining nulls with column median
1. Standard scale all features (`StandardScaler`) — required for PCA and LOF
1. Apply **PCA** to reduce dimensionality
- Retain components explaining **95% of cumulative variance**
- Log the number of components selected and their top feature loadings
- PCA output is the input to both models

-----

### 6.2 Primary Model — Isolation Forest

**File:** `src/aml_anomaly/models/isolation_forest.py`

Isolation Forest works by randomly partitioning the feature space. Anomalies are isolated in fewer splits than normal observations, resulting in a lower anomaly score.

|Parameter      |Value |Notes                                   |
|---------------|------|----------------------------------------|
|`n_estimators` |200   |Number of trees                         |
|`contamination`|0.04  |Assumed ~4% anomaly rate; tune as needed|
|`max_samples`  |`auto`|Uses min(256, n_samples)                |
|`random_state` |42    |For reproducibility                     |

**Output:** `isolation_forest_score` (continuous, lower = more anomalous) and `isolation_forest_flag` (1 = anomaly, 0 = normal)

-----

### 6.3 Secondary Model — Local Outlier Factor

**File:** `src/aml_anomaly/models/lof.py`

LOF measures how isolated an account is relative to its neighbors in feature space. Accounts in sparse regions score higher.

|Parameter      |Value      |Notes                          |
|---------------|-----------|-------------------------------|
|`n_neighbors`  |20         |Number of neighbors to consider|
|`contamination`|0.04       |Match Isolation Forest setting |
|`metric`       |`euclidean`|Standard distance metric       |

**Output:** `lof_score` and `lof_flag`

-----

### 6.4 Ensemble Score

**File:** `src/aml_anomaly/models/ensemble.py`

Combine both model outputs into a single **composite anomaly score**:

```
anomaly_score = (normalized_isolation_forest_score × 0.6) + (normalized_lof_score × 0.4)
```

Normalize each model’s raw score to [0, 1] before combining. Final output:

- `anomaly_score` — continuous composite score (higher = more anomalous)
- `anomaly_rank` — rank of each account from most to least anomalous
- `anomaly_flag` — binary flag: 1 if account is in the top 4% by anomaly score

-----

## 7. Train / Score Pipeline Split

This is a critical design requirement. The model must support two distinct operating modes:

### 7.1 Training Mode

Fits the model on the full synthetic dataset and saves all fitted objects to disk. This represents the “learning what normal looks like” phase.

**Steps:**

1. Generate synthetic data (accounts, securities, trades)
1. Engineer features → produce training feature matrix
1. Fit `StandardScaler` → save scaler to `models/scaler.pkl`
1. Fit PCA → save to `models/pca.pkl` (retain components explaining 95% variance)
1. Fit Isolation Forest → save to `models/isolation_forest.pkl`
1. Fit LOF → save to `models/lof.pkl`
1. Save training population feature statistics (mean, std per feature) to `models/population_stats.json` — used for deviation scoring at reporting time

**File:** `src/aml_anomaly/models/train.py`

-----

### 7.2 Scoring Mode (Holdout / New Data)

Loads all fitted objects and scores a new batch of accounts **without retraining**. This is the production-representative flow — new trades come in, get featurized, and are scored against the already-fitted model.

**Steps:**

1. Accept a new trades file as input (same schema as `data/raw/trades.csv`)
1. Engineer features using the same pipeline — but apply the **saved scaler and PCA**, do not refit
1. Run accounts through saved Isolation Forest and LOF
1. Produce anomaly scores and flags
1. Generate analyst report (see Section 8)

**File:** `src/aml_anomaly/models/score.py`

**CLI usage:**

```bash
python score.py --trades data/holdout/new_trades.csv --output outputs/scored_holdout.xlsx
```

The holdout dataset should be a ~20% split of the synthetic trades, set aside before training, to serve as the default test of the scoring pipeline.

-----

## 8. Reporting & Analyst Output

Every flagged account produces a two-layer output designed for both AML analysts and data scientists to use together.

### 8.1 Layer 1 — Plain-English Narrative

**File:** `src/aml_anomaly/reporting/flags.py`

A concise, readable summary of why the account was flagged. Written in plain English with no variable names or statistical jargon. Generated programmatically by evaluating each flagged account’s top features against population thresholds.

**Example narratives:**

> *“This account’s trading activity surged to 12 times its normal pace over the past 7 days, with nearly all of that activity concentrated in two thinly-traded stocks. The account also executed 6 same-day buy/sell reversals — a pattern not seen in its prior history.”*

> *“This account directed 94% of its monthly trade value into micro-cap stocks and placed 14 trades within a single 20-minute window. Its total trade value this month was 8 standard deviations above its own 90-day average.”*

> *“This account traded exclusively with a single counterparty across 31 transactions, with an equal split of buys and sells on the same tickers — consistent with wash trading patterns.”*

Narrative generation rules:

- Only reference features that are in the top 5 contributors for that account
- Express deviations in plain terms (“12 times its normal pace”, “8 standard deviations above its own average”) — not raw variable values
- Do not use variable names in the narrative — those appear in Layer 2
- Keep each narrative to 2–4 sentences

-----

### 8.2 Layer 2 — Ranked Feature Contributions

**File:** `src/aml_anomaly/reporting/flags.py` (same module, separate function)

For every flagged account, produce a ranked table of the features that contributed most to its anomaly score. Rank is determined by how far each feature value deviates from the population mean in standard deviations (σ).

**Example output table:**

|Rank|Feature                       |Plain-English Label                   |Account Value     |Population Avg|Deviation|
|----|------------------------------|--------------------------------------|------------------|--------------|---------|
|1   |`velocity_ratio_7d_vs_30d`    |Recent trading pace vs. 30-day average|12.4×             |1.1×          |+10.3σ   |
|2   |`micro_cap_trade_pct`         |% of trades in micro-cap stocks       |94%               |12%           |+4.1σ    |
|3   |`same_day_reversal_count`     |Same-day buy/sell reversals           |6                 |0.3           |+3.8σ    |
|4   |`top_ticker_concentration_pct`|% of activity in single stock         |87%               |18%           |+3.5σ    |
|5   |`value_zscore_vs_self`        |Trade value vs. own 90-day history    |8.2σ above own avg|0.1σ          |+3.2σ    |

Every feature in this table must also have its plain-English label pulled from the data dictionary — so both the variable name and the human-readable label appear side by side. The data scientist sees the variable name; the analyst reads the label.

-----

### 8.3 Layer 3 — Supporting Trade Detail

For each flagged account, append the 10–20 most relevant individual trades that support the flag — i.e. the specific transactions that drove the anomalous features.

|Column                   |Description                                    |
|-------------------------|-----------------------------------------------|
|`trade_id`               |Unique trade identifier                        |
|`trade_date`             |Date of the trade                              |
|`trade_time`             |Time of the trade                              |
|`ticker`                 |Stock traded                                   |
|`trade_direction`        |Buy or Sell                                    |
|`quantity`               |Shares traded                                  |
|`trade_value_usd`        |Dollar value                                   |
|`counterparty_account_id`|Who was on the other side                      |
|`is_off_hours`           |Whether trade was outside market hours         |
|`is_round_value`         |Whether trade value was suspiciously round     |
|`flag_reason`            |Which anomaly pattern this trade contributed to|

-----

### 8.4 Export Files

**File:** `src/aml_anomaly/reporting/export.py`

|Output File                    |Audience                      |Description                                                        |
|-------------------------------|------------------------------|-------------------------------------------------------------------|
|`outputs/anomaly_scores.csv`   |Data scientists               |All accounts with raw anomaly scores, ranks, and all feature values|
|`outputs/flagged_accounts.xlsx`|AML analysts + data scientists|Multi-tab Excel workbook (see below)                               |

**Excel workbook structure (`flagged_accounts.xlsx`):**

- **Tab 1 — Summary:** One row per flagged account. Columns: account profile, anomaly score, anomaly rank, narrative summary. Sorted by anomaly rank descending.
- **Tab 2 — Feature Detail:** One row per flagged account. Columns: account ID, all ranked feature contributions with plain-English labels, deviation scores.
- **Tab 3 — Supporting Trades:** One row per relevant trade for flagged accounts. Linked to Tab 1 by account ID.
- **Tab 4 — Population Benchmarks:** The population mean and standard deviation for every feature, so analysts can contextualize the deviation scores in Tabs 1–2.

-----

## 9. Notebooks

|Notebook                      |Purpose                                                                                                                                                                            |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`01_data_exploration.ipynb`   |Distribution of accounts, trades, and securities. Basic sanity checks on generated data.                                                                                           |
|`02_feature_engineering.ipynb`|Feature distributions, correlation heatmap, null rates. Identify any features to drop before modeling.                                                                             |
|`03_pca_analysis.ipynb`       |Scree plot, cumulative variance explained, top feature loadings per component.                                                                                                     |
|`04_model_training.ipynb`     |Run Isolation Forest and LOF in training mode. Tune contamination parameter. Save fitted model objects.                                                                            |
|`05_scoring_holdout.ipynb`    |Load fitted models. Score the holdout trades dataset. Confirm scoring pipeline works end to end without retraining.                                                                |
|`06_results_review.ipynb`     |Top flagged accounts from holdout scoring. Anomaly score distributions. Validate injected anomalies were caught. Review sample analyst output narratives and ranked feature tables.|

-----

## 10. Data Dictionary

This dictionary defines every variable used across the project in plain English. It is intended for both technical and non-technical audiences.

### Accounts

|Variable                   |What It Means                                                                                                                   |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------|
|`account_id`               |A unique ID number assigned to each customer account                                                                            |
|`customer_name`            |The name of the person or entity who owns the account                                                                           |
|`account_type`             |What kind of account it is: individual investor (retail), a company (institutional), or a financial intermediary (broker-dealer)|
|`account_open_date`        |The date the account was first opened                                                                                           |
|`account_age_days`         |How many days the account has been open                                                                                         |
|`country_of_origin`        |The country where the account holder is based                                                                                   |
|`risk_tier`                |A number from 1 to 5 assigned during account opening based on AML risk factors. 1 is lowest risk, 5 is highest.                 |
|`is_pep`                   |Yes/No — whether the account holder is a government official or politically connected person (Politically Exposed Person)       |
|`is_high_risk_jurisdiction`|Yes/No — whether the account holder’s country appears on an international financial watchlist                                   |
|`annual_income_usd`        |The account holder’s stated annual income at the time of account opening                                                        |
|`net_worth_usd`            |The account holder’s stated total net worth at the time of account opening                                                      |

### Securities

|Variable              |What It Means                                                                                                                                                       |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`ticker`              |The short code used to identify a stock (e.g. AAPL for Apple)                                                                                                       |
|`company_name`        |The name of the company the stock belongs to                                                                                                                        |
|`sector`              |The industry the company operates in (e.g. Technology, Healthcare, Energy)                                                                                          |
|`market_cap_tier`     |How big the company is: Large (big, well-known companies), Mid, Small, or Micro (very small companies with limited trading)                                         |
|`avg_daily_volume`    |The typical number of shares of this stock traded on any given day                                                                                                  |
|`avg_price_usd`       |The typical price of one share of this stock                                                                                                                        |
|`bid_ask_spread_pct`  |The difference between what buyers are willing to pay and what sellers are asking for, expressed as a percentage. A large spread means the stock is harder to trade.|
|`price_volatility_30d`|How much the stock price typically moves up or down over a 30-day period                                                                                            |
|`exchange`            |The marketplace where the stock is listed and traded                                                                                                                |
|`is_illiquid`         |Yes/No — whether this stock trades in very low volumes, making it hard to buy or sell large amounts without moving the price                                        |

### Trades

|Variable                 |What It Means                                                                                                                                                   |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`trade_id`               |A unique ID assigned to each individual trade                                                                                                                   |
|`account_id`             |The account that placed this trade                                                                                                                              |
|`counterparty_account_id`|The account on the other side of this trade (who sold if the account was buying, and vice versa)                                                                |
|`ticker`                 |The stock that was traded                                                                                                                                       |
|`trade_date`             |The date the trade took place                                                                                                                                   |
|`trade_time`             |The time of day the trade was executed                                                                                                                          |
|`trade_direction`        |Whether the account was buying or selling                                                                                                                       |
|`quantity`               |The number of shares traded                                                                                                                                     |
|`price_usd`              |The price paid or received per share                                                                                                                            |
|`trade_value_usd`        |The total dollar amount of the trade (shares × price)                                                                                                           |
|`order_type`             |How the trade was placed: Market (execute immediately at current price), Limit (only execute at a specific price), or Stop (execute when price hits a threshold)|
|`time_to_execution_ms`   |How many milliseconds passed between placing the order and it being filled. Very short times can indicate automated trading.                                    |
|`is_off_hours`           |Yes/No — whether the trade happened outside normal market hours (before 9:30am or after 4:00pm Eastern Time)                                                    |
|`is_round_value`         |Yes/No — whether the total trade value is a suspiciously round number (e.g. exactly $9,500 or $10,000), which can be a sign of deliberate structuring           |

### Engineered Features

|Variable                            |What It Means                                                                                                                                                                                                       |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`trades_per_day_30d`                |How many trades this account placed per day on average over the past 30 days                                                                                                                                        |
|`velocity_ratio_7d_vs_30d`          |How much faster or slower the account has been trading this week compared to its 30-day average. A ratio much greater than 1 means a sudden spike in activity.                                                      |
|`burst_event_count`                 |How many times the account placed 5 or more trades within a 30-minute window                                                                                                                                        |
|`top_ticker_concentration_pct`      |What percentage of the account’s total trading activity went into just one stock                                                                                                                                    |
|`illiquid_stock_trade_pct`          |What percentage of the account’s trading was in thinly-traded stocks                                                                                                                                                |
|`trade_size_vs_adv_max`             |The largest single trade this account placed, measured as a percentage of the stock’s normal daily trading volume. A very high number means the trade was unusually large relative to that stock’s typical activity.|
|`buy_sell_ratio_30d`                |The ratio of buying to selling over 30 days. A ratio very close to 1.0 (equal buying and selling) can be a sign of wash trading.                                                                                    |
|`round_value_trade_pct`             |What percentage of the account’s trades had suspiciously round dollar values                                                                                                                                        |
|`value_zscore_vs_self`              |How unusual this account’s recent trading volume is compared to its own history, measured in standard deviations. A score above 3 means the account is doing something very different from its normal pattern.      |
|`new_ticker_pct_30d`                |What percentage of recent trades involved stocks the account has never traded before                                                                                                                                |
|`off_hours_trade_pct`               |What percentage of this account’s trades happened outside normal market hours                                                                                                                                       |
|`avg_holding_period_minutes`        |On average, how long this account holds a stock between buying and selling it                                                                                                                                       |
|`min_holding_period_minutes`        |The shortest time between a buy and a sell of the same stock. Very short holding periods can indicate in-and-out trading patterns.                                                                                  |
|`unique_counterparties_30d`         |How many different accounts this account traded with in the past 30 days. A very small number may indicate circular or coordinated trading.                                                                         |
|`top_counterparty_concentration_pct`|What percentage of this account’s trades went to a single counterparty                                                                                                                                              |
|`same_day_reversal_count`           |How many times the account bought and sold the same stock on the same day                                                                                                                                           |
|`circular_trade_flag`               |Yes/No — whether this account appears to be part of a circular trading chain where the same stock passes through multiple accounts and ends up back at the start                                                    |
|`anomaly_score`                     |A composite score combining both models. Higher scores indicate more unusual behavior.                                                                                                                              |
|`anomaly_rank`                      |The account’s rank from most to least anomalous, across all accounts                                                                                                                                                |
|`anomaly_flag`                      |Yes/No — whether this account is in the top 4% most anomalous accounts in the dataset                                                                                                                               |

-----

## 11. Implementation Notes for Claude Code

- Generate data in this order: securities → accounts → trades (trades depend on both)
- Set aside 20% of trades as holdout **before** any feature engineering or model fitting — save to `data/holdout/new_trades.csv`
- Seed all random generators with `random_state=42` throughout
- The `pipeline.py` feature assembler should be the single entry point for the modeling notebooks — it calls all feature modules and returns one clean DataFrame
- `train.py` fits and saves all model objects (scaler, PCA, Isolation Forest, LOF) to a `models/` directory as `.pkl` files; `score.py` loads them — never refits
- Save population feature statistics (mean, std per feature) to `models/population_stats.json` at training time — the reporting module reads this to compute deviation scores
- The plain-English label for every feature must be maintained in a central dictionary in `reporting/flags.py` so it can be referenced by both narrative generation and the ranked feature table
- All file paths should be relative and use `pathlib.Path`
- Add docstrings to every function with a one-line plain-English description of what it does
- The `injected_anomalies.csv` reference file should be written during data generation but never read by any model or feature module — it is for validation only
- Include a `run_all.py` script at the project root that executes the full pipeline end to end: data generation → train/holdout split → feature engineering → training → scoring holdout → export

-----

*End of project specification.*