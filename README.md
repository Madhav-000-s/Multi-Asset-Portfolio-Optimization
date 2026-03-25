# DSR Portfolio Optimizer

**Deep Portfolio Optimization Using Differential Sharpe Ratio and LSTM Networks**

A full-stack portfolio optimization system that trains a 2-layer LSTM to directly output portfolio weights by maximizing the Differential Sharpe Ratio (DSR), with walk-forward backtesting, CVaR risk constraints, and interactive dashboards.

---

## Results (Test Period: Dec 2023 – Dec 2024)

| Strategy | Ann. Return | Sharpe | Volatility | Max Drawdown |
|---|---|---|---|---|
| DSR Unconstrained | 69.1% | 3.02 | 22.9% | 12.9% |
| DSR Constrained | 52.6% | 2.51 | 21.0% | 14.3% |
| Equal Weight (benchmark) | 43.2% | 2.13 | 20.2% | 14.4% |

Assets: AAPL · MSFT · GOOGL · AMZN · META · Weekly rebalancing · 10 bps transaction costs

---

## System Architecture

```
config.yaml
  → data_loader.py       # Download OHLCV via yfinance → data/raw/prices.parquet
    → features.py        # 6 technical indicators × 5 assets → (batch, T=60, F=30) tensor
      → train.py         # LSTM + DSR loss → models/best_model.pt
        → predict.py     # Load model, return weights w_t for any date
          → backtest.R   # Walk-forward via reticulate → daily portfolio returns
            → constraints.R  # CVaR + max 30% per asset
              → metrics.R / report_utils.R → results/ + reports/report.html
                → dashboard.py / portfolio_builder.py / market_analysis.py
```

### Layer 1 — Data Ingestion (Python)
- `python/data_loader.py` — Downloads OHLCV for 5 tickers via `yfinance`, saves to `data/raw/prices.parquet`

### Layer 2 — Feature Engineering (Python)
- `python/features.py` — 6 features per asset: log returns, rolling 20d volatility, RSI-14, MACD, Bollinger %B, 50d/200d SMA ratio
- Rolling z-score normalization (no lookahead bias), temporal 70/15/15 train/val/test split
- Output: `SequenceDataset` with `(batch, T=60, F=30)` tensors

### Layer 3 — DSR-LSTM Core (Python/PyTorch)
- `python/model.py` — 2-layer LSTM (128→64 hidden), Softmax head → weights ∈ ℝ^N, sum to 1
- `python/dsr_loss.py` — Differential Sharpe Ratio loss (Moody & Saffell 2001): `D_t = (B_{t-1}·ΔA − 0.5·A_{t-1}·ΔB) / (B_{t-1} − A_{t-1}²)^{3/2}`, η=0.01
- `python/train.py` — Sequential training (no shuffle), Adam, ReduceLROnPlateau, early stopping, saves `models/best_model.pt`
- `python/predict.py` — Inference interface with caching, designed for R↔Python bridge

### Layer 4 — Execution & Risk Engine (R)
- `R/backtest.R` — Walk-forward simulation calling `predict.py` via reticulate on each rebalance date
- `R/constraints.R` — Min/max weight bounds + CVaR constraint (95% confidence, max 30% per asset)
- `R/transaction_costs.R` — 10 bps per rebalance, turnover tracking
- `R/efficient_frontier.R` — Monte Carlo efficient frontier with DSR portfolio overlay
- `R/metrics.R` — Sharpe, Sortino, Calmar, MaxDD, VaR, CVaR via `PerformanceAnalytics`

### Layer 5 — Interactive Applications

Three Streamlit apps, each serving a different purpose:

**1. `dashboard.py` — Main Backtesting Dashboard**
```bash
streamlit run dashboard.py --server.port 8501
```
7 tabs covering the full backtest analysis:
- **Portfolio Performance** — Equity curves, annualised return/Sharpe/volatility/drawdown cards, strategy comparison table
- **Weight Allocation** — Stacked area chart of weights over time, current allocation pie, HHI concentration metric
- **Performance Analytics** — Monthly returns heatmap, rolling Sharpe, rolling volatility, high/low volatility regime breakdown
- **Risk Monitor** — Underwater drawdown chart, efficient frontier overlay, rolling VaR/CVaR, full risk comparison table
- **Live Data** — Real-time LSTM weight recommendation vs last backtest, recent price chart for all watchlist assets
- **FinBERT Sentiment** — Per-asset sentiment scores (FinBERT), sentiment heatmap, sentiment vs returns overlay, ablation results
- **Real-Time Technical Signals** — Market open/close status, composite signal table (STRONG BUY → STRONG SELL) for all watchlist tickers, live candlestick chart with RSI/MACD/Bollinger detail

**2. `portfolio_builder.py` — Custom Portfolio Builder**
```bash
streamlit run portfolio_builder.py --server.port 8502
```
Select any 2–10 stocks from a universe of ~50 tickers (Tech, Finance, Healthcare, Consumer, Energy, ETFs):
- If you pick the 5 trained assets (AAPL, MSFT, GOOGL, AMZN, META) → uses the trained LSTM model directly
- Otherwise → Mean-Variance Optimization with a 30% technical signal tilt (RSI, MACD, SMA, Bollinger)
- Outputs: allocation pie chart, 8 risk metrics (Ann. Return, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, VaR 95%, CVaR 95%), 1-year hypothetical performance chart
- Risk metrics computed via R `PerformanceAnalytics` if available, else falls back to NumPy

**3. `market_analysis.py` — Bull vs Bear Market Analysis**
```bash
streamlit run market_analysis.py --server.port 8503
```
Side-by-side comparison of DSR strategy performance across market regimes:
- 2024 Bull Market (Dec 2023–Dec 2024): cumulative returns, metrics, weight allocation
- 2022 Bear Market stress test: drawdown analysis, metrics vs equal-weight benchmark
- Reads pre-computed results from `results/bull_2024/` and `results/bear_2022/`

---

## Quickstart

### 1. Python setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. R setup (required for backtesting and risk metrics)

```bash
Rscript -e 'install.packages(c("reticulate", "PortfolioAnalytics", "PerformanceAnalytics", "xts", "zoo", "ggplot2", "plotly", "DT", "scales", "tidyr", "yaml", "rmarkdown"))'
```

### 3. Launch a dashboard (uses pre-computed results — no training needed)

```bash
streamlit run dashboard.py --server.port 8501        # Main backtesting dashboard
streamlit run portfolio_builder.py --server.port 8502 # Custom portfolio builder
streamlit run market_analysis.py --server.port 8503   # Bull vs bear analysis
```

---

## Regenerating Results from Scratch

Run the steps below **in order**. Each step depends on the previous one.

### Step 1 — Download price data

```bash
python python/data_loader.py
# Output: data/raw/prices.parquet (OHLCV for 5 assets, 2018–present)
```

### Step 2 — Train the LSTM model

```bash
python python/train.py
# Output: models/best_model.pt
# Trains for up to 100 epochs with early stopping (patience=10).
# Prints epoch-by-epoch DSR and Sharpe on train/val/test sets.
```

### Step 3 — Run the main walk-forward backtest (requires R)

```bash
Rscript R/backtest.R
```
Or from within R:
```r
source("R/backtest.R")
```
Outputs:
- `results/backtest_results.rds` — full R backtest object
- `results/weights_history.csv` — constrained weights (weekly rebalance dates)
- `results/weights_constrained.csv` — with CVaR + position limits applied
- `results/weights_unconstrained.csv` — raw LSTM outputs
- `results/metrics.csv` — Sharpe, Sortino, Calmar, MaxDD, VaR, CVaR summary
- `results/equity_curve.png`, `drawdown.png`, `weight_allocation.png`, `efficient_frontier.png`

To run the bull 2024 scenario explicitly:
```bash
BACKTEST_START_DATE=2023-12-01 BACKTEST_END_DATE=2024-12-31 Rscript R/backtest.R
# Outputs: results/bull_2024/
```

### Step 4 — Run the bear market stress test

```bash
python python/bear_test_2022.py
# Output: results/bear_2022/ (metrics, returns, weights CSVs for the 2022 bear period)
```

### Step 5 — Regenerate the HTML report (requires R + rmarkdown)

```bash
Rscript -e "rmarkdown::render('reports/report.Rmd')"
# Output: reports/report.html
```

---

## File Structure

```
├── config.yaml                    # Asset universe, dates, hyperparameters
├── requirements.txt               # Python dependencies
├── dashboard.py                   # Streamlit — Main Backtesting Dashboard (7 tabs)
├── portfolio_builder.py           # Streamlit — Custom Portfolio Builder (LSTM/MPT + R risk)
├── market_analysis.py             # Streamlit — Bull vs Bear Market Analysis
│
├── python/
│   ├── data_loader.py             # OHLCV download + parquet storage
│   ├── features.py                # Technical indicators + SequenceDataset
│   ├── model.py                   # PortfolioLSTM (2-layer, Softmax output)
│   ├── dsr_loss.py                # Differential Sharpe Ratio loss
│   ├── train.py                   # Training loop (sequential, Adam, early stopping)
│   ├── predict.py                 # Inference + caching for R bridge
│   ├── bear_test_2022.py          # Bear market stress test backtest
│   ├── ablation.py                # Feature importance analysis
│   ├── sentiment.py               # FinBERT sentiment scoring pipeline
│   ├── scraper.py                 # Financial headlines downloader
│   ├── signals.py                 # Technical signal generators
│   └── test_bridge.py             # R↔Python bridge test helpers
│
├── R/
│   ├── backtest.R                 # Walk-forward backtesting engine
│   ├── constraints.R              # CVaR + position limit constraints
│   ├── efficient_frontier.R       # Monte Carlo frontier + visualization
│   ├── metrics.R                  # Performance metrics + plots
│   ├── report_utils.R             # Monthly heatmap, rolling metrics, regime analysis
│   ├── transaction_costs.R        # Turnover tracking + cost deduction
│   ├── portfolio_risk.R           # PerformanceAnalytics metrics for Portfolio Builder
│   ├── utils.R                    # Reticulate setup helpers
│   └── test_bridge.R              # Bridge integration tests
│
├── data/
│   └── raw/prices.parquet         # OHLCV for 5 assets (2018–present)
│
├── models/
│   ├── best_model.pt              # Trained checkpoint (val Sharpe: 1.59)
│   └── best_model_2024_bull.pt    # Retrained on 2018–2021 for bear market test
│
├── results/
│   ├── backtest_results.rds       # Full R backtest object
│   ├── metrics.csv                # Performance summary table
│   ├── weights_history.csv        # Constrained weights (56 rebalance dates)
│   ├── weights_unconstrained.csv  # Unconstrained LSTM weights
│   ├── equity_curve.png           # Cumulative returns chart
│   ├── drawdown.png               # Drawdown analysis
│   ├── weight_allocation.png      # Stacked area allocation
│   ├── efficient_frontier.png     # Risk-return scatter
│   ├── bull_2024/                 # Bull market (Dec 2023–Dec 2024) results
│   └── bear_2022/                 # Bear market stress test (2022) results
│
└── reports/
    ├── report.Rmd                 # R Markdown source
    └── report.html                # Pre-generated HTML report
```

---

## Configuration (`config.yaml`)

```yaml
assets:
  tickers: [AAPL, MSFT, GOOGL, AMZN, META]
data:
  start_date: "2018-01-01"
  end_date:   "2024-12-31"
model:
  lookback_window: 60       # days of history per sample
  lstm_hidden_1: 128
  lstm_hidden_2: 64
  dropout: 0.3
  dsr_eta: 0.01             # DSR adaptation rate
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
  train_ratio: 0.70
  val_ratio:   0.15
  test_ratio:  0.15
backtest:
  rebalance_frequency: weekly
  transaction_cost_bps: 10
constraints:
  max_position_weight: 0.30
  min_position_weight: 0.02
  cvar_confidence: 0.95
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Loss function | Differential Sharpe Ratio | Enables gradient descent through Sharpe; no reward shaping needed |
| Model | LSTM (not Transformer) | Better for modest-length financial time-series; fewer parameters |
| Output head | Softmax | Guarantees long-only weights that sum to 1 |
| Normalization | Rolling z-score | Prevents lookahead bias; adapts to changing market regimes |
| Data split | Temporal 70/15/15 | Prevents future leakage; mirrors real deployment |
| R for risk | PortfolioAnalytics + PerformanceAnalytics | Industry-standard CVaR solver and performance reporting |
| Bridge | reticulate | Seamless R↔Python; passes numpy arrays as R matrices |
