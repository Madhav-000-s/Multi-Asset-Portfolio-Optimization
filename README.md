# DSR Portfolio Optimizer

**Deep Portfolio Optimization Using Differential Sharpe Ratio and LSTM Networks**

A full-stack portfolio optimization system that trains a 2-layer LSTM to directly output portfolio weights by maximizing the Differential Sharpe Ratio (DSR), with walk-forward backtesting, CVaR risk constraints, and an interactive dashboard.

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
                → dashboard.py / shiny/  → interactive dashboard
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

### Layer 5 — Presentation
- `dashboard.py` — Streamlit dashboard (4 tabs: Overview, Weights, Analytics, Risk)
- `shiny/` — R Shiny dashboard equivalent (requires R)
- `reports/report.Rmd` — R Markdown → comprehensive HTML report

---

## Quickstart

### Python setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install yfinance pandas numpy pyarrow ta scikit-learn pyyaml
pip install streamlit plotly   # for the dashboard
```

### Run the interactive dashboard

```bash
streamlit run dashboard.py --server.port 8501
# Open http://localhost:8501
```

### Re-train the model

```bash
cd python
python train.py          # trains from scratch, saves models/best_model.pt
```

### Re-run data download

```bash
cd python
python data_loader.py    # downloads fresh OHLCV to data/raw/prices.parquet
```

### Run the R backtest (requires R)

```r
# Install R packages
install.packages(c("reticulate", "PortfolioAnalytics", "PerformanceAnalytics",
                   "xts", "zoo", "shiny", "shinydashboard", "ggplot2",
                   "plotly", "DT", "scales", "tidyr", "yaml", "rmarkdown"))

# Run full backtest pipeline
source("R/backtest.R")

# Launch Shiny dashboard
shiny::runApp("shiny/", port = 3838)
```

---

## File Structure

```
├── config.yaml                    # Asset universe, dates, hyperparameters
├── requirements.txt               # Python dependencies
├── dashboard.py                   # Streamlit dashboard (no R needed)
│
├── python/
│   ├── data_loader.py             # OHLCV download + parquet storage
│   ├── features.py                # Technical indicators + SequenceDataset
│   ├── model.py                   # PortfolioLSTM (2-layer, Softmax output)
│   ├── dsr_loss.py                # Differential Sharpe Ratio loss
│   ├── train.py                   # Training loop (sequential, Adam, early stopping)
│   ├── predict.py                 # Inference + caching for R bridge
│   └── test_bridge.py             # R↔Python bridge test helpers
│
├── R/
│   ├── backtest.R                 # Walk-forward backtesting engine
│   ├── constraints.R              # CVaR + position limit constraints
│   ├── efficient_frontier.R       # Monte Carlo frontier + visualization
│   ├── metrics.R                  # Performance metrics + plots
│   ├── report_utils.R             # Monthly heatmap, rolling metrics, regime analysis
│   ├── transaction_costs.R        # Turnover tracking + cost deduction
│   ├── utils.R                    # Reticulate setup helpers
│   └── test_bridge.R              # Bridge integration tests
│
├── shiny/
│   ├── global.R                   # Data loading + package setup
│   ├── ui.R                       # Dashboard layout (4 tabs)
│   └── server.R                   # Reactive logic
│
├── data/
│   └── raw/prices.parquet         # OHLCV for 5 assets (2018–2024, 1760 rows)
│
├── models/
│   └── best_model.pt              # Trained checkpoint (val Sharpe: 1.59)
│
├── results/
│   ├── backtest_results.rds       # Full R backtest object
│   ├── metrics.csv                # Performance summary table
│   ├── weights_history.csv        # Constrained weights (56 rebalance dates)
│   ├── weights_unconstrained.csv  # Unconstrained LSTM weights
│   ├── equity_curve.png           # Cumulative returns chart
│   ├── drawdown.png               # Drawdown analysis
│   ├── weight_allocation.png      # Stacked area allocation
│   └── efficient_frontier.png     # Risk-return scatter
│
└── reports/
    ├── report.Rmd                 # R Markdown source
    └── report.html                # Pre-generated HTML report (1.5 MB)
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

---

## Build Plan Status

| Iteration | Description | Status |
|---|---|---|
| ITER 0 | Environment, skeleton, R↔Python bridge | Done |
| ITER 1 | LSTM + DSR loss + backtest + equity curve | Done |
| ITER 2 | FinBERT sentiment augmentation | Not started |
| ITER 3 | CVaR constraints + efficient frontier | Done |
| ITER 4 | Performance report (R Markdown) | Done |
| ITER 5 | Shiny + Streamlit dashboard | Done |
