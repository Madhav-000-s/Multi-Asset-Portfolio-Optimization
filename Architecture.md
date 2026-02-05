# DSR Portfolio Optimizer — Architecture & Build Plan

## Project Title
**Deep Portfolio Optimization Using Differential Sharpe Ratio and Sentiment-Augmented LSTM Networks**

---

## SYSTEM ARCHITECTURE (5 Layers)

### Layer 1: Data Ingestion [Python]
- **Price Fetcher**: OHLCV via `yfinance` → `data/raw/prices.parquet`
- **News Scraper**: Financial headlines (NewsAPI / Financial PhraseBank) → `data/raw/headlines.csv`
- **Config**: `config.yaml` controls asset universe (start with 5 stocks), date range, hyperparams

### Layer 2: Feature Engineering [Python]
- **FinBERT Sentiment**: `ProsusAI/finbert` scores headlines → daily "Alpha Signal" per ticker (net sentiment = P(pos) − P(neg))
- **Technical Features**: Log returns, rolling 20d volatility, RSI-14, MACD, Bollinger %B, 50d/200d SMA ratio
- **Tensor Builder**: Merges price + sentiment → 3D tensor `(batch, T=60, F)`, z-score normalized, no lookahead bias. Train/val/test split by date (70/15/15)

### Layer 3: DSR-LSTM Core [Python/PyTorch] ← THE CORE
- **LSTM Encoder**: 2-layer LSTM (128→64 hidden), dropout=0.3
- **Softmax Head**: Linear → Softmax → portfolio weights w_t ∈ ℝ^N, sum to 1, long-only
- **DSR Loss**: Custom loss = −D_t (Differential Sharpe Ratio). Uses exponential moving averages of returns (A_t) and squared returns (B_t). Adaptation rate η=0.01
- **Training**: Sequential through time, Adam optimizer, ReduceLROnPlateau, early stopping on val DSR. Saves `models/best_model.pt`

### Layer 4: Execution & Risk Engine [R]
- **Reticulate Bridge**: R calls Python `predict.py` → receives weight vector w_t
- **CVaR Constraint Layer**: `PortfolioAnalytics` enforces CVaR@95%, max 30% per asset, long-only. LSTM weights used as initial guess → ROI solver optimizes within constraints
- **Backtester**: Walk-forward simulation, configurable rebalance frequency, transaction costs (10bps)
- **Benchmark Engine**: Equal Weight, Buy & Hold, 60/40. Metrics via `PerformanceAnalytics`: Sharpe, Sortino, MaxDD, Calmar, VaR, CVaR

### Layer 5: Presentation [R/Shiny]
- **Shiny Dashboard** (4 pages): Portfolio overview + equity curve, weight allocation stacked area, sentiment heatmap, risk monitor (rolling VaR/CVaR/drawdown)
- **Report Generator**: R Markdown → PDF with all performance tables, charts, ablation results

---

## FILE STRUCTURE

```
dsr-portfolio/
├── config.yaml                  # Asset universe, dates, hyperparams
├── requirements.txt             # Python deps
├── renv.lock                    # R deps
├── data/raw/                    # prices.parquet, headlines.csv
├── data/processed/              # features.parquet, sentiment_scores.parquet
├── python/
│   ├── data_loader.py           # Download & clean price data
│   ├── sentiment.py             # FinBERT scoring pipeline
│   ├── features.py              # Feature engineering + tensor builder
│   ├── model.py                 # PortfolioLSTM class
│   ├── dsr_loss.py              # Differential Sharpe Ratio loss
│   ├── train.py                 # Training loop
│   └── predict.py               # Load model, return weights
├── R/
│   ├── backtest.R               # Walk-forward simulation
│   ├── constrain_weights.R      # CVaR + PortfolioAnalytics
│   ├── benchmarks.R             # EW, B&H, 60/40
│   ├── metrics.R                # Performance metrics
│   └── utils.R                  # Helpers
├── shiny/app.R, ui.R, server.R  # Dashboard
├── reports/report.Rmd           # R Markdown report
├── models/best_model.pt         # Trained checkpoint
├── results/                     # backtest_returns.rds, weights_history.csv, metrics.csv
└── notebooks/                   # EDA + experiments
```

---

## DATA FLOW (Pipeline)

```
config.yaml
  → data_loader.py → prices.parquet
  → scraper/dataset → headlines.csv
    → sentiment.py (FinBERT) → sentiment_scores.parquet
      → features.py (merge + normalize) → Tensor(batch, T=60, F)
        → train.py (LSTM + DSR loss) → best_model.pt
          → backtest.R calls predict.py via reticulate → raw w_t
            → constrain_weights.R (CVaR/PortfolioAnalytics) → constrained w_t
              → w_t × r_(t+1) → backtest_returns.rds
                → report.Rmd → PDF | app.R → Shiny dashboard
```

---

## ITERATIVE BUILD PLAN

Each iteration yields a **working, demonstrable system**. Can stop after any iteration.

### ITER 0 — Environment & Skeleton (2–3 days)
**Goal**: Dev environment boots, R↔Python bridge works
- [ ] Conda env: torch, transformers, yfinance, pandas, numpy
- [ ] Create folder structure + config.yaml (5 stocks, 2018-2024)
- [ ] `data_loader.py` — download OHLCV to parquet
- [ ] R packages: reticulate, PortfolioAnalytics, PerformanceAnalytics, xts, shiny
- [ ] Verify `reticulate::py_config()` finds conda env
- [ ] Test: R calls dummy Python function, gets vector back
- **Deliverable**: Running environment. Price data on disk. R↔Python bridge verified.

### ITER 1 — Minimal End-to-End Pipeline, No Sentiment (7–10 days)
**Goal**: LSTM outputs weights → R backtests → equity curve
- [ ] Compute price features: log returns, rolling vol, RSI, SMA ratio
- [ ] Z-score normalize (rolling window, no lookahead)
- [ ] Build `SequenceDataset`: sliding window → (batch, T=60, F) tensors
- [ ] Implement `DSRLoss`: exponential moving A_t, B_t, compute D_t
- [ ] Build `PortfolioLSTM`: 2-layer LSTM → Linear → Softmax
- [ ] Training loop: sequential through time, Adam, early stopping
- [ ] `predict.py`: loads model, takes features, returns w_t
- [ ] `backtest.R`: walk-forward loop calling Python predict at each rebalance
- [ ] Compute portfolio returns: w_t · r_(t+1), no constraints yet
- [ ] Generate equity curve + Equal Weight benchmark
- [ ] Print Sharpe, MaxDD, Annualized Return
- **Deliverable**: Working backtest. Equity curve chart. DSR vs Equal Weight table.

### ITER 2 — Add Sentiment / Alpha Signal (5–7 days)
**Goal**: FinBERT scores augment features, model re-trains
- [ ] Pick dataset: Financial PhraseBank OR NewsAPI headlines
- [ ] Load `ProsusAI/finbert`, write `sentiment.py`
- [ ] Score headlines → (date, ticker, sentiment_score)
- [ ] Aggregate daily: mean sentiment per ticker → Alpha Signal time-series
- [ ] Handle missing days (forward-fill / neutral default)
- [ ] Append sentiment to feature tensor, re-train LSTM
- [ ] Ablation: compare val DSR with vs without sentiment
- [ ] Re-run backtest, add "DSR (no sentiment)" as third benchmark
- **Deliverable**: 3-way equity curve. Ablation study table.

### ITER 3 — Risk Constraints: CVaR & Efficient Frontier (5–7 days)
**Goal**: LSTM weights filtered through PortfolioAnalytics
- [ ] Define portfolio spec: long-only, sum-to-1, max 30% per asset
- [ ] Add CVaR constraint (95% confidence)
- [ ] `constrain_weights()`: LSTM weights as initial guess → ROI solver
- [ ] Integrate into walk-forward loop
- [ ] Add transaction cost model (10bps)
- [ ] Compare: unconstrained vs constrained vs benchmarks
- [ ] Plot Efficient Frontier with DSR portfolio marked
- **Deliverable**: Risk-constrained portfolio. Efficient Frontier plot. Reduced MaxDD.

### ITER 4 — Performance Report & Analytics (4–5 days)
**Goal**: Comprehensive R Markdown report
- [ ] Monthly returns heatmap per strategy
- [ ] Rolling Sharpe (252d window)
- [ ] Drawdown + underwater plot
- [ ] Full metrics table: Sharpe, Sortino, Calmar, MaxDD, VaR, CVaR
- [ ] Convergence speed comparison: DSR epochs vs RL (simulated/cited)
- [ ] High-vol regime analysis (COVID, 2022 bear)
- [ ] Feature importance ablation
- [ ] Sensitivity to lookback window (30, 60, 90 days)
- **Deliverable**: PDF/HTML report via R Markdown.

### ITER 5 — Shiny Dashboard (5–7 days)
**Goal**: Interactive dashboard
- [ ] Page 1: Portfolio overview — equity curve, cumulative returns, metric cards
- [ ] Page 2: Weight allocation — stacked area chart, current allocation pie
- [ ] Page 3: Sentiment heatmap — asset × date, overlay with returns
- [ ] Page 4: Risk monitor — rolling VaR, CVaR, drawdown, Efficient Frontier
- [ ] Sidebar: date range, rebalance frequency, benchmark selector
- [ ] Connect to pre-computed results (no live computation)
- [ ] Download button for report PDF
- **Deliverable**: Functional Shiny app. Deployable on shinyapps.io.

### BONUS — Extensions (if time permits)
- [ ] Simple RL baseline (DQN) for convergence speed comparison
- [ ] Expand to 15–20 assets (ETFs, commodities)
- [ ] Attention mechanism on LSTM for interpretability
- [ ] Social media sentiment (Reddit/Twitter archived data)
- [ ] Monte Carlo robustness test (bootstrap returns)
- [ ] Dockerize project

---

## KEY TECHNICAL DECISIONS

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Loss function | Differential Sharpe Ratio (not standard Sharpe) | Enables online/incremental optimization via gradient descent |
| Model | LSTM (not Transformer) | Better for modest-length financial time-series; fewer params |
| No RL | Direct optimization via DSR loss | Faster convergence, no reward shaping, stable training |
| R for risk | PortfolioAnalytics + PerformanceAnalytics | Industry-standard, CVaR solver built-in, rich reporting |
| Bridge | reticulate | Seamless R↔Python, passes numpy arrays as R matrices |
| Sentiment model | FinBERT (not generic BERT) | Pre-trained on financial text, no fine-tuning needed |

---

## CORE PACKAGES

**Python**: torch, transformers, yfinance, pandas, numpy, ta-lib (or ta), scikit-learn, pyarrow
**R**: reticulate, PortfolioAnalytics, PerformanceAnalytics, ROI, DEoptim, xts, zoo, shiny, shinydashboard, plotly, rmarkdown, knitr, kableExtra