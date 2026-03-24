# Presentation Generation Prompt
## DSR Portfolio Optimizer — 25-Slide Deck

**Purpose:** This file is a complete blueprint for generating a 25-slide PowerPoint presentation.
Each slide entry specifies the title, subtitle, content, visual/layout direction, and speaker notes.
Feed this file to a presentation generator (e.g. `create_presentation.py`, ChatGPT, or python-pptx) to produce the final deck.

---

## Global Design Spec

| Property | Value |
|----------|-------|
| Dimensions | 10 × 7.5 inches (standard widescreen) |
| Primary colour | `#1f4e79` (dark navy) — slide headers, Python elements |
| R accent colour | `#e67e22` (orange) — all R-related slides and elements |
| Success colour | `#1e8b4c` (green) — results, outputs |
| Highlight colour | `#2e86c1` (blue) — secondary Python elements |
| Background | `#f2f3f4` (light grey) for content slides; `#1f4e79` for title/results |
| Body font | Calibri or Inter, 12–14pt body, 24–28pt titles |
| Code font | Consolas or Fira Code, 10pt, dark background box |

**Slide structure (25 total):**
- Slides 1–11 → High-level overview (project, architecture, all 5 layers, results)
- Slides 12–25 → R deep dive (14 slides)

---

## PART 1 — HIGH-LEVEL OVERVIEW (Slides 1–11)

---

### SLIDE 1 — Title

**Layout:** Full-bleed navy background, orange top and bottom accent bars (0.1in height), decorative orange left strip.

**Title (large, white, centred):**
```
Deep Portfolio Optimization
Using DSR + Sentiment-Augmented LSTM
```

**Subtitle (medium, light blue, centred):**
```
Architecture Deep Dive — With a Special Focus on R's Role
```

**Tagline (small, grey, centred):**
```
Python  ·  R  ·  PyTorch  ·  reticulate
```

**Bottom line (tiny, grey):**
```
5 Assets  ·  2018–2024 Training  ·  Walk-Forward Backtest  ·  Live Dashboard
```

**Speaker notes:**
> "This project combines deep learning for portfolio weight prediction with R's quantitative finance ecosystem for risk management. It's a two-language system where each language does what it's genuinely best at."

---

### SLIDE 2 — What Are We Building?

**Layout:** Navy header bar, light grey background, left column text + right column 3 info cards.

**Header:** What Are We Building?
**Subheader:** A casual overview — no jargon required

**Pull quote (italic, dark, left column):**
> "Use deep learning to figure out how to split money across 5 big tech stocks,
> then let R double-check that we're not taking crazy risks."

**Bullet points (left column, below quote):**
- We have 5 assets: AAPL, MSFT, GOOGL, AMZN, META
- An LSTM reads 60 days of price history each week
- It outputs 5 weights — e.g. 30% Apple, 20% Google…
- R then checks: are those weights safe? Too concentrated?
- R runs the full backtest and measures how we'd have done
- A Streamlit dashboard + standalone Portfolio Builder show it live

**Right column — 3 stacked cards (navy/blue/green):**
```
Training data: 2018–2024
Test period:   Dec 2023 – Dec 2024
Rebalance:     Weekly · 10 bps transaction cost
```

**Speaker notes:**
> "The key insight is the two-language split. Python is better at deep learning, R is better at financial risk management. We use both where they shine instead of forcing one language to do everything."

---

### SLIDE 3 — System Architecture — 5 Layers

**Layout:** Navy header. Full-slide matplotlib diagram (embedded as PNG).

**Header:** System Architecture — 5 Layers
**Subheader:** Each layer feeds the next; Python hands off to R at Layer 4

**Diagram spec (vertical stack, left boxes + right artifact boxes):**

```
┌────────────────────────────────────┐      ┌─────────────────────────┐
│  Layer 1 — Data Ingestion          │ ───► │ prices.parquet          │
│  yfinance · Alpha Vantage · NewsAPI│      │ headlines.csv           │
└────────────────────────────────────┘      └─────────────────────────┘
                    ↓
┌────────────────────────────────────┐      ┌─────────────────────────┐
│  Layer 2 — Feature Engineering     │ ───► │ features (30–35 dims)   │
│  Log Ret · Vol · RSI · MACD · FinBERT    │ sentiment_scores.parquet│
└────────────────────────────────────┘      └─────────────────────────┘
                    ↓
┌────────────────────────────────────┐      ┌─────────────────────────┐
│  Layer 3 — LSTM + DSR Loss         │ ───► │ best_model.pt           │
│  PyTorch · 2-Layer LSTM · Softmax  │      │ weights w_t ∈ ℝ⁵       │
└────────────────────────────────────┘      └─────────────────────────┘
                    ↓  [reticulate]
┌────────────────────────────────────┐      ┌─────────────────────────┐
│  Layer 4 — R Risk Engine  [ORANGE] │ ───► │ metrics.csv             │
│  PortfolioAnalytics · PerformanceA │      │ weights_history.csv     │
└────────────────────────────────────┘      └─────────────────────────┘
                    ↓
┌────────────────────────────────────┐      ┌─────────────────────────┐
│  Layer 5 — Dashboard & Report[GRN] │ ───► │ Interactive Dashboard   │
│  Streamlit · R Shiny · R Markdown  │      │ HTML/PDF Report         │
└────────────────────────────────────┘      └─────────────────────────┘
```

The "reticulate" label appears as a small orange badge on the arrow between Layer 3 and Layer 4.
Layer 4 box uses orange border + light orange fill. Layer 5 uses green.

**Speaker notes:**
> "The critical handoff is between Layer 3 and Layer 4. Python trains the neural network and produces raw portfolio weights. R takes those weights, applies real financial constraints, runs a realistic backtest, and produces everything the dashboard reads."

---

### SLIDE 4 — Data Flow End to End

**Layout:** Navy header. Full-slide node-and-edge diagram (matplotlib PNG).

**Header:** Data Flow — End to End
**Subheader:** From raw Yahoo Finance data to an interactive dashboard

**Diagram spec (horizontal flow, 3 columns: Python left, R right, dashed boundary):**

```
[config.yaml]
      │
      ├──► [data_loader.py] ──► [features.py] ──┐
      │     yfinance OHLCV     30 tech features  │
      │                                          ▼
      └──► [scraper.py] ──► [sentiment.py] ──► [train.py] ──► [backtest.R] ──► [Dashboard]
            AV headlines     FinBERT scores   LSTM+DSR        walk-forward         Streamlit
                                                               R engine             Shiny
                                                                    │
                                                                    └──► [Report]
                                                                          R Markdown
```

Dashed orange vertical line separates Python (left) from R (right).
Label at bottom: "Python ←————————————→ R"

**Speaker notes:**
> "Every arrow represents a file or a function call. Nothing is hardwired. config.yaml is the single source of truth — change the tickers or dates there and the whole pipeline adapts."

---

### SLIDE 5 — Layer 1: Data Ingestion

**Layout:** Blue header. Three-column layout with coloured header boxes.

**Header:** Layer 1 — Data Ingestion
**Subheader:** Pulling price history and news headlines from the web

**Column 1 — "Prices (yfinance)" [blue header]:**
- Daily OHLCV for 5 tickers
- 2018-01-01 → 2024-12-31
- Saved as `prices.parquet`
- Delta-updated for live mode

**Column 2 — "News Headlines (Alpha Vantage)" [orange header]:**
- Historical articles per ticker
- Free API — 25 req/day
- Saved as `headlines.csv`
- Also yfinance recent news

**Column 3 — "config.yaml" [navy header]:**
- Asset universe
- Date range
- Hyper-parameters
- Constraint limits

**Footer note (grey italic):**
> "Why parquet? Columnar format → 10× faster than CSV for financial time-series. Delta updates append only new rows → no re-downloading 6 years of data every refresh."

---

### SLIDE 6 — Layer 2: Feature Engineering

**Layout:** Dark grey header. Full-width feature table.

**Header:** Layer 2 — Feature Engineering
**Subheader:** Turning raw prices into a rich 3D tensor the LSTM can read

**Table (7 rows, navy header row):**

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| Log Return | ln(P_t / P_{t-1}) | Daily price momentum |
| Rolling Vol | 20d std × √252 | Annualised volatility |
| RSI-14 | Relative Strength Index | Overbought / oversold |
| MACD | EMA(12) − EMA(26) signal | Trend direction |
| Bollinger %B | (Price − lower) / (upper − lower) | Band position |
| SMA Ratio | SMA(50) / SMA(200) | Golden/death cross |
| Sentiment * | P(pos) − P(neg) via FinBERT | News alpha signal |

**Footer note:**
> "* Sentiment is the 7th feature — optional, enabled via config.yaml. Output tensor: (batch, 60 days, 30 or 35 features) · Z-score normalised with 252-day rolling window."

---

### SLIDE 7 — Layer 3: LSTM + Differential Sharpe Ratio

**Layout:** Navy header. LSTM architecture diagram (top), explanatory text (bottom).

**Header:** Layer 3 — LSTM + Differential Sharpe Ratio
**Subheader:** The core ML engine — trained to maximise risk-adjusted returns directly

**Diagram (horizontal block flow):**
```
Input              LSTM Layer 1      LSTM Layer 2     Linear      Softmax
(batch, 60, 30)  → hidden=128     → hidden=64     → 64→5      → weights ∈ ℝ⁵
6 feat × 5 assets  dropout=0.3      dropout=0.3                  sum = 1
```
Each block is a rounded rectangle. Arrows between blocks. Softmax in green.

**Caption below diagram (orange italic):**
> "← Trained with Differential Sharpe Ratio (DSR) Loss — not cross-entropy! →"
> "η = 0.01  |  Adam optimiser  |  Early stopping on val DSR"

**Section: "What makes this different from regular deep learning?"**
Body text:
> "Normal neural nets minimise prediction error (MSE). This model minimises −DSR: the negative Differential Sharpe Ratio. That means it directly learns to maximise risk-adjusted profit — no intermediate step."

---

### SLIDE 8 — Layer 5: Dashboard & Portfolio Builder

**Layout:** Green header. Two-column layout (Streamlit left, R Shiny right).

**Header:** Layer 5 — Dashboard & Reporting
**Subheader:** Two interfaces: validated backtest results + live portfolio builder

**Left column — "Streamlit Dashboard" [green header]:**
Tabs:
- 📊 Portfolio Overview — equity curve
- ⚖️ Weight Allocation — stacked area
- 📈 Analytics — drawdown, rolling Sharpe
- 🛡️ Risk Monitor — rolling VaR/CVaR
- 🔴 Live Data — current prices + LSTM reco
- 📰 Sentiment — FinBERT heatmap
- 📡 Real-Time Signals — RSI/MACD any ticker

**Right column — "Portfolio Builder" [blue header] (new):**
- Select 2–10 stocks from 40-stock universe
- If 5 trained stocks → LSTM model used directly
- Otherwise → MPT optimizer + technical signal tilt
- R's PerformanceAnalytics computes all 8 metrics
- Pie chart, risk metrics, 1-year performance chart

**Footer:**
> "Both apps are standalone. `streamlit run dashboard.py` for historical results. `streamlit run portfolio_builder.py` for custom allocation."

---

### SLIDE 9 — Results: Bull Market (Dec 2023 – Dec 2024)

**Layout:** Navy background. Three metric cards side by side.

**Header (white):** Key Results — Bull Market Test (Dec 2023 – Dec 2024)

**Three cards:**

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  DSR Constrained    │  │  DSR Unconstrained   │  │  Equal Weight       │
│  [BLUE]             │  │  [PURPLE]            │  │  (benchmark)[GREY]  │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│    52.6%            │  │    69.1%             │  │    43.2%            │
│  ann. return        │  │  ann. return         │  │  ann. return        │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│  Sharpe  2.51       │  │  Sharpe  3.02        │  │  Sharpe  2.13       │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│  Max DD  14.3%      │  │  Max DD  12.9%       │  │  Max DD  14.4%      │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

**Summary line (light blue, centred):**
> "DSR Constrained beats Equal Weight by +9.4 pp annualised return with a higher Sharpe ratio and similar drawdown."

**Small print:**
> "Model trained: 2018–2024 · 5 assets · 60-day lookback · LSTM (128→64) · DSR loss (η=0.01) · PortfolioAnalytics constraints"

---

### SLIDE 10 — Results: Bear Market Stress Test (2022)

**Layout:** Navy background. Same three-card layout as Slide 9. Red accent for drawdown rows.

**Header (white):** Stress Test — Bear Market (2022, NASDAQ −33%)

**Context banner (orange):**
> "Model retrained on 2018–2021 only. 2022 never seen during training. NASDAQ dropped 33% this year — the hardest test for any tech portfolio."

**Three cards:**

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  DSR Constrained    │  │  DSR Unconstrained   │  │  Equal Weight       │
│  [BLUE]             │  │  [PURPLE]            │  │  (benchmark)[GREY]  │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│   −34.3%            │  │   −41.2%             │  │   −39.8%            │
│  ann. return        │  │  ann. return         │  │  ann. return        │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│  Sharpe  −1.06      │  │  Sharpe  −1.31       │  │  Sharpe  −1.17      │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│  Max DD  −34.0%     │  │  Max DD  −48.1%      │  │  Max DD  −42.3%     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

**Key insight (light blue):**
> "In a bear market, the constrained model loses LESS than the benchmark (−34.3% vs −39.8%). The unconstrained model, allowed to concentrate 65% in META (which fell −64%), performs worst."

**Takeaway line:**
> "The CVaR constraint enforced by R's PortfolioAnalytics is the reason the constrained model outperforms in both bull AND bear conditions."

---

### SLIDE 11 — Why R + Python Together?

**Layout:** Navy header. Full-width comparison table.

**Header:** Why R + Python Together?
**Subheader:** Each language does what it's genuinely best at

**Table (8 rows):**

| Task | Python | R | Winner |
|------|--------|---|--------|
| Deep Learning / LSTM | ✅ PyTorch | ❌ keras (weaker) | Python |
| Feature Engineering | ✅ pandas, ta | ⚠️ xts (workable) | Python |
| Sentiment / NLP | ✅ HuggingFace | ❌ limited | Python |
| Portfolio Constraint Solving | ⚠️ cvxpy (manual) | ✅ PortfolioAnalytics | R |
| Risk Metrics (Sharpe/CVaR) | ⚠️ manual NumPy | ✅ PerformanceAnalytics | R |
| Time-Series Backtesting | ⚠️ manual loop | ✅ xts + endpoints() | R |
| Report Generation | ⚠️ Jupyter/nbconv | ✅ R Markdown | R |
| Interactive Dashboard | ✅ Streamlit/Dash | ✅ Shiny | Both |

Winner column coloured green for R wins, blue for Python wins.

---

## PART 2 — R DEEP DIVE (Slides 12–25)

> All slides in Part 2 use an orange header accent to signal "R territory".

---

### SLIDE 12 — R's Role: The Big Picture

**Layout:** Orange header. Left: bullet overview. Right: role summary card.

**Header:** R's Role in This Project
**Subheader:** R does the heavy quantitative lifting — not just the charts

**Left column bullets:**
- R is responsible for **all 9 result files** in `results/`
- Python produces the trained model weights
- R produces everything the dashboard reads
- `backtest.R` is 600 lines — the backbone of the experiment
- R packages used: `PortfolioAnalytics`, `PerformanceAnalytics`, `xts`, `reticulate`, `quantmod`, `arrow`, `yaml`

**Right card (orange fill, white text):**
```
R owns:
  ✓ Walk-forward backtesting
  ✓ CVaR-constrained optimization
  ✓ 8 risk metrics computation
  ✓ Efficient frontier generation
  ✓ Transaction cost modelling
  ✓ Report generation (R Markdown)
  ✓ Risk metrics for Portfolio Builder
```

**Speaker notes:**
> "When someone asks 'what did R do in this project?' the answer is: R did all the financial engineering. Python did the machine learning. If you removed R, you'd have model weights and nothing else. If you removed Python, you'd lose the LSTM but R could still run a classical backtest."

---

### SLIDE 13 — R Component Diagram

**Layout:** Orange header. Full-slide component diagram (matplotlib PNG, orange/blue theme).

**Header:** Layer 4 — R Risk Engine: Component Diagram
**Subheader:** How the R files relate to each other and to Python

**Diagram spec:**

```
                    ┌──────────────────────────────────┐
                    │         backtest.R  [ORANGE]      │
                    │    Walk-forward orchestrator       │
                    └────┬──────┬──────┬──────┬─────────┘
                         │      │      │      │     ↔ reticulate ↔ [predict.py BLUE]
            ┌────────────┘      │      │      └─────────────┐
            ▼                   ▼      ▼                     ▼
  ┌──────────────┐   ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐
  │constraints.R │   │transaction_  │  │ metrics.R    │  │efficient_frontier │
  │Min/max bounds│   │costs.R       │  │Sharpe·Sortino│  │.R                 │
  │CVaR optim.   │   │Turnover calc │  │VaR·CVaR·Cal  │  │Mean-var frontier  │
  │              │   │10 bps deduct │  │mar            │  │plot               │
  └──────────────┘   └──────────────┘  └──────┬───────┘  └───────────────────┘
                                               │ Saved outputs
                                               ▼
  ┌─────────────────┐  ┌──────────┐  ┌──────────────────┐  ┌──────────┐
  │weights_history  │  │metrics   │  │backtest_results  │  │*.png     │
  │.csv             │  │.csv      │  │.rds              │  │plots     │
  └─────────────────┘  └──────────┘  └──────────────────┘  └──────────┘
```

**Speaker notes:**
> "backtest.R is the orchestrator. It calls Python for LSTM weights via reticulate, then passes those weights to constraints.R for box and CVaR checking, to transaction_costs.R for cost deduction, and finally to metrics.R for performance measurement."

---

### SLIDE 14 — The reticulate Bridge

**Layout:** Orange header. Full-slide sequence diagram (matplotlib PNG).

**Header:** R ↔ Python Bridge: reticulate
**Subheader:** R calls the LSTM model directly — no files exchanged, no REST API needed

**Sequence diagram (3 lifelines: backtest.R | reticulate | predict.py):**

```
backtest.R (R)          reticulate (bridge)        predict.py (Python)
     │                        │                           │
     │─source_python('predict.py')──►│                    │
     │                        │                           │
  ╔══╪══ loop [each weekly rebalance date] ══════════════╗│
  ║  │─predict_at_date('2024-01-08')──►│                 ║│
  ║  │                        │─forward call────────────►║│
  ║  │                        │                    load_model() — if not cached
  ║  │                        │                    compute_features(prices)
  ║  │                        │                    lstm.forward(X)
  ║  │                        │◄─return np.array([w1…w5])║│
  ║  │◄──weights as R vector──│                          ║│
  ║  │                        │                          ║│
  ║  │  apply_constraints(weights)                       ║│
  ║  │  w · r_{t+1} → portfolio return                   ║│
  ╚══╪═══════════════════════════════════════════════════╝│
```

**Key fact box (orange):**
> "reticulate embeds a Python interpreter inside the R session. No files written, no REST API, no serialization overhead. Python functions become R functions."

**Code snippet (dark background):**
```r
library(reticulate)
source_python("python/predict.py")          # loads Python into R
weights <- predict_at_date("2024-01-08")    # calls Python from R
# weights is now a native R numeric vector
```

---

### SLIDE 15 — xts: R's Financial Time Series

**Layout:** Orange header. Left: explanation. Right: code examples.

**Header:** xts — R's Financial Time Series Engine
**Subheader:** Why xts instead of a regular R data frame?

**Left column — key properties:**
- `xts` (eXtensible Time Series) is the standard in R finance
- Every observation is **date-indexed** — no separate Date column to manage
- Arithmetic automatically aligns on dates — no manual merging
- `endpoints()` generates weekly/monthly rebalance dates in one line
- Subset by date range: `prices["2023-01-01/2023-12-31"]`
- Used by PortfolioAnalytics and PerformanceAnalytics natively

**Right column — code examples (dark boxes):**

```r
# Read prices.parquet into xts
library(arrow); library(xts)
prices_df <- read_parquet("data/raw/prices.parquet")
prices <- xts(prices_df[,-1], order.by = as.Date(prices_df$Date))

# Get weekly rebalance endpoints
ep <- endpoints(prices, on = "weeks")
rebalance_dates <- index(prices)[ep]

# Subset to test period
test_prices <- prices["2023-12-01/2024-12-31"]

# Date arithmetic is automatic
returns <- diff(log(prices$AAPL_Close))
```

**Why it matters:**
> "Without xts, you'd need to manually ensure every operation is date-aligned. In a walk-forward backtest with 150+ steps, one off-by-one date error corrupts the entire result. xts makes date alignment automatic."

---

### SLIDE 16 — Walk-Forward Backtesting in Detail

**Layout:** Orange header. 6-step flow diagram (top), explanation text + bullet points (bottom).

**Header:** R's Role: Walk-Forward Backtesting
**Subheader:** `backtest.R` — 600 lines, the backbone of the whole experiment

**Step diagram (6 horizontal boxes with arrows):**

```
[Step 1: Load Data] → [Step 2: Get Dates] → [Step 3: Predict] → [Step 4: Constrain] → [Step 5: Return] → [Step 6: Save]
Read prices.parquet   Weekly rebalance       Call Python:          Apply min/max          w_t · r_{t+1}     weights_history
via arrow package     dates (xts endpoints)  predict_at_date()     & CVaR bounds          minus 10 bps cost  metrics.csv .rds
```

Step 3 (Predict) coloured blue (Python call). Step 4 (Constrain) coloured orange (R logic).

**Explanation text:**
> "xts time series in R keeps everything perfectly date-aligned. `PerformanceAnalytics::Return.portfolio()` handles weight × return multiplication correctly across all 56 weekly rebalance points."

**Bullet points:**
- ~56 rebalance points over Dec 2023–Dec 2024 test window
- Each point: Python inference (~0.3s) + R constraint check + return calculation
- Total backtest runtime: ~2 minutes on CPU
- Walk-forward = no future data ever leaks into past decisions

**Speaker notes:**
> "Walk-forward is essential for honest evaluation. Many backtests cheat by fitting the model on the full period and then 'testing' on the same period. Here, at every rebalance point, the LSTM only sees data that would have been available at that point in time."

---

### SLIDE 17 — constraints.R: CVaR Deep Dive

**Layout:** Orange header. Flow diagram (top half). CVaR explanation (bottom half).

**Header:** R's Role: PortfolioAnalytics & CVaR Constraints
**Subheader:** Turning raw LSTM weights into regulated, risk-controlled positions

**Flow diagram (4 boxes + feedback arrow):**

```
                     CVaR violated → re-optimise
         ◄────────────────────────────────────┐
[LSTM Output    →  [Box Constraints  →  [CVaR Check      →  [Final Portfolio
 raw softmax       w_min = 2%            95% confidence       weights
 weights w_t]      w_max = 30%]          level]               ∑w = 1]

predict_at_date()   constrain_weights()   compute_portfolio_cvar()
```

**What is CVaR? (explanation box):**
```
VaR (95%):  The loss we'd expect to NOT exceed 95% of the time.
            Example: VaR = 2.2% means in 95% of weeks, we lose less than 2.2%.

CVaR (95%): The AVERAGE loss in the worst 5% of weeks.
            More conservative than VaR — captures tail risk.
            Example: CVaR = 3.0% means in bad weeks, we lose 3.0% on average.

R code:
  ES(returns, p = 0.95, method = "historical")   # CVaR = Expected Shortfall
```

**Why PortfolioAnalytics?**
> "It provides an ROI solver designed for portfolio constraints — min/max weights, CVaR limits, long-only, sum-to-one — all in one call. `scipy.optimize` can do it, but PortfolioAnalytics handles infeasible regions and binding constraints out of the box."

**Transaction cost note:**
> "Transaction cost: 10 bps × |Δw| applied at every rebalance. backtest.R · transaction_costs.R deducts this from each week's return."

---

### SLIDE 18 — PerformanceAnalytics: 8 Metrics

**Layout:** Orange header. Full-width table.

**Header:** R's Role: PerformanceAnalytics
**Subheader:** 8 industry-standard risk metrics computed in fewer than 10 lines of R

**Table (8 rows, orange header):**

| Metric | R Function (PerformanceAnalytics) | Plain English | Our Value (Constrained) |
|--------|----------------------------------|---------------|------------------------|
| Annualised Return | `Return.annualized(r, scale=252)` | How much did we make? | **52.6%** |
| Sharpe Ratio | `SharpeRatio.annualized(r, Rf=0)` | Return per unit of risk | **2.51** |
| Sortino Ratio | `SortinoRatio(r)` | Return per unit of downside risk | 0.20 |
| Max Drawdown | `maxDrawdown(r)` | Worst peak-to-trough loss | **14.3%** |
| Calmar Ratio | `Return / maxDrawdown(r)` | Return vs max pain | **3.68** |
| VaR (95%) | `VaR(r, p=0.95, method='historical')` | Daily loss at 95% confidence | 2.17% |
| CVaR / ES (95%) | `ES(r, p=0.95, method='historical')` | Expected loss beyond VaR | 2.99% |
| Annualised Volatility | `StdDev.annualized(r, scale=252)` | How bumpy was the ride? | 21.0% |

**Code block (dark background):**
```r
library(PerformanceAnalytics)
r <- portfolio_returns   # xts object of weekly portfolio returns

metrics <- list(
  ann_return  = Return.annualized(r, scale=252),
  sharpe      = SharpeRatio.annualized(r, Rf=0),
  sortino     = SortinoRatio(r),
  max_dd      = maxDrawdown(r),
  calmar      = Return.annualized(r, scale=252) / maxDrawdown(r),
  var_95      = VaR(r, p=0.95, method="historical"),
  cvar_95     = ES(r,  p=0.95, method="historical"),
  ann_vol     = StdDev.annualized(r, scale=252)
)
```

---

### SLIDE 19 — efficient_frontier.R

**Layout:** Orange header. Left: explanation. Right: diagram of efficient frontier concept.

**Header:** R's Role: Efficient Frontier
**Subheader:** Visualising the theoretical optimum for the asset universe

**Left column:**
- The efficient frontier shows all portfolios that maximise return for a given level of risk
- Computed using PortfolioAnalytics + 50 sample points
- Shows where our LSTM-constrained portfolio sits relative to the theoretical optimum
- Also shows the Equal Weight benchmark position

**Diagram spec (right column — scatter plot concept):**
```
Return (%)
    │                          ✦ Max Sharpe
    │                   ╭──── Efficient Frontier (curve)
    │               ╭───╯
    │          ╭────╯  ★ Our constrained portfolio
    │     ╭────╯
    │ ──╯       ● Equal Weight benchmark
    └──────────────────────── Risk (Volatility %)
```

**Code block:**
```r
pspec <- portfolio.spec(assets = tickers)
pspec <- add.constraint(pspec, type = "full_investment")
pspec <- add.constraint(pspec, type = "box", min=0.02, max=0.30)
pspec <- add.objective(pspec, type = "return", name = "mean")
pspec <- add.objective(pspec, type = "risk",   name = "StdDev")

ef <- create.EfficientFrontier(R = returns, portfolio = pspec,
                               type = "mean-StdDev", n.portfolios = 50)
chart.EfficientFrontier(ef, match.col = "StdDev", pch = 4)
```

---

### SLIDE 20 — transaction_costs.R Deep Dive

**Layout:** Orange header. Left: concept. Right: code + formula.

**Header:** R's Role: Transaction Cost Modelling
**Subheader:** Real portfolios don't rebalance for free

**Left column — why it matters:**
- Every real trade incurs bid-ask spread + broker commissions
- Without modelling costs, backtests look unrealistically good ("paper trading")
- 10 basis points (0.10%) per unit of weight change is a conservative estimate for large-cap equities
- Turnover = total absolute change in weights at each rebalance
- High-turnover strategies are penalised; stable strategies are rewarded

**Example calculation box:**
```
Week 1 weights: [30%, 25%, 20%, 15%, 10%]
Week 2 weights: [35%, 20%, 20%, 15%, 10%]

Change: |5%| + |5%| + 0 + 0 + 0 = 10% total turnover
Cost:   10% × 10 bps = 0.10% deducted from Week 2 return
```

**Code block (right column):**
```r
# transaction_costs.R
compute_transaction_cost <- function(w_new, w_old,
                                     cost_bps = 10) {
  turnover    <- sum(abs(w_new - w_old))
  cost_factor <- cost_bps / 10000          # bps → decimal
  return(turnover * cost_factor)
}

# In backtest.R — deducted each week
tc   <- compute_transaction_cost(w_t, w_prev)
r_net <- r_gross - tc
```

---

### SLIDE 21 — The Bear Market Test: R's Proof of Value

**Layout:** Orange header. Split: context left, results right.

**Header:** The Bear Market Stress Test — R's Constraints Save the Portfolio
**Subheader:** 2022: NASDAQ −33%. What happened to our model?

**Context left column:**
- Model retrained on 2018–2021 only — 2022 never seen during training
- NASDAQ dropped 33% in 2022
- META fell 64%, AMZN fell 50%, GOOGL fell 39%
- Unconstrained LSTM concentrated up to 65% in META
- PortfolioAnalytics CVaR constraint capped META at 30%

**Results right column (two comparison cards):**

```
Without R constraints (unconstrained):    With R constraints:
  Ann. Return:  −41.2%                      Ann. Return:  −34.3%
  Max DD:       −48.1%                      Max DD:       −34.0%
  Benchmark EW: −39.8%                      Benchmark EW: −39.8%

  → Worse than benchmark                    → BEATS benchmark by +5.5pp
  → Concentrated in worst performer         → CVaR cap prevented META overweight
```

**Key insight (orange box):**
> "The CVaR constraint enforced by PortfolioAnalytics is the difference between beating and losing to the benchmark in a crash. R's risk management is what makes this a production-ready system."

---

### SLIDE 22 — R Markdown: The Report

**Layout:** Orange header. Left: explanation. Right: report structure.

**Header:** R's Role: Report Generation with R Markdown
**Subheader:** Code, output, and narrative in a single reproducible document

**Left column:**
- `R/report.Rmd` generates a complete HTML or PDF document
- Reads pre-computed `backtest_results.rds` — no re-running the model
- Combines: methodology explanation + charts + metrics table + conclusions
- One command: `rmarkdown::render("R/report.Rmd", output_format="html_document")`
- The grader/examiner reads the HTML without running any code
- Code chunks are shown alongside their output — fully transparent

**Code block:**
```r
---
title: "DSR Portfolio Optimizer — Backtest Report"
output: html_document
---

```{r setup, include=FALSE}
load("results/backtest_results.rds")
library(PerformanceAnalytics); library(ggplot2)
```

## Performance Summary
```{r metrics-table, echo=FALSE}
knitr::kable(metrics_df, digits=4,
             caption="PerformanceAnalytics Risk Metrics")
```
```

**Right column — report sections:**
1. Executive Summary
2. Methodology (LSTM + DSR + reticulate)
3. Walk-Forward Backtest Explanation
4. Equity Curve Chart
5. Weight Allocation Over Time
6. Efficient Frontier
7. 8 Risk Metrics Table
8. Conclusions + Limitations

---

### SLIDE 23 — Portfolio Builder: R's New Role

**Layout:** Orange header. Left: architecture. Right: code.

**Header:** The Portfolio Builder — R's Extended Role
**Subheader:** R validates risk for ANY stock selection, not just the 5 trained assets

**Left column — how it works:**
- User selects 2–10 stocks in `portfolio_builder.py`
- Python runs MPT optimizer (or LSTM for the 5 trained stocks)
- Python saves daily returns + optimal weights to `/tmp/`
- Python calls `Rscript R/portfolio_risk.R` via subprocess
- R reads the files, computes 8 metrics via PerformanceAnalytics
- R saves `/tmp/pb_metrics.csv`
- Python reads and displays the metrics in Streamlit

**Architecture note:**
> "Same separation as the main pipeline: Python optimizes, R validates. The bridge is subprocess instead of reticulate — but the principle is identical."

**Code block (R/portfolio_risk.R key lines):**
```r
library(PerformanceAnalytics); library(xts)

returns_raw <- read.csv("/tmp/pb_returns.csv", row.names=1)
weights_raw <- read.csv("/tmp/pb_weights.csv")

ret_xts  <- xts(as.matrix(returns_raw), order.by=as.Date(rownames(returns_raw)))
port_ret <- Return.portfolio(ret_xts, weights=as.numeric(weights_raw[1,]))

metrics <- data.frame(
  Metric = c("Ann. Return","Sharpe","Max Drawdown","CVaR (95%)", ...),
  Value  = c(Return.annualized(port_ret, scale=252),
             SharpeRatio.annualized(port_ret, Rf=0),
             maxDrawdown(port_ret),
             ES(port_ret, p=0.95, method="historical"), ...)
)
write.csv(metrics, "/tmp/pb_metrics.csv", row.names=FALSE)
```

---

### SLIDE 24 — Why R for This — Not Python?

**Layout:** Orange header. Side-by-side code comparison (Python manual vs R one-liner).

**Header:** Why R and Not Just Python for the Risk Layer?
**Subheader:** The same result takes 10× more code in Python — and is more likely to be wrong

**Two-column code comparison:**

**Left — Python (manual NumPy):**
```python
# Sharpe Ratio
mean_ret = returns.mean() * 252
std_ret  = returns.std() * np.sqrt(252)
sharpe   = mean_ret / std_ret

# Sortino (downside only)
neg  = returns[returns < 0]
sort = mean_ret / (neg.std() * np.sqrt(252))

# Max Drawdown
cum     = (1 + returns).cumprod()
rolling = cum.cummax()
dd      = (cum - rolling) / rolling
max_dd  = dd.min()

# CVaR
var  = np.percentile(returns, 5)
cvar = returns[returns <= var].mean()

# Calmar
calmar = mean_ret / abs(max_dd)
# ... 5 more metrics
# Total: ~40 lines, edge cases not handled
```

**Right — R (PerformanceAnalytics):**
```r
library(PerformanceAnalytics)

# All 8 metrics:
Return.annualized(r, scale=252)
SharpeRatio.annualized(r, Rf=0)
SortinoRatio(r)
maxDrawdown(r)
Return.annualized(r)/maxDrawdown(r)
VaR(r, p=0.95, method="historical")
ES(r,  p=0.95, method="historical")
StdDev.annualized(r, scale=252)

# Total: 8 lines
# Edge cases (empty periods, NA handling,
# scaling corrections): handled internally
```

**Verdict box (green):**
> "R's financial packages represent decades of academic and industry validation. VaR and CVaR in PerformanceAnalytics handle edge cases (empty tail distributions, scaling for non-daily data) that a manual NumPy implementation would silently get wrong."

---

### SLIDE 25 — Conclusion + Thank You

**Layout:** Navy background, orange top/bottom bars. Results summary + closing.

**Header (white, large):** Key Takeaways

**Three takeaway cards (side by side):**

```
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│  The Model Works       │  │  R Is Not Optional      │  │  The Architecture      │
│  [BLUE]                │  │  [ORANGE]               │  │  Scales  [GREEN]       │
│                        │  │                         │  │                        │
│  Beats benchmark in    │  │  CVaR constraints are   │  │  Portfolio Builder     │
│  BOTH bull (+9.4pp)    │  │  what separate          │  │  extends MPT to any    │
│  AND bear (+5.5pp)     │  │  bull from bear.        │  │  10-stock universe     │
│  markets on Sharpe     │  │  PortfolioAnalytics     │  │  R validates risk      │
│  and drawdown.         │  │  enforces them.         │  │  regardless of method. │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘
```

**Model footnote (grey, small):**
> "Model trained: 2018–2024 · 5 assets · 60-day lookback · LSTM (128→64) · DSR loss (η=0.01) · PortfolioAnalytics CVaR constraints · PerformanceAnalytics metrics"

**Closing line (white, bold, large, centred):**
```
Thanks for listening   —   Questions?
```

---

## Appendix: Generation Instructions

### If using `create_presentation.py` (python-pptx):

Each slide above maps to a function: `slide_N_name(prs)`.
Use `_section_header(sld, title, subtitle, accent=ORANGE)` for Part 2 slides.
Use `_png_to_slide()` for all diagram slides.
Colour constants: `NAVY`, `ORANGE`, `BLUE`, `GREEN`, `RED`, `GREY`, `WHITE`, `LIGHT`.

### If using an AI (ChatGPT / Claude):

Paste the slide spec verbatim. Request:
> "Generate a python-pptx script that creates this presentation exactly. Use the design spec at the top. Each slide should be a separate function."

### Slide count verification:

| Part | Slides | Count |
|------|--------|-------|
| Part 1 — Overview | 1–11 | 11 |
| Part 2 — R Deep Dive | 12–25 | 14 |
| **Total** | | **25** |
