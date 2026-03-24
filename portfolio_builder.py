"""
Portfolio Builder — Standalone Streamlit App
=============================================
Select up to 10 stocks and get an optimal allocation.

- If you pick the 5 trained stocks (AAPL, MSFT, GOOGL, AMZN, META):
    uses the LSTM model directly (our trained model)
- Any other selection:
    uses Mean-Variance Optimization (MPT) with technical signal tilting

R's PerformanceAnalytics computes all 8 risk metrics in both cases.

Run:  streamlit run portfolio_builder.py
"""

import sys
import subprocess
import warnings
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent
PYTHON = ROOT / "python"
sys.path.insert(0, str(PYTHON))

BACKTEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

STOCK_UNIVERSE = {
    "🖥️ Tech":        ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                       "NVDA", "TSLA", "AMD", "INTC", "ORCL", "CRM", "ADBE"],
    "🏦 Finance":     ["JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BRK-B"],
    "🏥 Healthcare":  ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO"],
    "🛒 Consumer":    ["WMT", "COST", "MCD", "SBUX", "NKE", "DIS", "AMZN"],
    "⚡ Energy":      ["XOM", "CVX", "COP", "SLB"],
    "📦 ETFs":        ["SPY", "QQQ", "IWM", "VTI", "GLD"],
}

ALL_STOCKS = sorted({s for group in STOCK_UNIVERSE.values() for s in group})

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Builder",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }
    .badge-lstm  { background:#1a6bb5; color:white; padding:4px 12px;
                   border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-mpt   { background:#e67e22; color:white; padding:4px 12px;
                   border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-r     { background:#1e8b4c; color:white; padding:4px 12px;
                   border-radius:20px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_weight_constraints(weights, w_min=0.02, w_max=0.30, max_iter=300):
    """Project weights onto [w_min, w_max] simplex (same as dashboard.py)."""
    w = np.array(weights, dtype=float)
    for _ in range(max_iter):
        w = np.clip(w, w_min, w_max)
        s = w.sum()
        if s > 0:
            w /= s
        if np.all(w >= w_min - 1e-9) and np.all(w <= w_max + 1e-9):
            break
    return w


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download daily close prices for tickers. Returns DataFrame (dates × tickers)."""
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, timeout=20)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if len(tickers) == 1 else raw
    close.columns = [str(c) for c in close.columns]
    close = close.dropna(how="all")
    return close


def mpt_optimize(log_returns: pd.DataFrame,
                 signal_scores: dict,
                 w_min=0.02, w_max=0.30) -> np.ndarray:
    """
    Maximize Sharpe Ratio using scipy.
    Blends annualised historical returns (70%) with
    normalised technical signal tilt (30%).
    """
    tickers = list(log_returns.columns)
    n = len(tickers)

    # Relax w_max when n is small (e.g. 3 stocks × 30% = 90% < 100%)
    w_max = max(w_max, 1.0 / n)

    mu    = log_returns.mean().values * 252
    sigma = log_returns.cov().values  * 252

    # Normalise mu to [0,1]
    mu_range = mu.max() - mu.min()
    mu_norm  = (mu - mu.min()) / mu_range if mu_range > 1e-9 else np.full(n, 1/n)

    # Technical signal tilt: scores shifted to [0,8], then normalised
    raw_scores = np.array([signal_scores.get(t, 0) + 4 for t in tickers], dtype=float)
    tilt = raw_scores / raw_scores.sum() if raw_scores.sum() > 0 else np.full(n, 1/n)

    mu_tilted = 0.70 * mu_norm + 0.30 * tilt

    def neg_sharpe(w):
        port_ret = w @ mu_tilted
        port_vol = np.sqrt(w @ sigma @ w)
        return -port_ret / port_vol if port_vol > 1e-9 else 0.0

    w0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(w_min, w_max)] * n

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})

    w_opt = result.x if result.success else w0
    return apply_weight_constraints(w_opt, w_min, w_max)


def lstm_weights(today_str: str) -> np.ndarray:
    """Call LSTM predict_at_date for the 5 trained stocks."""
    from predict import clear_cache, predict_at_date
    clear_cache()
    raw = np.array(predict_at_date(today_str), dtype=float)
    return apply_weight_constraints(raw)


def run_r_metrics(log_returns: pd.DataFrame,
                  weights: np.ndarray,
                  tickers: list[str]) -> pd.DataFrame | None:
    """
    Save inputs to /tmp, call R/portfolio_risk.R via subprocess,
    read back 8 metrics. Returns None if R unavailable.
    """
    try:
        # Save returns (only selected tickers)
        ret_sub = log_returns[tickers].dropna()
        ret_sub.to_csv("/tmp/pb_returns.csv")

        w_df = pd.DataFrame([weights], columns=tickers)
        w_df.to_csv("/tmp/pb_weights.csv", index=False)

        r_script = str(ROOT / "R" / "portfolio_risk.R")
        result = subprocess.run(
            ["Rscript", r_script],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            st.warning(f"R script error: {result.stderr[:300]}")
            return None

        metrics = pd.read_csv("/tmp/pb_metrics.csv")
        return metrics
    except Exception:
        return None


def python_metrics(log_returns: pd.DataFrame,
                   weights: np.ndarray,
                   tickers: list[str]) -> pd.DataFrame:
    """Fallback: compute 8 metrics in NumPy when R is unavailable."""
    ret = log_returns[tickers].dropna()
    port = ret.values @ weights  # daily portfolio returns

    ann_ret = port.mean() * 252
    ann_vol = port.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    neg      = port[port < 0]
    sortino  = (ann_ret / (neg.std() * np.sqrt(252))) if len(neg) > 0 else 0.0

    cum   = np.cumprod(1 + port)
    roll_max = np.maximum.accumulate(cum)
    dd    = (cum - roll_max) / roll_max
    max_dd = abs(dd.min())

    calmar  = ann_ret / max_dd if max_dd > 0 else 0.0
    var_95  = abs(np.percentile(port, 5))
    cvar_95 = abs(port[port <= -var_95].mean()) if (port <= -var_95).any() else var_95

    return pd.DataFrame({
        "Metric": ["Ann. Return", "Ann. Volatility", "Sharpe Ratio",
                   "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
                   "VaR (95%)", "CVaR / ES (95%)"],
        "Value":  [ann_ret, ann_vol, sharpe, sortino,
                   max_dd, calmar, var_95, cvar_95],
    })


# ══════════════════════════════════════════════════════════════════════════════
# UI — PHASE 1: STOCK SELECTION
# ══════════════════════════════════════════════════════════════════════════════

st.title("📐 Portfolio Builder")
st.markdown(
    "Select **2–10 stocks**, click **Optimise**, and get an AI-powered allocation. "
    "Picks from our 5 trained stocks use the **LSTM model**. "
    "Any other selection uses **Mean-Variance Optimisation** with technical signal tilting. "
    "**R's PerformanceAnalytics** computes all risk metrics either way."
)

st.markdown("---")

# ── Sector tabs for quick selection ───────────────────────────────────────────
st.subheader("Step 1 — Choose your stocks")

col_sel, col_info = st.columns([3, 1])

with col_sel:
    selected = st.multiselect(
        "Search or select stocks (2–10):",
        options=ALL_STOCKS,
        default=["AAPL", "MSFT", "NVDA"],
        max_selections=10,
        help="Type any ticker symbol to search. Max 10 stocks.",
    )

with col_info:
    st.markdown("**Quick picks by sector:**")
    for sector, tickers in STOCK_UNIVERSE.items():
        with st.expander(sector, expanded=False):
            for t in tickers:
                st.caption(t)

# Custom ticker input
custom_input = st.text_input(
    "Add a custom ticker (e.g. PLTR, COIN, TSM):",
    placeholder="Type ticker and press Enter",
)
if custom_input:
    custom_ticker = custom_input.strip().upper()
    if custom_ticker not in selected:
        if len(selected) < 10:
            selected = selected + [custom_ticker]
            st.success(f"Added {custom_ticker}")
        else:
            st.error("Maximum 10 stocks reached.")

# Validation
if len(selected) < 2:
    st.warning("Please select at least 2 stocks to build a portfolio.")
    st.stop()

# Method badge
is_lstm = set(selected) == set(BACKTEST_TICKERS)
if is_lstm:
    st.markdown(
        '<span class="badge-lstm">🧠 LSTM Model</span> '
        'You selected all 5 trained stocks — using the LSTM directly.',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<span class="badge-mpt">📊 MPT + Signals</span> '
        'Using Mean-Variance Optimisation with technical signal tilting.',
        unsafe_allow_html=True
    )
st.markdown('<span class="badge-r">🔬 R PerformanceAnalytics</span> Risk metrics computed by R.', unsafe_allow_html=True)

st.markdown("---")
optimise_btn = st.button("⚡ Optimise Portfolio", type="primary", use_container_width=True)

if not optimise_btn:
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# UI — PHASE 2: OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Fetching data and optimising..."):

    # 1. Fetch price data
    close = fetch_prices(selected, period="1y")
    valid_tickers = [t for t in selected if t in close.columns and close[t].notna().sum() > 60]

    if len(valid_tickers) < 2:
        st.error("Could not fetch enough data for the selected stocks. Check ticker symbols.")
        st.stop()

    if len(valid_tickers) < len(selected):
        missing = set(selected) - set(valid_tickers)
        st.warning(f"No data found for: {', '.join(missing)}. Proceeding with {valid_tickers}.")

    close = close[valid_tickers].dropna()
    log_ret = np.log(close / close.shift(1)).dropna()

    today_str = str(date.today())

    # 2. Compute technical signals (works for any ticker)
    try:
        from signals import get_signals
        sig_df = get_signals(valid_tickers, period="1y")
        signal_scores = sig_df["score"].to_dict()
        signal_labels = sig_df["signal"].to_dict()
        signals_available = True
    except Exception:
        signal_scores  = {t: 0 for t in valid_tickers}
        signal_labels  = {t: "⚪ HOLD" for t in valid_tickers}
        signals_available = False

    # 3. Optimize
    n_stocks = len(valid_tickers)
    eff_w_max = max(0.30, 1.0 / n_stocks)   # relax cap when n is small
    if is_lstm and set(valid_tickers) == set(BACKTEST_TICKERS):
        try:
            weights = lstm_weights(today_str)
            method  = "LSTM"
        except Exception as e:
            st.warning(f"LSTM prediction failed ({e}). Falling back to MPT.")
            weights = mpt_optimize(log_ret, signal_scores, w_max=eff_w_max)
            method  = "MPT"
    else:
        weights = mpt_optimize(log_ret, signal_scores, w_max=eff_w_max)
        method  = "MPT"

    # 4. R risk metrics (with Python fallback)
    r_metrics = run_r_metrics(log_ret, weights, valid_tickers)
    if r_metrics is not None:
        metrics_df   = r_metrics
        metrics_src  = "R"
    else:
        metrics_df   = python_metrics(log_ret, weights, valid_tickers)
        metrics_src  = "Python"


# ══════════════════════════════════════════════════════════════════════════════
# UI — PHASE 3: RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.success(f"Optimisation complete — {method} + {metrics_src} risk metrics")

st.subheader("Step 2 — Your Optimal Portfolio")

# ── Row 1: Pie chart + allocation table ───────────────────────────────────────
col_pie, col_table = st.columns([1, 1])

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=valid_tickers,
        values=weights,
        hole=0.45,
        textinfo="label+percent",
        marker_colors=px.colors.qualitative.Set2[:len(valid_tickers)],
        textfont_size=13,
    ))
    fig_pie.update_layout(
        title=f"Optimal Allocation ({method})",
        height=380,
        showlegend=False,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    alloc_df = pd.DataFrame({
        "Ticker":     valid_tickers,
        "Weight":     [f"{w:.1%}" for w in weights],
        "Signal":     [signal_labels.get(t, "—") for t in valid_tickers],
        "$ per $100": [f"${w*100:.2f}" for w in weights],
    })
    st.markdown("**Allocation breakdown**")
    st.dataframe(alloc_df, hide_index=True, use_container_width=True)

    # Plain English
    top2 = sorted(zip(weights, valid_tickers), reverse=True)[:2]
    st.info(
        f"The model recommends concentrating most in "
        f"**{top2[0][1]}** ({top2[0][0]:.1%}) and "
        f"**{top2[1][1]}** ({top2[1][0]:.1%}). "
        f"Every position is between 2% and 30% — same risk limits as our main backtest."
    )

st.markdown("---")

# ── Row 2: Risk metrics ────────────────────────────────────────────────────────
st.subheader(f"Risk Metrics — computed by {metrics_src}")
if metrics_src == "R":
    st.markdown('<span class="badge-r">🔬 R PerformanceAnalytics</span>', unsafe_allow_html=True)
else:
    st.caption("⚠️ R unavailable — metrics computed in Python (NumPy).")

pct_metrics = {"Ann. Return", "Ann. Volatility", "Max Drawdown", "VaR (95%)", "CVaR / ES (95%)"}
m_cols = st.columns(4)
for i, row in metrics_df.iterrows():
    val = row["Value"]
    label = row["Metric"]
    if label in pct_metrics:
        display = f"{val:.2%}"
    else:
        display = f"{val:.3f}" if pd.notna(val) else "—"
    with m_cols[i % 4]:
        st.metric(label=label, value=display)

st.markdown("---")

# ── Row 3: Signal breakdown ────────────────────────────────────────────────────
if signals_available:
    st.subheader("Technical Signal Breakdown")
    st.caption("Rule-based signals (RSI, MACD, SMA50/200, Bollinger %B, 5d momentum) — same engine as Real-Time Signals tab in the main dashboard.")

    try:
        sig_display = sig_df.loc[valid_tickers, [
            "signal", "score", "current_price", "change_pct_1d",
            "rsi", "bb_pct_b", "momentum_5d"
        ]].copy()
        sig_display.columns = ["Signal", "Score", "Price ($)", "1d Change", "RSI", "BB %B", "5d Mom"]
        sig_display["Price ($)"]  = sig_display["Price ($)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
        sig_display["1d Change"]  = sig_display["1d Change"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
        sig_display["RSI"]        = sig_display["RSI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        sig_display["BB %B"]      = sig_display["BB %B"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        sig_display["5d Mom"]     = sig_display["5d Mom"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
        st.dataframe(sig_display, use_container_width=True)
    except Exception:
        pass

    st.markdown("---")

# ── Row 4: Historical performance chart ───────────────────────────────────────
st.subheader("Hypothetical Portfolio Performance — Last 1 Year")
st.caption("What $100 would have become if you'd held these weights for the past year (for illustration only — past performance ≠ future results).")

port_cum = (log_ret[valid_tickers] @ weights).cumsum().apply(np.exp) * 100

fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(
    x=port_cum.index, y=port_cum.values,
    name="Your Portfolio", line=dict(color="#2e86c1", width=2.5),
))

# Add individual stocks as thin lines
palette = px.colors.qualitative.Pastel
for i, t in enumerate(valid_tickers):
    cum_t = log_ret[t].cumsum().apply(np.exp) * 100
    fig_perf.add_trace(go.Scatter(
        x=cum_t.index, y=cum_t.values,
        name=t, line=dict(width=1, dash="dot"),
        marker_color=palette[i % len(palette)],
        opacity=0.7,
    ))

fig_perf.add_hline(y=100, line_dash="dash", line_color="grey", opacity=0.5)
fig_perf.update_layout(
    height=380, template="plotly_dark",
    yaxis_title="Value ($, starting at $100)",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(t=40, b=20),
)
st.plotly_chart(fig_perf, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"**Methodology:** {'LSTM (DSR-trained, 5-asset model)' if method == 'LSTM' else 'Mean-Variance Optimisation (Markowitz MPT) + technical signal tilt (30% blend)'}  ·  "
    f"**Risk metrics:** {metrics_src} {'PerformanceAnalytics' if metrics_src == 'R' else '(NumPy fallback)'}  ·  "
    f"**Constraints:** min 2% · max 30% per position  ·  "
    f"Data via yfinance · {today_str}"
)
