"""
DSR Portfolio Optimizer — Streamlit Dashboard
Mirrors the Shiny app using pre-computed results from results/
"""

import streamlit as st

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import yaml
from datetime import datetime as _dt_cls

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _AUTOREFRESH_AVAILABLE = True
except ImportError:
    _AUTOREFRESH_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSR Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
# Fixed model-trained tickers — LSTM checkpoint is locked to these 5 assets.
BACKTEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# ── Python subdirectory on path (for live predict imports) ────────────────────
import sys as _sys
_PYTHON_DIR = str(ROOT / "python")
if _PYTHON_DIR not in _sys.path:
    _sys.path.insert(0, _PYTHON_DIR)

# ── Market status ─────────────────────────────────────────────────────────────
def market_status() -> dict:
    """Return US equity market open/closed status based on Eastern Time."""
    import zoneinfo
    from datetime import time as _time
    ET = zoneinfo.ZoneInfo("America/New_York")
    now = _dt_cls.now(tz=ET)
    t = now.time()
    is_open = (now.weekday() < 5) and (_time(9, 30) <= t < _time(16, 0))
    if is_open:
        close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
        delta = int((close_dt - now).total_seconds())
        h, m = divmod(delta // 60, 60)
        detail = f"Closes in {h}h {m:02d}m"
    else:
        days = 1
        while True:
            cand = now + pd.Timedelta(days=days)
            if cand.weekday() < 5:
                open_dt = cand.replace(hour=9, minute=30, second=0, microsecond=0)
                if open_dt > now:
                    break
            days += 1
        delta = int((open_dt - now).total_seconds())
        h, m = divmod(delta // 60, 60)
        detail = f"Opens in {h}h {m:02d}m"
    return {"is_open": is_open, "label": "OPEN" if is_open else "CLOSED",
            "detail": detail, "now_et": now}


# ── Live data helpers ─────────────────────────────────────────────────────────
def fetch_and_append_prices(tickers=None):
    """
    Download OHLCV from the day after the current parquet end-date to today.
    Appends new rows (and any new ticker columns) to prices.parquet in-place.
    Returns (close_prices_df, last_date_str) for the requested tickers.
    """
    if tickers is None:
        tickers = BACKTEST_TICKERS

    import yfinance as yf
    from datetime import date, timedelta

    parquet_path = ROOT / "data/raw/prices.parquet"
    existing = pd.read_parquet(parquet_path)
    existing.index = pd.to_datetime(existing.index)

    last_existing = existing.index.max().date()
    fetch_start = last_existing + timedelta(days=1)
    fetch_end = date.today() + timedelta(days=1)   # yfinance end is exclusive

    if fetch_start >= fetch_end:
        available = [t for t in tickers if f"{t}_Close" in existing.columns]
        close = existing[[f"{t}_Close" for t in available]].copy()
        close.columns = available
        return close, str(last_existing)

    raw = yf.download(
        tickers=tickers,
        start=str(fetch_start),
        end=str(fetch_end),
        auto_adjust=True,
        progress=False,
        timeout=20,
    )

    if raw is None or raw.empty:
        available = [t for t in tickers if f"{t}_Close" in existing.columns]
        close = existing[[f"{t}_Close" for t in available]].copy()
        close.columns = available
        return close, str(last_existing)

    # Flatten MultiIndex: (price_type, ticker) → "AAPL_Close"
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [f"{ticker}_{price_type}" for price_type, ticker in raw.columns]
    raw.index = pd.to_datetime(raw.index)

    # Concat without restricting columns — new tickers get NaN for old rows
    combined = pd.concat([existing, raw])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(parquet_path, engine="pyarrow")

    last_date = combined.index.max().date()
    available = [t for t in tickers if f"{t}_Close" in combined.columns
                 and combined[f"{t}_Close"].notna().any()]
    close = combined[[f"{t}_Close" for t in available]].copy()
    close.columns = available
    close.index = pd.to_datetime(close.index)
    return close, str(last_date)


def apply_weight_constraints(weights, w_min=0.02, w_max=0.30, max_iter=200):
    """Project softmax weights onto [w_min, w_max] simplex via iterative clipping."""
    w = np.array(weights, dtype=float)
    for _ in range(max_iter):
        w = np.clip(w, w_min, w_max)
        total = w.sum()
        if total > 0:
            w /= total
        if np.all(w >= w_min - 1e-9) and np.all(w <= w_max + 1e-9):
            break
    return w


def compute_live_weights(last_date_str):
    """Run LSTM prediction for last_date_str. Returns (raw_weights, constrained_weights)."""
    import predict
    predict.clear_cache()
    raw_w = np.array(predict.predict_at_date(last_date_str), dtype=float)
    return raw_w, apply_weight_constraints(raw_w)


@st.cache_data(ttl=300, show_spinner="Fetching live prices and running LSTM model...")
def load_live_data(fetch_key: str, user_tickers_key: str) -> dict:
    """Fetch live prices + LSTM weights.
    fetch_key busts cache on manual refresh.
    user_tickers_key (comma-joined) busts cache when watchlist changes.
    """
    user_tickers = user_tickers_key.split(",")
    result = {"close_prices": None, "last_date": None,
               "weights_raw": None, "weights_const": None,
               "user_tickers": user_tickers, "no_data_tickers": [],
               "error": None}
    try:
        close, last_date = fetch_and_append_prices(user_tickers)
        raw_w, con_w = compute_live_weights(last_date)
        no_data = [t for t in user_tickers if t not in close.columns]
        result.update(close_prices=close, last_date=last_date,
                      weights_raw=raw_w, weights_const=con_w,
                      user_tickers=list(close.columns),
                      no_data_tickers=no_data)
    except FileNotFoundError as exc:
        result["error"] = f"Model not found: {exc}. Run python/train.py first."
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ── Sentiment data loader ─────────────────────────────────────────────────────
@st.cache_data
def load_sentiment_data():
    """Load pre-computed FinBERT sentiment scores if available."""
    path = ROOT / "data/processed/sentiment_scores.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    weights_constrained = pd.read_csv(ROOT / "results/weights_history.csv", parse_dates=["Date"])
    weights_unconstrained = pd.read_csv(ROOT / "results/weights_unconstrained.csv", parse_dates=["Date"])

    prices = pd.read_parquet(ROOT / "data/raw/prices.parquet")
    tickers = BACKTEST_TICKERS
    close_prices = prices[[f"{t}_Close" for t in tickers]].copy()
    close_prices.columns = tickers
    close_prices.index = pd.to_datetime(close_prices.index)

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    return metrics, weights_constrained, weights_unconstrained, close_prices, tickers, config


@st.cache_data
def compute_portfolio_returns(weights_df, close_prices, tickers):
    """Compute daily portfolio returns from rebalance weights."""
    weights_df = weights_df.sort_values("Date").copy()
    close = close_prices.loc[weights_df["Date"].min():].copy()

    daily_returns = close[tickers].pct_change().fillna(0)

    portfolio_returns = pd.Series(0.0, index=daily_returns.index)
    prev_weights = None
    for i, row in weights_df.iterrows():
        date = row["Date"]
        w = row[tickers].values.astype(float)
        if prev_weights is None:
            prev_weights = (date, w)
            continue
        # Hold prev weights from prev_date+1 to this date
        prev_date, pw = prev_weights
        mask = (daily_returns.index > prev_date) & (daily_returns.index <= date)
        portfolio_returns[mask] = daily_returns.loc[mask, tickers].values @ pw
        prev_weights = (date, w)

    # Last segment
    if prev_weights:
        last_date, pw = prev_weights
        mask = daily_returns.index > last_date
        portfolio_returns[mask] = daily_returns.loc[mask, tickers].values @ pw

    portfolio_returns = portfolio_returns[portfolio_returns.index >= weights_df["Date"].min()]
    return portfolio_returns


@st.cache_data
def compute_equal_weight_returns(close_prices, tickers, start_date, end_date):
    close = close_prices.loc[start_date:end_date, tickers]
    returns = close.pct_change().fillna(0)
    return returns.mean(axis=1)


def equity_curve(returns):
    return (1 + returns).cumprod()


def rolling_sharpe(returns, window=126):
    mean_r = returns.rolling(window).mean() * 252
    std_r = returns.rolling(window).std() * np.sqrt(252)
    return (mean_r / std_r.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def max_drawdown_series(returns):
    cum = equity_curve(returns)
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def monthly_returns_table(returns):
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = monthly.to_frame("Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.strftime("%b")
    pivot = df.pivot(index="Year", columns="Month", values="Return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot.reindex(columns=[m for m in month_order if m in pivot.columns])


# ── Sidebar ───────────────────────────────────────────────────────────────────
metrics, wc, wu, close_prices, tickers, config = load_data()

STRATEGIES = {
    "DSR Constrained": "DSR_Constrained",
    "DSR Unconstrained": "DSR_Unconstrained",
    "Equal Weight": "EqualWeight",
}
COLORS = {
    "DSR Constrained": "#1f77b4",
    "DSR Unconstrained": "#ff7f0e",
    "Equal Weight": "#9467bd",
}

# ── Session state ─────────────────────────────────────────────────────────────
if "live_fetch_key" not in st.session_state:
    st.session_state.live_fetch_key = None
if "live_refresh_ts" not in st.session_state:
    st.session_state.live_refresh_ts = None
if "user_tickers" not in st.session_state:
    st.session_state.user_tickers = list(BACKTEST_TICKERS)

with st.sidebar:
    st.title("📈 DSR Portfolio Optimizer")
    st.markdown("---")

    selected = st.multiselect(
        "Strategies",
        list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
    )
    rolling_window = st.slider("Rolling window (days)", 21, 252, 126, step=21)

    st.markdown("---")
    st.markdown("**Assets:** AAPL · MSFT · GOOGL · AMZN · META")
    st.markdown("**Test period:** Dec 2023 – Dec 2024")
    st.markdown("**Rebalance:** Weekly · 10 bps costs")
    st.markdown("---")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    st.subheader("Auto-Refresh")
    if _AUTOREFRESH_AVAILABLE:
        _ar_label = st.selectbox(
            "Interval", ["Off", "1 min", "5 min", "15 min"],
            key="autorefresh_interval",
        )
        _ar_ms = {"Off": 0, "1 min": 60_000, "5 min": 300_000, "15 min": 900_000}
        if _ar_ms[_ar_label] > 0:
            _st_autorefresh(interval=_ar_ms[_ar_label], key="global_autorefresh")
            st.caption(f"Page reloads every {_ar_label}.")
        else:
            st.caption("Select an interval to enable.")
    else:
        st.caption("Run `pip install streamlit-autorefresh` to enable.")

    st.markdown("---")
    st.subheader("Live Data")
    if st.button("🔄 Refresh Live Data", type="primary", use_container_width=True):
        st.session_state.live_fetch_key = _dt_cls.now().isoformat()
        st.session_state.live_refresh_ts = _dt_cls.now()
        load_live_data.clear()
        st.rerun()
    if st.session_state.live_refresh_ts:
        st.caption(f"Last refreshed: {st.session_state.live_refresh_ts.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.caption("Click above to fetch live prices & model weights.")

    # ── Watchlist ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Watchlist (5–15 stocks)")
    st.caption("Price charts in Live Data tab show these stocks. LSTM model always uses the trained 5.")

    at_min = len(st.session_state.user_tickers) <= 5
    for t in list(st.session_state.user_tickers):
        col_t, col_x = st.columns([3, 1])
        col_t.write(t)
        if col_x.button(
            "✕", key=f"rm_{t}",
            disabled=at_min,
            help="Minimum 5 stocks" if at_min else f"Remove {t}",
        ):
            st.session_state.user_tickers.remove(t)
            load_live_data.clear()
            st.rerun()

    at_max = len(st.session_state.user_tickers) >= 15
    new_ticker = st.text_input(
        "Add ticker (e.g. NVDA)", key="new_ticker_input",
        disabled=at_max,
        placeholder="Disabled — 15 stock maximum" if at_max else "NVDA",
    )
    if st.button("+ Add to Watchlist", disabled=at_max or not new_ticker, use_container_width=True):
        t = new_ticker.strip().upper()
        if t and t not in st.session_state.user_tickers:
            st.session_state.user_tickers.append(t)
            load_live_data.clear()
            st.rerun()
        elif t in st.session_state.user_tickers:
            st.warning(f"{t} is already in the watchlist.")

    st.caption(f"{len(st.session_state.user_tickers)}/15 stocks")

    st.markdown("---")
    report_path = ROOT / "reports/report.html"
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button("📥 Download Full Report", f, "dsr_report.html", "text/html")

# ── Compute returns ───────────────────────────────────────────────────────────
returns_c = compute_portfolio_returns(wc, close_prices, tickers)
returns_u = compute_portfolio_returns(wu, close_prices, tickers)
start_date = max(wc["Date"].min(), wu["Date"].min())
end_date = wc["Date"].max()
returns_ew = compute_equal_weight_returns(close_prices, tickers, start_date, end_date)

all_returns = {
    "DSR Constrained": returns_c,
    "DSR Unconstrained": returns_u,
    "Equal Weight": returns_ew,
}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Portfolio Overview",
    "⚖️ Weight Allocation",
    "📈 Analytics",
    "🛡️ Risk Monitor",
    "🔴 Live Data",
    "📰 Sentiment",
    "📡 Real-Time Signals",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Portfolio Overview
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Portfolio Performance Overview")

    # Primary strategy = DSR Constrained
    primary = metrics[metrics["Strategy"] == "DSR_Constrained"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{primary['Ann_Return']:.1%}", delta="vs EW: " + f"{(primary['Ann_Return'] - metrics[metrics['Strategy']=='EqualWeight']['Ann_Return'].values[0]):.1%}")
    col2.metric("Sharpe Ratio", f"{primary['Sharpe_Ratio']:.2f}")
    col3.metric("Annualized Volatility", f"{primary['Ann_Volatility']:.1%}")
    col4.metric("Max Drawdown", f"-{primary['Max_Drawdown']:.1%}")

    st.markdown("---")

    # Equity curves
    fig = go.Figure()
    for name in selected:
        r = all_returns[name]
        eq = equity_curve(r)
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=name, line=dict(color=COLORS[name], width=2),
            hovertemplate="%{x|%Y-%m-%d}: $%{y:.3f}<extra></extra>"
        ))
    fig.update_layout(
        title="Cumulative Portfolio Value (Starting $1.00)",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Performance Metrics")
    display_cols = {
        "Strategy": "Strategy",
        "Ann_Return_Pct": "Ann. Return",
        "Ann_Volatility_Pct": "Volatility",
        "Sharpe_Ratio": "Sharpe",
        "Sortino_Ratio": "Sortino",
        "Max_Drawdown_Pct": "Max DD",
        "VaR_95_Pct": "VaR 95%",
        "CVaR_95_Pct": "CVaR 95%",
        "Calmar_Ratio": "Calmar",
    }
    tbl = metrics[list(display_cols.keys())].rename(columns=display_cols)
    tbl["Sharpe"] = tbl["Sharpe"].round(3)
    tbl["Sortino"] = tbl["Sortino"].round(3)
    tbl["Calmar"] = tbl["Calmar"].round(3)
    st.dataframe(tbl.set_index("Strategy"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Weight Allocation
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Portfolio Weight Allocation")

    col_pie, col_stats = st.columns([1, 1])

    with col_pie:
        latest = wc.sort_values("Date").iloc[-1]
        weights_now = latest[tickers].values.astype(float)
        fig_pie = go.Figure(go.Pie(
            labels=tickers, values=weights_now,
            hole=0.4,
            marker_colors=px.colors.qualitative.Set2[:5],
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            title=f"Current Allocation ({latest['Date'].strftime('%Y-%m-%d')})",
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_stats:
        st.subheader("Weight Statistics")
        w_numeric = wc[tickers]
        stats = pd.DataFrame({
            "Mean": w_numeric.mean(),
            "Std": w_numeric.std(),
            "Min": w_numeric.min(),
            "Max": w_numeric.max(),
            "Current": wc.sort_values("Date").iloc[-1][tickers],
        }).round(4)
        st.dataframe(stats.style.format("{:.2%}"), use_container_width=True)

        hhi = (weights_now ** 2).sum()
        col_a, col_b = st.columns(2)
        col_a.metric("HHI (Concentration)", f"{hhi:.3f}", help="0=equal weight, 1=fully concentrated")
        col_b.metric("Max Position", f"{weights_now.max():.1%}")

    st.markdown("---")
    st.subheader("Weight Allocation Over Time (Constrained)")

    fig_area = go.Figure()
    wc_sorted = wc.sort_values("Date")
    colors_area = px.colors.qualitative.Set2[:5]
    for i, ticker in enumerate(tickers):
        fig_area.add_trace(go.Scatter(
            x=wc_sorted["Date"], y=wc_sorted[ticker].values,
            name=ticker, stackgroup="one",
            mode="lines", line=dict(width=0.5),
            fillcolor=colors_area[i],
            hovertemplate=f"{ticker}: %{{y:.1%}}<extra></extra>",
        ))
    fig_area.update_layout(
        title="DSR Constrained Weight Allocation",
        xaxis_title="Date", yaxis_title="Weight",
        yaxis_tickformat=".0%",
        height=400, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("Constrained vs Unconstrained Weights")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Constrained**")
        st.dataframe(wc.set_index("Date")[tickers].tail(10).style.format("{:.2%}"), use_container_width=True)
    with cols[1]:
        st.markdown("**Unconstrained**")
        st.dataframe(wu.set_index("Date")[tickers].tail(10).style.format("{:.2%}"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analytics
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Performance Analytics")

    # Monthly heatmap for selected primary strategy
    primary_name = "DSR Constrained" if "DSR Constrained" in selected else selected[0] if selected else "DSR Constrained"
    r = all_returns[primary_name]
    monthly = monthly_returns_table(r)

    fig_heat = go.Figure(go.Heatmap(
        z=monthly.values * 100,
        x=monthly.columns.tolist(),
        y=monthly.index.tolist(),
        colorscale=[
            [0, "#d73027"], [0.3, "#fc8d59"], [0.5, "#ffffbf"],
            [0.7, "#91bfdb"], [1, "#1a6faf"]
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in monthly.values * 100],
        texttemplate="%{text}",
        colorbar=dict(title="Return (%)"),
    ))
    fig_heat.update_layout(
        title=f"Monthly Returns Heatmap — {primary_name}",
        height=300, template="plotly_white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # Rolling Sharpe
    col_rs, col_rv = st.columns(2)

    with col_rs:
        fig_rs = go.Figure()
        for name in selected:
            rs = rolling_sharpe(all_returns[name], window=rolling_window)
            fig_rs.add_trace(go.Scatter(
                x=rs.index, y=rs.values, name=name,
                line=dict(color=COLORS[name], width=1.5),
            ))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_rs.update_layout(
            title=f"Rolling Sharpe Ratio ({rolling_window}d)",
            height=320, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rs, use_container_width=True)

    with col_rv:
        fig_rv = go.Figure()
        for name in selected:
            rv = all_returns[name].rolling(rolling_window).std() * np.sqrt(252)
            fig_rv.add_trace(go.Scatter(
                x=rv.index, y=rv.values, name=name,
                line=dict(color=COLORS[name], width=1.5),
            ))
        fig_rv.update_layout(
            title=f"Rolling Volatility ({rolling_window}d, annualized)",
            yaxis_tickformat=".1%",
            height=320, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rv, use_container_width=True)

    st.markdown("---")

    # Regime analysis
    st.subheader("High-Volatility Regime Analysis")
    vol_window = 21
    vol_threshold = 0.20  # annualized
    market_vol = returns_ew.rolling(vol_window).std() * np.sqrt(252)
    high_vol = market_vol > vol_threshold

    regime_rows = []
    for name in selected:
        r = all_returns[name]
        full_ann = (1 + r).prod() ** (252 / len(r)) - 1
        hv = high_vol.reindex(r.index, fill_value=False)
        high_vol_r = r[hv]
        low_vol_r = r[~hv]
        row = {
            "Strategy": name,
            "Full Period Ann. Return": f"{full_ann:.1%}",
            "High-Vol Period Return": f"{(1+high_vol_r).prod()-1:.1%}" if len(high_vol_r) > 0 else "N/A",
            "Low-Vol Period Return": f"{(1+low_vol_r).prod()-1:.1%}" if len(low_vol_r) > 0 else "N/A",
            "High-Vol Days": str(high_vol.sum()),
        }
        regime_rows.append(row)

    st.dataframe(pd.DataFrame(regime_rows).set_index("Strategy"), use_container_width=True)
    st.caption(f"High-volatility threshold: {vol_threshold:.0%} annualized ({vol_window}d rolling)")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Monitor
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Risk Monitor")

    # Risk metrics
    p = metrics[metrics["Strategy"] == "DSR_Constrained"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR (95%)", f"{p['VaR_95']:.2%}", help="Daily Value at Risk at 95% confidence")
    c2.metric("CVaR (95%)", f"{p['CVaR_95']:.2%}", help="Expected Shortfall (Expected loss beyond VaR)")
    c3.metric("Calmar Ratio", f"{p['Calmar_Ratio']:.2f}", help="Annualized return / Max Drawdown")
    c4.metric("Sortino Ratio", f"{p['Sortino_Ratio']:.3f}", help="Return per unit of downside risk")

    st.markdown("---")

    # Drawdown / Underwater chart
    fig_dd = go.Figure()
    for name in selected:
        dd = max_drawdown_series(all_returns[name]) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=name, fill="tozeroy",
            line=dict(color=COLORS[name], width=1),
            opacity=0.6,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
        ))
    fig_dd.update_layout(
        title="Drawdown (Underwater Plot)",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    col_ef, col_rr = st.columns(2)

    with col_ef:
        st.subheader("Efficient Frontier")
        ef_img = ROOT / "results/efficient_frontier.png"
        if ef_img.exists():
            st.image(str(ef_img), use_container_width=True)
        else:
            # Scatter of each strategy
            fig_ef = go.Figure()
            for _, row in metrics.iterrows():
                fig_ef.add_trace(go.Scatter(
                    x=[row["Ann_Volatility"]*100], y=[row["Ann_Return"]*100],
                    mode="markers+text",
                    name=row["Strategy"],
                    text=[row["Strategy"]], textposition="top center",
                    marker=dict(size=12),
                ))
            fig_ef.update_layout(
                xaxis_title="Annualized Volatility (%)",
                yaxis_title="Annualized Return (%)",
                height=350, template="plotly_white",
            )
            st.plotly_chart(fig_ef, use_container_width=True)

    with col_rr:
        st.subheader("Rolling VaR / CVaR (DSR Constrained)")
        r_c = all_returns["DSR Constrained"]
        roll_var = r_c.rolling(rolling_window).quantile(0.05).abs() * 100
        roll_cvar = r_c.rolling(rolling_window).apply(
            lambda x: -x[x <= np.quantile(x, 0.05)].mean() * 100 if len(x[x <= np.quantile(x, 0.05)]) > 0 else np.nan
        )
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=roll_var.index, y=roll_var.values, name=f"VaR 95% ({rolling_window}d)", line=dict(color="#e74c3c")))
        fig_risk.add_trace(go.Scatter(x=roll_cvar.index, y=roll_cvar.values, name=f"CVaR 95% ({rolling_window}d)", line=dict(color="#c0392b", dash="dash")))
        fig_risk.update_layout(
            yaxis_title="Daily Risk (%)",
            height=350, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")
    st.subheader("Risk Comparison Table")
    risk_tbl = metrics[["Strategy", "Max_Drawdown_Pct", "VaR_95_Pct", "CVaR_95_Pct", "Calmar_Ratio", "Sortino_Ratio"]].copy()
    risk_tbl["Calmar_Ratio"] = risk_tbl["Calmar_Ratio"].round(3)
    risk_tbl["Sortino_Ratio"] = risk_tbl["Sortino_Ratio"].round(3)
    risk_tbl.columns = ["Strategy", "Max Drawdown", "VaR 95%", "CVaR 95%", "Calmar", "Sortino"]
    st.dataframe(risk_tbl.set_index("Strategy"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Data
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Live Market Data & Current Model Recommendation")

    if st.session_state.live_fetch_key is None:
        st.info(
            "Click **🔄 Refresh Live Data** in the sidebar to download the latest prices "
            "from Yahoo Finance and get the model's current portfolio recommendation."
        )
    else:
        live = load_live_data(
            st.session_state.live_fetch_key,
            ",".join(st.session_state.user_tickers),
        )

        if live["error"]:
            st.error(f"Could not fetch live data: {live['error']}")
            st.info("The historical backtest results in the other tabs are still available.")
        else:
            last_date      = live["last_date"]
            close_live     = live["close_prices"]
            weights_raw    = live["weights_raw"]
            weights_const  = live["weights_const"]
            live_tickers   = live["user_tickers"]   # tickers that actually have data
            no_data        = live["no_data_tickers"]

            st.success(f"Prices current through **{last_date}**")

            if no_data:
                st.warning(
                    f"No data found for: **{', '.join(no_data)}**. "
                    "Check the ticker symbol(s) — they may be invalid or delisted."
                )

            # ── Section 1: LSTM Weight Recommendation ─────────────────────
            st.subheader("LSTM Model Recommendation")
            st.caption(
                "Left: raw LSTM softmax output (unconstrained). "
                "Right: after applying position limits [2%, 30%] from config.yaml. "
                "_Model trained on AAPL · MSFT · GOOGL · AMZN · META — weights shown for these 5 only._"
            )
            col_raw, col_con = st.columns(2)
            with col_raw:
                fig_raw = go.Figure(go.Pie(
                    labels=BACKTEST_TICKERS, values=weights_raw, hole=0.4,
                    marker_colors=px.colors.qualitative.Set2[:5],
                    textinfo="label+percent",
                ))
                fig_raw.update_layout(
                    title=f"Unconstrained ({last_date})",
                    height=320, template="plotly_white",
                )
                st.plotly_chart(fig_raw, use_container_width=True)

            with col_con:
                fig_con = go.Figure(go.Pie(
                    labels=BACKTEST_TICKERS, values=weights_const, hole=0.4,
                    marker_colors=px.colors.qualitative.Set2[:5],
                    textinfo="label+percent",
                ))
                fig_con.update_layout(
                    title=f"Constrained ({last_date})",
                    height=320, template="plotly_white",
                )
                st.plotly_chart(fig_con, use_container_width=True)

            # Delta vs last backtest date
            st.subheader("Live vs Last Backtest Weights (Dec 2024)")
            last_bt = wc.sort_values("Date").iloc[-1][BACKTEST_TICKERS].values.astype(float)
            delta_df = pd.DataFrame({
                "Live Constrained": weights_const,
                "Last Backtest":    last_bt,
                "Delta":            weights_const - last_bt,
            }, index=BACKTEST_TICKERS)

            def _color_delta(v):
                c = "green" if v > 0.001 else ("red" if v < -0.001 else "")
                return f"color: {c}"

            st.dataframe(
                delta_df.style
                    .format("{:.2%}", subset=["Live Constrained", "Last Backtest"])
                    .format("{:+.2%}", subset=["Delta"])
                    .map(_color_delta, subset=["Delta"]),
                use_container_width=True,
            )

            st.markdown("---")

            # ── Section 2: Recent Price Performance ───────────────────────
            N = 60
            st.subheader(f"Recent Price Performance — Last {N} Trading Days")
            st.caption(f"Watchlist: {' · '.join(live_tickers)}  ({len(live_tickers)} stocks)")
            recent = close_live.tail(N).copy()
            norm = (recent / recent.iloc[0]) * 100

            fig_rec = go.Figure()
            palette = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
            for i, t in enumerate(live_tickers):
                if t not in norm.columns:
                    continue
                fig_rec.add_trace(go.Scatter(
                    x=norm.index, y=norm[t].values, name=t,
                    line=dict(color=palette[i % len(palette)], width=2),
                    hovertemplate=f"{t}: %{{y:.1f}}<extra></extra>",
                ))
            fig_rec.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)
            fig_rec.update_layout(
                title=f"Indexed price (base=100 at {recent.index[0].date()})",
                xaxis_title="Date", yaxis_title="Indexed Price",
                height=420, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_rec, use_container_width=True)

            # Period return summary
            available_live = [t for t in live_tickers if t in close_live.columns]
            period_ret = (close_live[available_live].tail(N).iloc[-1]
                          / close_live[available_live].tail(N).iloc[0] - 1)
            st.dataframe(
                period_ret.to_frame(f"{N}-Day Return").T.style.format("{:+.2%}"),
                use_container_width=True,
            )

            st.markdown("---")

            # ── Section 3: Raw price table ─────────────────────────────────
            st.subheader(f"Closing Prices — Last {N} Trading Days")
            disp = close_live[available_live].tail(N).copy()
            disp.index = disp.index.strftime("%Y-%m-%d")
            st.dataframe(disp.style.format("${:.2f}"), use_container_width=True, height=280)

            st.caption(
                "Live prices via Yahoo Finance (yfinance). LSTM trained on 2018–2024 data; "
                "predictions extrapolate beyond the training period. "
                "Backtest metrics in other tabs are frozen at the Dec 2023–Dec 2024 test period."
            )

            st.markdown("---")

            # ── Section 4: Today's Intraday Chart ─────────────────────────────
            st.subheader("Today's Intraday Price (5-Minute Bars)")
            ms = market_status()
            status_icon = "🟢" if ms["is_open"] else "🔴"
            st.caption(
                f"Market: {status_icon} **{ms['label']}** — {ms['detail']}  |  "
                f"ET: {ms['now_et'].strftime('%H:%M')}"
            )
            try:
                _sys.path.insert(0, str(ROOT / "python"))
                from signals import fetch_intraday as _fetch_intraday
                intraday_data = _fetch_intraday(BACKTEST_TICKERS, interval="5m")
                if intraday_data:
                    _intra_ticker = st.selectbox(
                        "Ticker", list(intraday_data.keys()), key="tab5_intraday_ticker"
                    )
                    _bars = intraday_data[_intra_ticker]
                    _fig_intra = go.Figure()
                    _fig_intra.add_trace(go.Scatter(
                        x=_bars.index,
                        y=_bars["Close"].values,
                        mode="lines",
                        name="Price",
                        line=dict(color="#1f77b4", width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(31,119,180,0.08)",
                        hovertemplate="%{x|%H:%M}: $%{y:.2f}<extra></extra>",
                    ))
                    _fig_intra.update_layout(
                        title=f"{_intra_ticker} — Today's Session (5m bars)",
                        xaxis_title="Time (ET)",
                        yaxis_title="Price ($)",
                        height=320,
                        template="plotly_white",
                    )
                    if not ms["is_open"]:
                        _fig_intra.add_annotation(
                            text="Market closed — showing last trading session",
                            xref="paper", yref="paper", x=0.5, y=0.95,
                            showarrow=False, font=dict(color="gray", size=11),
                        )
                    st.plotly_chart(_fig_intra, use_container_width=True)
                else:
                    st.info("Intraday data unavailable. yfinance may be rate-limited — try again shortly.")
            except Exception as _e:
                st.warning(f"Could not load intraday data: {_e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Sentiment
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("FinBERT Sentiment Analysis")

    sentiment_data = load_sentiment_data()

    if sentiment_data is None:
        st.info(
            "No sentiment data found. Run the pipeline to generate it:\n\n"
            "```bash\n"
            "cd python\n"
            "python scraper.py          # download headlines (add AV key for historical)\n"
            "python sentiment.py        # score with FinBERT → data/processed/sentiment_scores.parquet\n"
            "```\n\n"
            "Optionally set `sentiment.alphavantage_key` in `config.yaml` for full 2018–2024 coverage."
        )
    else:
        sent = sentiment_data.copy()

        # ── Summary metrics ───────────────────────────────────────────────────
        st.subheader("Current Sentiment Snapshot")
        latest_sent = sent.iloc[-1]
        cols_sent = st.columns(len(BACKTEST_TICKERS))
        for i, t in enumerate(BACKTEST_TICKERS):
            if t in latest_sent.index:
                val = latest_sent[t]
                label = "Positive" if val > 0.1 else ("Negative" if val < -0.1 else "Neutral")
                delta_color = "normal"
                cols_sent[i].metric(
                    label=t,
                    value=f"{val:+.3f}",
                    delta=label,
                    delta_color=delta_color,
                )

        st.caption(
            f"Alpha signal = P(positive) − P(negative) via ProsusAI/finbert  |  "
            f"Latest date: {sent.index[-1].date()}"
        )
        st.markdown("---")

        # ── Heatmap: asset × time ─────────────────────────────────────────────
        st.subheader("Sentiment Heatmap — Asset × Time")

        # Let user pick time window
        window_opts = {"3 months": 63, "6 months": 126, "1 year": 252, "All": len(sent)}
        window_label = st.selectbox("Time window", list(window_opts.keys()), index=2)
        n_days = window_opts[window_label]
        sent_window = sent[BACKTEST_TICKERS].tail(n_days)

        # Resample to weekly for readability
        sent_weekly = sent_window.resample("W").mean()

        fig_heat = go.Figure(go.Heatmap(
            z=sent_weekly.T.values,
            x=sent_weekly.index.strftime("%Y-%m-%d").tolist(),
            y=BACKTEST_TICKERS,
            colorscale=[
                [0.0,  "#d73027"],
                [0.35, "#fc8d59"],
                [0.5,  "#ffffbf"],
                [0.65, "#91bfdb"],
                [1.0,  "#1a6faf"],
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Sentiment", tickvals=[-1, -0.5, 0, 0.5, 1]),
            hovertemplate="Date: %{x}<br>Ticker: %{y}<br>Sentiment: %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title=f"Weekly Mean Sentiment — {window_label}",
            xaxis_title="Week",
            yaxis_title="Ticker",
            height=300,
            template="plotly_white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # ── Time-series: sentiment vs returns ────────────────────────────────
        st.subheader("Sentiment vs. Returns Overlay")
        ticker_sel = st.selectbox("Select ticker", BACKTEST_TICKERS, key="sent_ticker")

        if ticker_sel in sent.columns and ticker_sel in close_prices.columns:
            sent_ts = sent[ticker_sel].tail(n_days)
            price_ts = close_prices[ticker_sel].reindex(sent_ts.index, method="ffill")
            ret_ts = price_ts.pct_change().fillna(0)

            fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ov.add_trace(
                go.Bar(
                    x=sent_ts.index, y=sent_ts.values,
                    name="Sentiment",
                    marker_color=["#1a6faf" if v >= 0 else "#d73027" for v in sent_ts.values],
                    opacity=0.7,
                ),
                secondary_y=False,
            )
            fig_ov.add_trace(
                go.Scatter(
                    x=ret_ts.index, y=(1 + ret_ts).cumprod().values,
                    name=f"{ticker_sel} Cumulative Return",
                    line=dict(color="#2ca02c", width=2),
                ),
                secondary_y=True,
            )
            fig_ov.update_layout(
                title=f"{ticker_sel} — Sentiment vs. Cumulative Return",
                height=380, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_ov.update_yaxes(title_text="Sentiment score", secondary_y=False)
            fig_ov.update_yaxes(title_text="Cumulative return", secondary_y=True)
            st.plotly_chart(fig_ov, use_container_width=True)

        st.markdown("---")

        # ── Ablation study results ────────────────────────────────────────────
        st.subheader("Ablation Study: With vs Without Sentiment")
        ablation_path = ROOT / "results/ablation.csv"
        if ablation_path.exists():
            abl = pd.read_csv(ablation_path)
            display_cols = ["variant", "input_size", "best_val_dsr", "test_dsr", "test_sharpe"]
            available = [c for c in display_cols if c in abl.columns]
            st.dataframe(
                abl[available].rename(columns={
                    "variant": "Variant",
                    "input_size": "Input Size",
                    "best_val_dsr": "Best Val DSR",
                    "test_dsr": "Test DSR",
                    "test_sharpe": "Test Sharpe",
                }).set_index("Variant"),
                use_container_width=True,
            )
            if len(abl) == 2:
                d_dsr = abl.iloc[1]["best_val_dsr"] - abl.iloc[0]["best_val_dsr"]
                d_sh = abl.iloc[1]["test_sharpe"] - abl.iloc[0]["test_sharpe"]
                c1, c2 = st.columns(2)
                c1.metric(
                    "Sentiment impact on Val DSR",
                    f"{d_dsr:+.6f}",
                    delta=("improved" if d_dsr > 0 else "degraded"),
                    delta_color="normal" if d_dsr > 0 else "inverse",
                )
                c2.metric(
                    "Sentiment impact on Test Sharpe",
                    f"{d_sh:+.4f}",
                    delta=("improved" if d_sh > 0 else "degraded"),
                    delta_color="normal" if d_sh > 0 else "inverse",
                )
        else:
            st.info(
                "No ablation results yet. Run `python ablation.py` after sentiment scoring "
                "to see the with/without comparison."
            )

        # ── Raw score table ───────────────────────────────────────────────────
        with st.expander("Raw sentiment scores (last 30 days)"):
            disp = sent[BACKTEST_TICKERS].tail(30).copy()
            disp.index = disp.index.strftime("%Y-%m-%d")
            st.dataframe(disp.style.format("{:+.4f}").background_gradient(
                cmap="RdYlBu", vmin=-1, vmax=1
            ), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Real-Time Signals
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Real-Time Technical Signals")

    # ── Market status banner ──────────────────────────────────────────────────
    ms = market_status()
    if ms["is_open"]:
        st.success(f"🟢 Market **OPEN** — {ms['detail']}  |  ET: {ms['now_et'].strftime('%H:%M:%S')}")
    else:
        st.error(f"🔴 Market **CLOSED** — {ms['detail']}  |  ET: {ms['now_et'].strftime('%H:%M:%S')}")

    st.caption(
        "Signals are rule-based (RSI, MACD, SMA, Bollinger Bands, momentum). "
        "Works for any ticker in the Watchlist — not limited to the trained 5. "
        f"Last page load: {_dt_cls.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.markdown("---")

    # ── Load signals ──────────────────────────────────────────────────────────
    _signals_path_ok = (ROOT / "python" / "signals.py").exists()
    if not _signals_path_ok:
        st.error("python/signals.py not found.")
    else:
        _sys.path.insert(0, str(ROOT / "python"))
        from signals import get_signals, fetch_intraday

        _watchlist = st.session_state.get("user_tickers", list(BACKTEST_TICKERS))

        with st.spinner(f"Computing signals for {len(_watchlist)} ticker(s)..."):
            try:
                _sig_df = get_signals(_watchlist, period="1y")
            except Exception as _exc:
                _sig_df = None
                st.error(f"Signal computation failed: {_exc}")

        if _sig_df is not None and not _sig_df.empty:

            # ── Signal summary table ──────────────────────────────────────────
            st.subheader("Signal Summary")

            def _sig_color(val):
                colors = {
                    "STRONG BUY":  "background-color:#1a6faf;color:white",
                    "BUY":         "background-color:#91bfdb;color:black",
                    "HOLD":        "background-color:#ffffbf;color:black",
                    "SELL":        "background-color:#fc8d59;color:black",
                    "STRONG SELL": "background-color:#d73027;color:white",
                }
                for k, v in colors.items():
                    if k in str(val):
                        return v
                return ""

            _disp = _sig_df[["signal", "score", "current_price", "change_pct_1d",
                              "rsi", "sma50_pts", "sma200_pts", "bb_pct_b", "momentum_5d"]].copy()
            _disp.columns = ["Signal", "Score", "Price ($)", "1d Change",
                              "RSI", "vs SMA50", "vs SMA200", "BB %B", "Mom 5d"]
            _disp["Price ($)"] = _disp["Price ($)"].apply(
                lambda x: f"${float(x):.2f}" if x is not None else "N/A"
            )
            _disp["1d Change"] = _disp["1d Change"].apply(
                lambda x: f"{float(x):+.2%}" if x is not None else "N/A"
            )
            _disp["RSI"] = _disp["RSI"].apply(
                lambda x: f"{float(x):.1f}" if x is not None else "N/A"
            )
            _disp["BB %B"] = _disp["BB %B"].apply(
                lambda x: f"{float(x):.3f}" if x is not None else "N/A"
            )
            _disp["Mom 5d"] = _disp["Mom 5d"].apply(
                lambda x: f"{float(x):+.2%}" if x is not None else "N/A"
            )
            _disp["vs SMA50"]  = _disp["vs SMA50"].apply(
                lambda x: "▲ Above" if x == 1 else ("▼ Below" if x == -1 else "—")
            )
            _disp["vs SMA200"] = _disp["vs SMA200"].apply(
                lambda x: "▲ Above" if x == 1 else ("▼ Below" if x == -1 else "—")
            )

            st.dataframe(
                _disp.style.map(_sig_color, subset=["Signal"]),
                use_container_width=True,
            )

            st.markdown("---")

            # ── Intraday chart ────────────────────────────────────────────────
            st.subheader("Intraday Chart — Today's Session")
            _intra_sel = st.selectbox(
                "Select ticker", _watchlist, key="tab7_intraday_ticker"
            )
            _intra_interval = st.radio(
                "Bar interval", ["1m", "5m", "15m"], index=1,
                horizontal=True, key="tab7_intraday_interval"
            )
            with st.spinner("Loading intraday data..."):
                try:
                    _intra = fetch_intraday([_intra_sel], interval=_intra_interval)
                except Exception as _exc:
                    _intra = {}
                    st.warning(f"Intraday fetch failed: {_exc}")

            if _intra_sel in _intra and not _intra[_intra_sel].empty:
                _bars = _intra[_intra_sel]
                _fig7 = go.Figure()
                _fig7.add_trace(go.Candlestick(
                    x=_bars.index,
                    open=_bars["Open"], high=_bars["High"],
                    low=_bars["Low"],   close=_bars["Close"],
                    name=_intra_sel,
                    increasing_line_color="#2ca02c",
                    decreasing_line_color="#d62728",
                ))
                # Volume bars on secondary axis
                _fig7 = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25], vertical_spacing=0.03,
                )
                _fig7.add_trace(go.Candlestick(
                    x=_bars.index,
                    open=_bars["Open"], high=_bars["High"],
                    low=_bars["Low"],   close=_bars["Close"],
                    name="Price",
                    increasing_line_color="#2ca02c",
                    decreasing_line_color="#d62728",
                ), row=1, col=1)
                _fig7.add_trace(go.Bar(
                    x=_bars.index, y=_bars["Volume"].values,
                    name="Volume",
                    marker_color=[
                        "#2ca02c" if c >= o else "#d62728"
                        for o, c in zip(_bars["Open"].values, _bars["Close"].values)
                    ],
                    opacity=0.6,
                ), row=2, col=1)
                _fig7.update_layout(
                    title=f"{_intra_sel} — {_intra_interval} bars  "
                          f"({'Live session' if ms['is_open'] else 'Last trading session'})",
                    xaxis_rangeslider_visible=False,
                    height=500, template="plotly_white",
                    showlegend=False,
                )
                _fig7.update_yaxes(title_text="Price ($)", row=1, col=1)
                _fig7.update_yaxes(title_text="Volume",   row=2, col=1)
                if not ms["is_open"]:
                    _fig7.add_annotation(
                        text="Market closed — showing last trading session",
                        xref="paper", yref="paper", x=0.5, y=1.03,
                        showarrow=False, font=dict(color="gray", size=11),
                    )
                st.plotly_chart(_fig7, use_container_width=True)
            else:
                st.info(
                    f"No intraday data for **{_intra_sel}**. "
                    "yfinance may be rate-limited — try again in a moment, "
                    "or switch to a wider bar interval."
                )

            st.markdown("---")

            # ── Per-ticker detail expanders ───────────────────────────────────
            st.subheader("Signal Detail")
            st.caption("Expand a ticker to see individual indicator breakdown.")

            for _t in _sig_df.index:
                _r = _sig_df.loc[_t]
                with st.expander(f"{_t}  {_r.get('signal', '')}  |  Score: {_r.get('score', 0):+d}"):
                    _dc1, _dc2, _dc3 = st.columns(3)
                    _dc1.metric("Price", f"${float(_r['current_price']):.2f}",
                                delta=f"{float(_r['change_pct_1d']):+.2%}" if _r['change_pct_1d'] is not None else None)
                    _dc2.metric("RSI-14", f"{float(_r['rsi']):.1f}" if _r['rsi'] is not None else "N/A",
                                delta=f"{int(_r['rsi_pts']):+d} pts")
                    _dc3.metric("5d Momentum",
                                f"{float(_r['momentum_5d']):+.2%}" if _r['momentum_5d'] is not None else "N/A",
                                delta=f"{int(_r['mom_pts']):+d} pts")

                    _dd1, _dd2, _dd3 = st.columns(3)
                    _dd1.metric("MACD vs Signal",
                                "Above" if _r['macd_pts'] == 1 else "Below",
                                delta=f"{int(_r['macd_pts']):+d} pts")
                    _dd2.metric("vs 50d SMA",
                                "Above" if _r['sma50_pts'] == 1 else "Below",
                                delta=f"{int(_r['sma50_pts']):+d} pts")
                    _dd3.metric("vs 200d SMA",
                                "Above" if _r['sma200_pts'] == 1 else "Below",
                                delta=f"{int(_r['sma200_pts']):+d} pts")

                    _bb_val = float(_r['bb_pct_b']) if _r['bb_pct_b'] is not None else 0.5
                    _bb_label = "Oversold (<15%)" if _bb_val < 0.15 else (
                        "Overbought (>85%)" if _bb_val > 0.85 else f"Neutral ({_bb_val:.2f})"
                    )
                    st.metric("Bollinger %B", _bb_label, delta=f"{int(_r['bb_pts']):+d} pts")

                    # Score bar
                    _score = int(_r["score"])
                    _max = 6
                    _pct = (_score + _max) / (2 * _max)
                    st.progress(_pct, text=f"Composite score: {_score:+d} / {_max}")

        elif _sig_df is not None and _sig_df.empty:
            st.warning("No signals computed — check ticker symbols in your watchlist.")

        st.markdown("---")
        st.caption(
            "Signals are for informational/educational purposes only and do not constitute "
            "financial advice. Past technical patterns do not guarantee future performance."
        )
