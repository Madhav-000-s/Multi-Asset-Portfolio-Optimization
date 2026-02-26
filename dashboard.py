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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSR Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    weights_constrained = pd.read_csv(ROOT / "results/weights_history.csv", parse_dates=["Date"])
    weights_unconstrained = pd.read_csv(ROOT / "results/weights_unconstrained.csv", parse_dates=["Date"])

    prices = pd.read_parquet(ROOT / "data/raw/prices.parquet")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Overview",
    "⚖️ Weight Allocation",
    "📈 Analytics",
    "🛡️ Risk Monitor",
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
        high_vol_r = r[high_vol]
        low_vol_r = r[~high_vol]
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
