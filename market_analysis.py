import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Bull vs Bear Market Analysis",
    page_icon="📊",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────
COLORS = {
    "DSR Constrained":   "#1f77b4",
    "DSR Unconstrained": "#ff7f0e",
    "Equal Weight":      "#9467bd",
}
ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
ASSET_COLORS = px.colors.qualitative.Set2[:5]

# ── Data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_bull_metrics():
    df = pd.read_csv("results/bull_2024/metrics.csv")
    df["Strategy"] = (df["Strategy"]
                      .str.replace("_", " ")
                      .str.replace("EqualWeight", "Equal Weight"))
    return df

@st.cache_data
def load_bear_metrics():
    df = pd.read_csv("results/bear_2022/metrics_bear_2022.csv")
    df["Strategy"] = (df["Strategy"]
                      .str.replace("_2022", "", regex=False)
                      .str.replace("_", " ")
                      .str.replace("EqualWeight", "Equal Weight"))
    return df

@st.cache_data
def load_bear_returns():
    df = pd.read_csv("results/bear_2022/returns_bear_2022.csv",
                     parse_dates=["Date"])
    return df.set_index("Date").sort_index()

@st.cache_data
def load_bull_weights():
    df = pd.read_csv("results/bull_2024/weights_constrained.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()[ASSETS]

@st.cache_data
def load_bear_weights():
    df = pd.read_csv("results/bear_2022/weights_bear_2022.csv",
                     parse_dates=["Date"])
    return df.set_index("Date").sort_index()[ASSETS]

@st.cache_data
def compute_bull_cum_returns():
    """Compute daily cumulative returns for all 3 strategies in the bull period."""
    prices = pd.read_parquet("data/raw/prices.parquet")
    close_cols = [c for c in prices.columns if c.endswith("_Close")]
    close = prices[close_cols].copy()
    close.columns = [c.replace("_Close", "") for c in close.columns]
    close = close.loc["2023-12-15":"2024-12-30", ASSETS]
    daily_ret = close.pct_change().dropna()

    # Constrained
    w_con = load_bull_weights().reindex(daily_ret.index, method="ffill")
    port_con = (w_con * daily_ret).sum(axis=1)

    # Unconstrained
    w_unc_raw = pd.read_csv("results/bull_2024/weights_unconstrained.csv")
    w_unc_raw["Date"] = pd.to_datetime(w_unc_raw["Date"])
    w_unc = w_unc_raw.set_index("Date").sort_index()[ASSETS]
    w_unc = w_unc.reindex(daily_ret.index, method="ffill")
    port_unc = (w_unc * daily_ret).sum(axis=1)

    # Equal weight (1/5 each)
    port_eq = daily_ret.mean(axis=1)

    cum = pd.DataFrame({
        "DSR Constrained":   (1 + port_con).cumprod() - 1,
        "DSR Unconstrained": (1 + port_unc).cumprod() - 1,
        "Equal Weight":      (1 + port_eq).cumprod() - 1,
    })
    return cum

# ── Chart helpers ──────────────────────────────────────────────────────────

def cum_returns_chart(cum_df, title):
    fig = go.Figure()
    for col in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df[col],
            name=col,
            line=dict(color=COLORS.get(col, "#ffffff"), width=2),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=380,
        yaxis_tickformat=".0%",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def weight_area_chart(w_df, title):
    fig = go.Figure()
    for i, asset in enumerate(ASSETS):
        fig.add_trace(go.Scatter(
            x=w_df.index, y=w_df[asset],
            name=asset,
            stackgroup="one",
            line=dict(width=0.5, color=ASSET_COLORS[i]),
            fillcolor=ASSET_COLORS[i],
        ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=340,
        yaxis_tickformat=".0%",
        xaxis_title="Date",
        yaxis_title="Weight",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def metrics_table(df):
    display = df[["Strategy", "Ann_Return", "Ann_Volatility", "Sharpe_Ratio",
                  "Sortino_Ratio", "Max_Drawdown", "Calmar_Ratio",
                  "VaR_95", "CVaR_95"]].copy()
    display.columns = ["Strategy", "Ann Return", "Volatility", "Sharpe",
                       "Sortino", "Max DD", "Calmar", "VaR 95%", "CVaR 95%"]

    pct_cols  = ["Ann Return", "Volatility", "Max DD", "VaR 95%", "CVaR 95%"]
    rat_cols  = ["Sharpe", "Sortino", "Calmar"]
    fmt = {c: "{:.1%}" for c in pct_cols}
    fmt.update({c: "{:.2f}" for c in rat_cols})

    def _colour(val):
        if not isinstance(val, float):
            return ""
        return f"color: {'#2ecc71' if val >= 0 else '#e74c3c'}"

    styled = (display.style
              .format(fmt)
              .map(_colour, subset=pct_cols + rat_cols))
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Tab renderers ──────────────────────────────────────────────────────────

def render_overview(bull_m, bear_m):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🐂 Bull Market — Dec 2023 to Dec 2024")
        bull_con = bull_m[bull_m["Strategy"] == "DSR Constrained"].iloc[0]
        st.metric("DSR Constrained — Ann. Return", f"{bull_con['Ann_Return']*100:.1f}%")
        st.metric("Sharpe Ratio", f"{bull_con['Sharpe_Ratio']:.2f}")
    with c2:
        st.markdown("### 🐻 Bear Market — Full Year 2022")
        bear_con = bear_m[bear_m["Strategy"] == "DSR Constrained"].iloc[0]
        st.metric("DSR Constrained — Ann. Return", f"{bear_con['Ann_Return']*100:.1f}%")
        st.metric("Sharpe Ratio", f"{bear_con['Sharpe_Ratio']:.2f}")

    st.divider()
    st.markdown("#### Head-to-Head: Key Metrics")

    metrics_map = {
        "Ann. Return":  "Ann_Return",
        "Sharpe":       "Sharpe_Ratio",
        "Sortino":      "Sortino_Ratio",
        "Max Drawdown": "Max_Drawdown",
    }
    x_labels = list(metrics_map.keys())

    fig = go.Figure()
    bar_specs = [
        ("DSR Constrained", bull_m, "#1f77b4", 1.0,  "Bull — DSR Constrained"),
        ("Equal Weight",    bull_m, "#9467bd", 1.0,  "Bull — Equal Weight"),
        ("DSR Constrained", bear_m, "#1f77b4", 0.45, "Bear — DSR Constrained"),
        ("Equal Weight",    bear_m, "#9467bd", 0.45, "Bear — Equal Weight"),
    ]
    for strategy, src, color, opacity, label in bar_specs:
        row = src[src["Strategy"] == strategy].iloc[0]
        y_vals = [row[col] for col in metrics_map.values()]
        fig.add_trace(go.Bar(
            name=label, x=x_labels, y=y_vals,
            marker_color=color, opacity=opacity,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=430,
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    bull_edge = (bull_m[bull_m["Strategy"] == "DSR Constrained"].iloc[0]["Ann_Return"]
                 - bull_m[bull_m["Strategy"] == "Equal Weight"].iloc[0]["Ann_Return"])
    bear_edge = (bear_m[bear_m["Strategy"] == "DSR Constrained"].iloc[0]["Ann_Return"]
                 - bear_m[bear_m["Strategy"] == "Equal Weight"].iloc[0]["Ann_Return"])

    st.info(
        "**Model consistently beat the equal-weight benchmark across both regimes:**\n\n"
        f"🐂 **Bull market**: DSR Constrained outperformed by **+{bull_edge*100:.1f}pp**\n\n"
        f"🐻 **Bear market**: DSR Constrained outperformed by **+{bear_edge*100:.1f}pp** "
        f"(sustained smaller losses)"
    )


def render_bull(bull_m, bull_cum, bull_w):
    st.markdown("#### Performance Metrics")
    metrics_table(bull_m)

    st.markdown("#### Cumulative Returns")
    st.plotly_chart(
        cum_returns_chart(bull_cum, "Bull Market — Cumulative Returns (Dec 2023 – Dec 2024)"),
        use_container_width=True,
    )

    st.markdown("#### Portfolio Weights — DSR Constrained (Weekly Rebalances)")
    st.plotly_chart(
        weight_area_chart(bull_w, "Bull Market — Weight Allocation Over Time"),
        use_container_width=True,
    )


def render_bear(bear_m, bear_ret, bear_w):
    st.caption(
        "Model was retrained on 2018–2021 data only. "
        "The entire year 2022 is fully out-of-sample. "
        "S&P 500 fell ~19% this year."
    )
    st.markdown("#### Performance Metrics")
    metrics_table(bear_m)

    st.markdown("#### Cumulative Returns")
    cum = (1 + bear_ret).cumprod() - 1
    cum.columns = ["DSR Constrained", "DSR Unconstrained", "Equal Weight"]
    st.plotly_chart(
        cum_returns_chart(cum, "Bear Market — Cumulative Returns (Jan 2022 – Dec 2022)"),
        use_container_width=True,
    )

    st.markdown("#### Portfolio Weights — DSR Constrained (Weekly Rebalances)")
    st.plotly_chart(
        weight_area_chart(bear_w, "Bear Market — Weight Allocation Over Time"),
        use_container_width=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────

st.title("📊 Bull vs Bear Market Analysis")
st.caption("DSR-LSTM Portfolio Optimizer · Performance across market regimes")

bull_metrics = load_bull_metrics()
bear_metrics = load_bear_metrics()
bear_returns = load_bear_returns()
bull_weights = load_bull_weights()
bear_weights = load_bear_weights()

with st.spinner("Computing bull market returns from prices..."):
    bull_cum = compute_bull_cum_returns()

tab1, tab2, tab3 = st.tabs(["📋 Overview", "🐂 Bull Market 2024", "🐻 Bear Market 2022"])

with tab1:
    render_overview(bull_metrics, bear_metrics)

with tab2:
    render_bull(bull_metrics, bull_cum, bull_weights)

with tab3:
    render_bear(bear_metrics, bear_returns, bear_weights)
