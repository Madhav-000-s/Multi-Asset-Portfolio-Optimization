"""
Bear Market Test — 2022
Walks forward through 2022 week by week using the model trained on 2018-2021.
Replicates backtest.R logic in Python (no R dependency needed).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import predict as _predict
from predict import clear_cache, predict_at_date

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
BEAR_START     = "2022-01-01"
BEAR_END       = "2022-12-31"
TC_BPS         = 0.001        # 10 basis points per unit of turnover
W_MIN, W_MAX   = 0.02, 0.30
PARQUET        = Path(__file__).parent.parent / "data/raw/prices.parquet"

# ── Helpers ───────────────────────────────────────────────────────────────────

def constrain_weights(w, w_min=W_MIN, w_max=W_MAX, max_iter=200):
    w = np.array(w, dtype=float)
    for _ in range(max_iter):
        w = np.clip(w, w_min, w_max)
        w /= w.sum()
        if np.all(w >= w_min - 1e-9) and np.all(w <= w_max + 1e-9):
            break
    return w


def load_prices_2022():
    df = pd.read_parquet(PARQUET)
    df.index = pd.to_datetime(df.index)
    close_cols = [f"{t}_Close" for t in TICKERS]
    close = df[close_cols].copy()
    close.columns = TICKERS
    return close


def weekly_rebalance_dates(close, start, end):
    mask = (close.index >= start) & (close.index <= end)
    dates = close.index[mask]
    # xts::endpoints("weeks") equivalent — last trading day of each week
    weeks = dates.to_series().resample("W").last().dropna()
    return pd.DatetimeIndex(weeks.values)


def compute_metrics(returns_series, label):
    r = returns_series.dropna()
    ann_factor = 52           # weekly returns
    ann_ret  = (1 + r).prod() ** (ann_factor / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(ann_factor)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    sortino_denom = r[r < 0].std() * np.sqrt(ann_factor)
    sortino  = ann_ret / sortino_denom if sortino_denom > 0 else 0
    cumret   = (1 + r).cumprod()
    peak     = cumret.cummax()
    drawdown = (cumret - peak) / peak
    max_dd   = drawdown.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
    var95    = np.percentile(r, 5)
    cvar95   = r[r <= var95].mean()

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Ann. Return:   {ann_ret:+.2%}")
    print(f"  Ann. Vol:       {ann_vol:.2%}")
    print(f"  Sharpe:         {sharpe:.3f}")
    print(f"  Sortino:        {sortino:.3f}")
    print(f"  Max Drawdown:   {max_dd:.2%}")
    print(f"  Calmar:         {calmar:.3f}")
    print(f"  VaR (95%):      {var95:.2%}")
    print(f"  CVaR (95%):     {cvar95:.2%}")

    return {
        "Strategy": label,
        "Ann_Return": ann_ret, "Ann_Volatility": ann_vol,
        "Sharpe_Ratio": sharpe, "Sortino_Ratio": sortino,
        "Max_Drawdown": max_dd, "Calmar_Ratio": calmar,
        "VaR_95": var95, "CVaR_95": cvar95,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*60)
    print("  BEAR MARKET TEST — 2022")
    print("  Model trained on: 2018-01-01 → 2021-12-31")
    print("  Test period:      2022-01-01 → 2022-12-31")
    print("  (Model never saw 2022 data)")
    print("="*60)

    clear_cache()
    close = load_prices_2022()
    reb_dates = weekly_rebalance_dates(close, BEAR_START, BEAR_END)
    print(f"\nRebalance dates: {len(reb_dates)}")

    # Daily returns for return calculation
    daily_ret = close.pct_change()

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    constrained_returns   = []
    unconstrained_returns = []
    ew_returns            = []
    weights_log           = []
    prev_w_con = np.ones(5) / 5
    prev_w_unc = np.ones(5) / 5

    for i, date in enumerate(reb_dates[:-1]):
        date_str = str(date.date())
        next_date = reb_dates[i + 1]

        # Get LSTM weights
        try:
            raw_w = predict_at_date(date_str)
        except Exception as e:
            print(f"  WARNING: predict failed for {date_str}: {e}")
            raw_w = prev_w_con.copy()

        con_w = constrain_weights(raw_w)

        # Period return = sum(w * returns) for each day until next rebalance
        period_dates = daily_ret.index[
            (daily_ret.index > date) & (daily_ret.index <= next_date)
        ]
        if len(period_dates) == 0:
            continue

        period_ret = daily_ret.loc[period_dates, TICKERS]
        ew_w = np.ones(5) / 5

        # Portfolio returns (constant weight within period)
        con_port  = (period_ret * con_w).sum(axis=1)
        unc_port  = (period_ret * raw_w).sum(axis=1)
        ew_port   = (period_ret * ew_w).sum(axis=1)

        # Transaction costs on first day
        tc_con = TC_BPS * np.abs(con_w - prev_w_con).sum()
        tc_unc = TC_BPS * np.abs(raw_w - prev_w_unc).sum()

        weekly_con = (1 + con_port).prod() - 1 - tc_con
        weekly_unc = (1 + unc_port).prod() - 1 - tc_unc
        weekly_ew  = (1 + ew_port).prod() - 1

        constrained_returns.append((next_date, weekly_con))
        unconstrained_returns.append((next_date, weekly_unc))
        ew_returns.append((next_date, weekly_ew))
        weights_log.append([date_str] + list(con_w))

        prev_w_con = con_w
        prev_w_unc = raw_w

        if (i + 1) % 10 == 0:
            print(f"  Rebalance {i+1}/{len(reb_dates)-1} | {date_str} | "
                  f"con_w: {con_w.round(2)}")

    # ── Build series ──────────────────────────────────────────────────────────
    r_con = pd.Series(dict(constrained_returns))
    r_unc = pd.Series(dict(unconstrained_returns))
    r_ew  = pd.Series(dict(ew_returns))

    # ── Metrics ───────────────────────────────────────────────────────────────
    results = []
    results.append(compute_metrics(r_con, "DSR_Constrained_2022"))
    results.append(compute_metrics(r_unc, "DSR_Unconstrained_2022"))
    results.append(compute_metrics(r_ew,  "EqualWeight_2022"))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "results/bear_2022"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(out_dir / "metrics_bear_2022.csv", index=False)

    wh = pd.DataFrame(weights_log, columns=["Date"] + TICKERS)
    wh.to_csv(out_dir / "weights_bear_2022.csv", index=False)

    pd.DataFrame({"Date": r_con.index, "DSR_Constrained": r_con.values,
                  "DSR_Unconstrained": r_unc.values, "EqualWeight": r_ew.values}
                 ).to_csv(out_dir / "returns_bear_2022.csv", index=False)

    print(f"\nResults saved → {out_dir}")
    return metrics_df


if __name__ == "__main__":
    run()
