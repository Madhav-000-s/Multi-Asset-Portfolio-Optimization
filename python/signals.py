"""
Real-Time Technical Signal Generator

Computes rule-based BUY/HOLD/SELL signals for any ticker using live
daily data from yfinance. Works independently of the LSTM model.

Signal scoring (6 components, each contributes points):
  RSI-14         : <30 → +2 (oversold)  |  >70 → -2 (overbought)
  MACD vs signal : above → +1           |  below → -1
  vs 50d SMA     : above → +1           |  below → -1
  vs 200d SMA    : above → +1           |  below → -1
  Bollinger %B   : <0.15 → +1           |  >0.85 → -1
  5d momentum    : >+2% → +1            |  <-2% → -1

Composite score → label:
  ≥ +4  STRONG BUY  🟢
  +2/+3 BUY         🔵
  -1..+1 HOLD       ⚪
  -2/-3 SELL        🟠
  ≤ -4  STRONG SELL 🔴

Usage:
  python signals.py                     # signals for default 5 tickers
  python signals.py AAPL NVDA TSLA     # custom tickers
"""

import argparse
from datetime import datetime, time as dtime
import zoneinfo
import numpy as np
import pandas as pd
import yfinance as yf

# ── Signal constants ──────────────────────────────────────────────────────────
SIGNAL_LABELS = {
    "STRONG BUY":  ("STRONG BUY",  "🟢", 4),
    "BUY":         ("BUY",         "🔵", 2),
    "HOLD":        ("HOLD",        "⚪", 0),
    "SELL":        ("SELL",        "🟠", -2),
    "STRONG SELL": ("STRONG SELL", "🔴", -4),
}

ET = zoneinfo.ZoneInfo("America/New_York")
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


# ── Market status ─────────────────────────────────────────────────────────────

def market_status() -> dict:
    """
    Returns US equity market status based on current Eastern Time.

    Returns dict with keys:
        is_open    : bool
        label      : str  ("OPEN" or "CLOSED")
        detail     : str  e.g. "Closes in 2h 14m" or "Opens in 3h 45m"
        now_et     : datetime (ET)
    """
    now = datetime.now(tz=ET)
    weekday = now.weekday()   # 0=Mon … 6=Sun
    t = now.time()

    is_open = (weekday < 5) and (MARKET_OPEN <= t < MARKET_CLOSE)

    if is_open:
        close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
        delta = close_dt - now
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        detail = f"Closes in {h}h {m:02d}m"
    else:
        # Find next open: next weekday at 9:30 ET
        days_ahead = 1
        while True:
            candidate = now + pd.Timedelta(days=days_ahead)
            if candidate.weekday() < 5:
                open_dt = candidate.replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                if open_dt > now:
                    break
            days_ahead += 1
        delta = open_dt - now
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        detail = f"Opens in {h}h {m:02d}m"

    return {
        "is_open": is_open,
        "label":   "OPEN" if is_open else "CLOSED",
        "detail":  detail,
        "now_et":  now,
    }


# ── Technical indicators (pure pandas, no ta library dependency) ──────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> tuple[float, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def _bollinger_pct_b(close: pd.Series, window=20, std_dev=2) -> float:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower)
    return float(pct_b.iloc[-1]) if not pct_b.empty else float("nan")


# ── Signal computation ────────────────────────────────────────────────────────

def _score_ticker(hist: pd.DataFrame) -> dict:
    """Compute all signals from a daily OHLCV DataFrame. Returns signal dict."""
    close = hist["Close"].dropna()
    if len(close) < 30:
        return None

    price = float(close.iloc[-1])
    prev  = float(close.iloc[-2]) if len(close) > 1 else price
    chg_1d = (price - prev) / prev if prev != 0 else 0.0

    # Indicators
    rsi = _rsi(close)
    macd_val, macd_sig = _macd(close)
    sma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else float("nan")
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
    bb_pct = _bollinger_pct_b(close)
    mom5   = float((close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else float("nan"))

    # Score each component
    score = 0

    # RSI
    rsi_pts = 0
    if not np.isnan(rsi):
        if rsi < 30:   rsi_pts = +2
        elif rsi < 40: rsi_pts = +1
        elif rsi > 70: rsi_pts = -2
        elif rsi > 60: rsi_pts = -1
    score += rsi_pts

    # MACD
    macd_pts = 0
    if not (np.isnan(macd_val) or np.isnan(macd_sig)):
        macd_pts = +1 if macd_val > macd_sig else -1
    score += macd_pts

    # SMA50
    sma50_pts = 0
    if not np.isnan(sma50):
        sma50_pts = +1 if price > sma50 else -1
    score += sma50_pts

    # SMA200
    sma200_pts = 0
    if not np.isnan(sma200):
        sma200_pts = +1 if price > sma200 else -1
    score += sma200_pts

    # Bollinger %B
    bb_pts = 0
    if not np.isnan(bb_pct):
        if bb_pct < 0.15:   bb_pts = +1
        elif bb_pct > 0.85: bb_pts = -1
    score += bb_pts

    # 5-day momentum
    mom_pts = 0
    if not np.isnan(mom5):
        if mom5 > 0.02:   mom_pts = +1
        elif mom5 < -0.02: mom_pts = -1
    score += mom_pts

    # Composite label
    if score >= 4:    label, emoji = "STRONG BUY",  "🟢"
    elif score >= 2:  label, emoji = "BUY",         "🔵"
    elif score >= -1: label, emoji = "HOLD",        "⚪"
    elif score >= -3: label, emoji = "SELL",        "🟠"
    else:             label, emoji = "STRONG SELL", "🔴"

    return {
        "signal":       f"{emoji} {label}",
        "label":        label,
        "emoji":        emoji,
        "score":        score,
        "current_price": price,
        "change_pct_1d": chg_1d,
        "rsi":          round(rsi, 1)   if not np.isnan(rsi)    else None,
        "macd":         round(macd_val, 4) if not np.isnan(macd_val) else None,
        "macd_signal":  round(macd_sig, 4) if not np.isnan(macd_sig) else None,
        "macd_pts":     macd_pts,
        "sma50":        round(sma50, 2) if not np.isnan(sma50)  else None,
        "sma200":       round(sma200, 2) if not np.isnan(sma200) else None,
        "sma50_pts":    sma50_pts,
        "sma200_pts":   sma200_pts,
        "bb_pct_b":     round(bb_pct, 3) if not np.isnan(bb_pct) else None,
        "bb_pts":       bb_pts,
        "momentum_5d":  round(mom5, 4)  if not np.isnan(mom5)   else None,
        "mom_pts":      mom_pts,
        "rsi_pts":      rsi_pts,
    }


def get_signals(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Compute real-time technical signals for a list of tickers.

    Args:
        tickers: List of ticker symbols (any valid yfinance tickers)
        period:  yfinance history period string (default "1y" for 200d SMA)

    Returns:
        DataFrame indexed by ticker. Columns:
          signal, label, emoji, score, current_price, change_pct_1d,
          rsi, macd, macd_signal, sma50, sma200, bb_pct_b, momentum_5d,
          rsi_pts, macd_pts, sma50_pts, sma200_pts, bb_pts, mom_pts
        Tickers with insufficient data are excluded.
    """
    rows = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if hist.empty:
                print(f"  No data for {ticker}")
                continue
            result = _score_ticker(hist)
            if result:
                rows[ticker] = result
            else:
                print(f"  Insufficient data for {ticker}")
        except Exception as exc:
            print(f"  Error fetching {ticker}: {exc}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).T


# ── Intraday data ─────────────────────────────────────────────────────────────

def fetch_intraday(tickers: list[str], interval: str = "5m") -> dict[str, pd.DataFrame]:
    """
    Fetch today's intraday OHLCV bars for a list of tickers.

    Uses yf.Ticker.history(period="1d", interval=interval).
    Free tier: works for intervals ≥ 1m, limited to last 7 days.

    Args:
        tickers:  List of ticker symbols
        interval: Bar interval — "1m", "5m", "15m", "30m", "60m"

    Returns:
        Dict mapping ticker → DataFrame with DatetimeIndex (ET) and
        columns [Open, High, Low, Close, Volume].
        Missing/errored tickers are omitted from the dict.
    """
    result = {}
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period="1d", interval=interval, auto_adjust=True)
            if df.empty:
                continue
            df.index = pd.to_datetime(df.index)
            # Normalize timezone to ET for display
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert(ET)
            result[ticker] = df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as exc:
            print(f"  Intraday fetch failed for {ticker}: {exc}")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time technical signal generator")
    parser.add_argument("tickers", nargs="*",
                        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help="Ticker symbols (default: 5 portfolio assets)")
    parser.add_argument("--interval", default="5m",
                        help="Intraday bar interval (default: 5m)")
    args = parser.parse_args()

    print("=" * 65)
    print("  Real-Time Signal Generator")
    print("=" * 65)

    # Market status
    ms = market_status()
    status_icon = "🟢" if ms["is_open"] else "🔴"
    print(f"\nMarket: {status_icon} {ms['label']}  |  {ms['detail']}")
    print(f"Time (ET): {ms['now_et'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    # Signals
    print(f"Computing signals for: {args.tickers}")
    df = get_signals(args.tickers)

    if df.empty:
        print("No signals computed — check ticker symbols.")
    else:
        display = df[["signal", "score", "current_price", "change_pct_1d",
                       "rsi", "sma50_pts", "sma200_pts", "bb_pct_b", "momentum_5d"]].copy()
        display["current_price"] = display["current_price"].apply(lambda x: f"${x:.2f}")
        display["change_pct_1d"] = display["change_pct_1d"].apply(
            lambda x: f"{x:+.2%}" if x is not None else "N/A"
        )
        display.columns = ["Signal", "Score", "Price", "1d Chg", "RSI",
                           "vs SMA50", "vs SMA200", "BB%B", "Mom5d"]
        print(display.to_string())

    # Intraday
    print(f"\nFetching intraday data ({args.interval} bars)...")
    intraday = fetch_intraday(args.tickers, interval=args.interval)
    for ticker, bars in intraday.items():
        print(f"\n{ticker} — {len(bars)} bars  |  "
              f"last: ${bars['Close'].iloc[-1]:.2f}  "
              f"({bars.index[-1].strftime('%H:%M ET') if bars.index.tzinfo else bars.index[-1]})")
