"""
Headline Scraper for DSR Portfolio Optimizer

Downloads financial news headlines for the portfolio tickers.

Sources:
  1. yfinance Ticker.news  — free, ~30 recent articles per ticker
  2. Alpha Vantage NEWS_SENTIMENT — free API key, historical coverage 2018–present

Output: data/raw/headlines.csv
  columns: date (YYYY-MM-DD), ticker, headline, source

Usage:
  python scraper.py                          # yfinance only
  python scraper.py --av-key YOUR_KEY_HERE  # + Alpha Vantage historical
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import date
import yaml

ROOT = Path(__file__).parent.parent


def fetch_yfinance_news(tickers: list[str]) -> pd.DataFrame:
    """Fetch recent news headlines via yfinance (no API key needed)."""
    import yfinance as yf

    rows = []
    for ticker in tickers:
        try:
            news = yf.Ticker(ticker).news
            for item in (news or []):
                title = item.get("title", "").strip()
                ts = item.get("providerPublishTime") or item.get("publishTime")
                if title and ts:
                    rows.append({
                        "date": pd.to_datetime(ts, unit="s").date(),
                        "ticker": ticker,
                        "headline": title,
                        "source": "yfinance",
                    })
        except Exception as exc:
            print(f"  yfinance news failed for {ticker}: {exc}")

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "ticker", "headline", "source"]
    )


def fetch_alpha_vantage_news(
    tickers: list[str],
    api_key: str,
    start: str = "20180101",
    end: str = None,
) -> pd.DataFrame:
    """
    Fetch historical news via Alpha Vantage NEWS_SENTIMENT endpoint.
    Free tier: 25 requests/day, up to 1000 articles per request.

    Args:
        tickers: List of ticker symbols
        api_key: Alpha Vantage free API key
        start: Start date YYYYMMDD
        end: End date YYYYMMDD (defaults to today)
    """
    import requests

    if end is None:
        end = date.today().strftime("%Y%m%d")

    rows = []
    base = "https://www.alphavantage.co/query"

    for ticker in tickers:
        print(f"  Fetching Alpha Vantage headlines for {ticker}...")
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": f"{start}T0000",
            "time_to": f"{end}T2359",
            "limit": 1000,
            "apikey": api_key,
        }
        try:
            r = requests.get(base, params=params, timeout=30)
            data = r.json()

            # Free-tier rate-limit message
            if "Information" in data:
                print(f"  Alpha Vantage limit reached: {data['Information']}")
                break

            for item in data.get("feed", []):
                title = item.get("title", "").strip()
                time_pub = item.get("time_published", "")
                if title and time_pub:
                    dt = pd.to_datetime(time_pub, format="%Y%m%dT%H%M%S").date()
                    rows.append({
                        "date": dt,
                        "ticker": ticker,
                        "headline": title,
                        "source": "alphavantage",
                    })
        except Exception as exc:
            print(f"  Alpha Vantage failed for {ticker}: {exc}")

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "ticker", "headline", "source"]
    )


def download_headlines(config_path: str = None, av_key_override: str = None) -> pd.DataFrame:
    """
    Download headlines from all configured sources and save to headlines.csv.
    New records are merged without duplication.

    Args:
        config_path: Path to config.yaml (auto-detected if None)
        av_key_override: Override Alpha Vantage key from command-line

    Returns:
        DataFrame with all headlines
    """
    if config_path is None:
        config_path = ROOT / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tickers = config["assets"]["tickers"]
    sentiment_cfg = config.get("sentiment", {})
    av_key = av_key_override or sentiment_cfg.get("alphavantage_key")
    start = str(config["data"]["start_date"]).replace("-", "")
    end = str(config["data"]["end_date"]).replace("-", "")

    out_path = ROOT / "data/raw/headlines.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing headlines to deduplicate
    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["date"])
        print(f"Loaded {len(existing)} existing headlines.")
    else:
        existing = pd.DataFrame(columns=["date", "ticker", "headline", "source"])

    new_frames = [existing]

    # 1. yfinance news (free, recent ~30 articles per ticker)
    print("\n[1/2] Fetching recent headlines via yfinance...")
    yf_df = fetch_yfinance_news(tickers)
    if not yf_df.empty:
        new_frames.append(yf_df)
        print(f"      Got {len(yf_df)} headlines")
    else:
        print("      No headlines returned.")

    # 2. Alpha Vantage (historical, optional)
    if av_key and str(av_key).strip() not in ("", "null", "None", "YOUR_KEY_HERE"):
        print("\n[2/2] Fetching historical headlines via Alpha Vantage...")
        av_df = fetch_alpha_vantage_news(tickers, av_key, start=start, end=end)
        if not av_df.empty:
            new_frames.append(av_df)
            print(f"      Got {len(av_df)} headlines")
    else:
        print("\n[2/2] Alpha Vantage key not set — skipping historical headlines.")
        print("      Add sentiment.alphavantage_key to config.yaml for 2018–2024 coverage.")

    combined = pd.concat(new_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    combined = (
        combined
        .drop_duplicates(subset=["date", "ticker", "headline"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    combined.to_csv(out_path, index=False)

    print(f"\nHeadlines saved → {out_path}")
    print(f"  Total   : {len(combined):,} rows")
    print(f"  Tickers : {sorted(combined['ticker'].unique().tolist())}")
    if not combined.empty:
        print(f"  Dates   : {combined['date'].min()} → {combined['date'].max()}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download financial headlines")
    parser.add_argument(
        "--av-key",
        default=None,
        help="Alpha Vantage free API key (alphavantage.co)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DSR Portfolio Optimizer — Headline Scraper")
    print("=" * 60)

    df = download_headlines(av_key_override=args.av_key)

    if not df.empty:
        print("\nSample headlines:")
        print(df.sample(min(5, len(df)))[["date", "ticker", "headline"]].to_string(index=False))
