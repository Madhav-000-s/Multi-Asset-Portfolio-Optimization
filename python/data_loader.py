"""
Data Loader Module
Downloads OHLCV data from Yahoo Finance and saves to parquet format.
"""

import os
import yaml
import pandas as pd
import yfinance as yf
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download OHLCV data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV)
    """
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True
    )

    return data


def clean_prices(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Clean and validate price data.

    Args:
        df: Raw price DataFrame from yfinance
        tickers: List of ticker symbols

    Returns:
        Cleaned DataFrame with forward-filled missing values
    """
    # Handle single ticker case (no MultiIndex)
    if len(tickers) == 1:
        df.columns = pd.MultiIndex.from_product([df.columns, tickers])

    # Reorder columns to have ticker as first level
    df = df.swaplevel(axis=1).sort_index(axis=1)

    # Forward fill missing values (weekends/holidays already excluded by yfinance)
    df = df.ffill()

    # Drop any remaining NaN rows at the start
    df = df.dropna()

    # Validate data
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values per column:\n{df.isna().sum().sum()}")

    return df


def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to parquet format."""
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Flatten MultiIndex columns for parquet compatibility
    df_flat = df.copy()
    df_flat.columns = ['_'.join(col).strip() for col in df_flat.columns.values]

    df_flat.to_parquet(output_path, engine='pyarrow')
    print(f"\nSaved to {output_path}")


def main():
    """Main entry point for data loading."""
    # Get project root (one level up from python/)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"

    # Load configuration
    config = load_config(config_path)

    tickers = config["assets"]["tickers"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    raw_path = project_root / config["data"]["raw_path"]

    # Download data
    raw_data = download_prices(tickers, start_date, end_date)

    # Clean data
    clean_data = clean_prices(raw_data, tickers)

    # Save to parquet
    output_file = raw_path / "prices.parquet"
    save_to_parquet(clean_data, output_file)

    print("\nData loading complete!")
    print(f"Tickers: {tickers}")
    print(f"Records: {len(clean_data)}")

    return clean_data


if __name__ == "__main__":
    main()
