"""
Feature Engineering Module
Computes technical features, normalizes without lookahead bias, and creates PyTorch datasets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prices(parquet_path: str = None) -> pd.DataFrame:
    """
    Load prices.parquet and restructure for feature computation.

    Returns:
        DataFrame with DatetimeIndex, columns like 'AAPL_Close', 'AAPL_High', etc.
    """
    if parquet_path is None:
        parquet_path = Path(__file__).parent.parent / "data/raw/prices.parquet"

    df = pd.read_parquet(parquet_path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    return df


def get_tickers_from_columns(df: pd.DataFrame) -> list[str]:
    """Extract unique ticker symbols from column names."""
    tickers = set()
    for col in df.columns:
        # Columns are like 'AAPL_Close', 'MSFT_High'
        parts = col.split('_')
        if len(parts) >= 2:
            tickers.add(parts[0])
    return sorted(list(tickers))


def compute_log_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Compute log returns for Close prices: ln(P_t / P_{t-1})

    Uses .shift(1) to avoid lookahead bias - return at t uses price at t and t-1.
    """
    returns = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in prices.columns:
            # Log return: ln(P_t / P_{t-1})
            returns[f"{ticker}_log_ret"] = np.log(prices[close_col] / prices[close_col].shift(1))

    return returns


def compute_rolling_volatility(prices: pd.DataFrame, tickers: list[str], window: int = 20) -> pd.DataFrame:
    """
    Rolling standard deviation of log returns over 20 days.
    Annualized by multiplying by sqrt(252).
    """
    volatility = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in prices.columns:
            log_ret = np.log(prices[close_col] / prices[close_col].shift(1))
            # Rolling volatility, annualized
            volatility[f"{ticker}_vol_20d"] = log_ret.rolling(window=window).std() * np.sqrt(252)

    return volatility


def compute_rsi(prices: pd.DataFrame, tickers: list[str], period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index using ta library.
    RSI values range from 0 to 100.
    """
    rsi = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in prices.columns:
            indicator = RSIIndicator(close=prices[close_col], window=period)
            rsi[f"{ticker}_rsi_14"] = indicator.rsi()

    return rsi


def compute_macd(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    MACD signal line using ta library.
    Standard params: fast=12, slow=26, signal=9
    """
    macd_signals = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in prices.columns:
            indicator = MACD(close=prices[close_col], window_slow=26, window_fast=12, window_sign=9)
            macd_signals[f"{ticker}_macd"] = indicator.macd_signal()

    return macd_signals


def compute_bollinger_percent_b(prices: pd.DataFrame, tickers: list[str], window: int = 20) -> pd.DataFrame:
    """
    Bollinger %B = (Price - Lower Band) / (Upper Band - Lower Band)
    Values: 0 = at lower band, 1 = at upper band, can go outside [0,1]
    """
    bb_pct = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in prices.columns:
            indicator = BollingerBands(close=prices[close_col], window=window, window_dev=2)
            bb_pct[f"{ticker}_bb_pct_b"] = indicator.bollinger_pband()

    return bb_pct


def compute_sma_ratio(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Ratio of 50-day SMA to 200-day SMA.
    Values > 1 indicate bullish trend (golden cross territory).
    """
    sma_ratio = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in prices.columns:
            sma_50 = prices[close_col].rolling(window=50).mean()
            sma_200 = prices[close_col].rolling(window=200).mean()
            sma_ratio[f"{ticker}_sma_ratio"] = sma_50 / sma_200

    return sma_ratio


def compute_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all 6 features for all assets.

    Features per asset:
    1. log_ret - Log returns
    2. vol_20d - 20-day rolling volatility (annualized)
    3. rsi_14 - RSI with 14-day period
    4. macd - MACD signal line
    5. bb_pct_b - Bollinger %B
    6. sma_ratio - 50d/200d SMA ratio

    Returns:
        DataFrame shape: (num_days, 6 * num_assets)
    """
    tickers = get_tickers_from_columns(prices)
    print(f"Computing features for {len(tickers)} tickers: {tickers}")

    # Compute each feature type
    log_returns = compute_log_returns(prices, tickers)
    volatility = compute_rolling_volatility(prices, tickers)
    rsi = compute_rsi(prices, tickers)
    macd = compute_macd(prices, tickers)
    bb_pct = compute_bollinger_percent_b(prices, tickers)
    sma_ratio = compute_sma_ratio(prices, tickers)

    # Combine all features
    features = pd.concat([log_returns, volatility, rsi, macd, bb_pct, sma_ratio], axis=1)

    # Reorder columns by ticker for consistency
    ordered_cols = []
    for ticker in tickers:
        for suffix in ['log_ret', 'vol_20d', 'rsi_14', 'macd', 'bb_pct_b', 'sma_ratio']:
            col = f"{ticker}_{suffix}"
            if col in features.columns:
                ordered_cols.append(col)

    features = features[ordered_cols]

    print(f"Features shape before dropna: {features.shape}")

    # Drop rows with NaN (warm-up period for indicators)
    # The longest warm-up is SMA-200, so first 200 rows will have NaN
    features = features.dropna()

    print(f"Features shape after dropna: {features.shape}")

    return features


def zscore_normalize_rolling(features: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Z-score normalize using ROLLING window to prevent lookahead bias.

    For each feature column at time t:
        z_t = (x_t - mean(x_{t-window:t})) / std(x_{t-window:t})

    CRITICAL: Only uses data up to time t, never future data.
    """
    normalized = pd.DataFrame(index=features.index, columns=features.columns)

    for col in features.columns:
        rolling_mean = features[col].rolling(window=window, min_periods=window).mean()
        rolling_std = features[col].rolling(window=window, min_periods=window).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-8)

        normalized[col] = (features[col] - rolling_mean) / rolling_std

    # Drop NaN rows from warm-up period
    normalized = normalized.dropna()

    print(f"Normalized features shape: {normalized.shape}")

    return normalized.astype(np.float32)


def compute_forward_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Compute forward returns for each asset: r_{t+1} = (P_{t+1} - P_t) / P_t

    These are the returns we want to predict - the return from t to t+1.
    """
    returns = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in prices.columns:
            # Forward return: what we earn if we hold from t to t+1
            returns[ticker] = prices[close_col].pct_change().shift(-1)

    return returns


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for LSTM training with sliding window.

    Each sample contains:
    - X: (T, F) tensor - features from [idx : idx+T]
    - y: (N,) tensor - next-day returns for N assets at idx+T
    """

    def __init__(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 60
    ):
        """
        Args:
            features: Normalized feature DataFrame (num_days, num_features)
            returns: Forward returns DataFrame (num_days, num_assets)
                    Should already be aligned with features
            lookback: Sequence length T
        """
        self.lookback = lookback

        # Convert to numpy (assume already aligned)
        self.features = features.values.astype(np.float32)
        self.returns = returns.values.astype(np.float32)
        self.dates = features.index.tolist()

        # Valid indices for sampling - need lookback days of history
        self.valid_length = max(0, len(self.features) - lookback)

        print(f"Dataset: {self.valid_length} samples, lookback={lookback}, "
              f"features={self.features.shape[1]}, assets={self.returns.shape[1]}")

    def __len__(self) -> int:
        return self.valid_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: (T, F) feature tensor
            y: (N,) forward returns tensor
        """
        # Features from [idx : idx + lookback]
        X = torch.tensor(self.features[idx : idx + self.lookback], dtype=torch.float32)

        # Return at the end of the lookback window
        y = torch.tensor(self.returns[idx + self.lookback], dtype=torch.float32)

        return X, y

    def get_date(self, idx: int) -> pd.Timestamp:
        """Get the date for a given index (end of lookback window)."""
        return self.dates[idx + self.lookback]


def create_train_val_test_splits(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """
    Split data by DATE (not random) to prevent data leakage.

    Each split includes lookback days of overlap for context.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    n = len(features)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split features - include lookback overlap for val and test
    train_features = features.iloc[:train_end]
    train_returns = returns.iloc[:train_end]

    # Val needs lookback days before it for context
    val_start = max(0, train_end - lookback)
    val_features = features.iloc[val_start:val_end]
    val_returns = returns.iloc[val_start:val_end]

    # Test needs lookback days before it for context
    test_start = max(0, val_end - lookback)
    test_features = features.iloc[test_start:]
    test_returns = returns.iloc[test_start:]

    print(f"\nData splits (with lookback overlap for val/test):")
    print(f"  Train: {train_features.index[0]} to {train_features.index[-1]} ({len(train_features)} days)")
    print(f"  Val:   {val_features.index[0]} to {val_features.index[-1]} ({len(val_features)} days, effective: {len(val_features) - lookback})")
    print(f"  Test:  {test_features.index[0]} to {test_features.index[-1]} ({len(test_features)} days, effective: {len(test_features) - lookback})")

    train_dataset = SequenceDataset(train_features, train_returns, lookback)
    val_dataset = SequenceDataset(val_features, val_returns, lookback)
    test_dataset = SequenceDataset(test_features, test_returns, lookback)

    return train_dataset, val_dataset, test_dataset


def get_feature_tensor(
    features: pd.DataFrame,
    end_date: str,
    lookback: int = 60
) -> np.ndarray:
    """
    Extract feature tensor ending at a specific date.

    Used by predict.py during backtesting.

    Args:
        features: Normalized feature DataFrame
        end_date: Date string for the end of lookback window
        lookback: Number of days to look back

    Returns:
        (T, F) numpy array
    """
    # Find the index for end_date
    if end_date not in features.index:
        # Find nearest date
        end_date = features.index[features.index.get_indexer([end_date], method='ffill')[0]]

    end_idx = features.index.get_loc(end_date)
    start_idx = end_idx - lookback + 1

    if start_idx < 0:
        raise ValueError(f"Not enough history before {end_date} for lookback={lookback}")

    return features.iloc[start_idx:end_idx + 1].values.astype(np.float32)


def prepare_data(config: dict = None) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset, pd.DataFrame]:
    """
    Complete data preparation pipeline.

    Returns:
        train_dataset, val_dataset, test_dataset, normalized_features
    """
    if config is None:
        config = load_config()

    # Load prices
    project_root = Path(__file__).parent.parent
    prices = load_prices(project_root / "data/raw/prices.parquet")

    # Get tickers
    tickers = get_tickers_from_columns(prices)

    # Compute features
    features = compute_all_features(prices)

    # Normalize
    normalized = zscore_normalize_rolling(features)

    # Compute forward returns and align with normalized features
    returns = compute_forward_returns(prices, tickers)

    # IMPORTANT: Align returns to normalized features index
    common_idx = normalized.index.intersection(returns.index)
    normalized = normalized.loc[common_idx]
    returns = returns.loc[common_idx]

    # Drop last row (forward return not available for last date)
    normalized = normalized.iloc[:-1]
    returns = returns.iloc[:-1]

    print(f"Aligned data: {len(normalized)} samples")

    # Create datasets
    train_ds, val_ds, test_ds = create_train_val_test_splits(
        normalized,
        returns,
        lookback=config['model']['lookback_window'],
        train_ratio=config['training']['train_ratio'],
        val_ratio=config['training']['val_ratio']
    )

    return train_ds, val_ds, test_ds, normalized


if __name__ == "__main__":
    # Test the module
    print("Testing features module...")

    config = load_config()
    prices = load_prices()

    print(f"\nPrices shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

    tickers = get_tickers_from_columns(prices)
    print(f"Tickers: {tickers}")

    # Compute features
    features = compute_all_features(prices)
    print(f"\nFeatures columns: {features.columns.tolist()}")

    # Normalize
    normalized = zscore_normalize_rolling(features)
    print(f"\nNormalized stats:")
    print(normalized.describe())

    # Compute returns
    returns = compute_forward_returns(prices, tickers)
    print(f"\nReturns shape: {returns.shape}")

    # Create datasets
    train_ds, val_ds, test_ds = create_train_val_test_splits(
        normalized, returns,
        lookback=config['model']['lookback_window']
    )

    # Test a sample
    X, y = train_ds[0]
    print(f"\nSample X shape: {X.shape}")
    print(f"Sample y shape: {y.shape}")
    print(f"Sample y: {y}")
