"""
Prediction Module for DSR Portfolio Optimizer

Provides inference functions for:
- Loading trained model
- Predicting portfolio weights from features
- Interface for R backtesting via reticulate

Functions are designed to be called from R using reticulate.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from model import PortfolioLSTM
from features import (
    load_prices,
    get_tickers_from_columns,
    compute_all_features,
    zscore_normalize_rolling,
    get_feature_tensor,
    load_config
)

# Global cache to avoid reloading model for each prediction
_model = None
_device = None
_config = None
_features_cache = None


def get_config() -> dict:
    """Load and cache configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_device() -> torch.device:
    """Get and cache compute device."""
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def load_model(model_path: str = None) -> PortfolioLSTM:
    """
    Load trained model from checkpoint.

    Uses global cache to avoid reloading for each prediction.

    Args:
        model_path: Path to model checkpoint. Uses default if None.

    Returns:
        Loaded PortfolioLSTM model in eval mode
    """
    global _model

    if _model is not None:
        return _model

    config = get_config()
    device = get_device()

    if model_path is None:
        model_path = Path(__file__).parent.parent / "models/best_model.pt"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    # Initialize model architecture
    num_assets = len(config['assets']['tickers'])
    num_features = 6 * num_assets

    _model = PortfolioLSTM(
        input_size=num_features,
        hidden_size_1=config['model']['lstm_hidden_1'],
        hidden_size_2=config['model']['lstm_hidden_2'],
        num_assets=num_assets,
        dropout=config['model']['dropout']
    ).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()

    val_dsr = checkpoint.get('val_dsr', 'N/A')
    val_sharpe = checkpoint.get('val_sharpe', 'N/A')
    print(f"Loaded model from {model_path}")
    print(f"  Val DSR: {val_dsr:.6f}" if isinstance(val_dsr, float) else f"  Val DSR: {val_dsr}")
    print(f"  Val Sharpe: {val_sharpe:.3f}" if isinstance(val_sharpe, float) else f"  Val Sharpe: {val_sharpe}")

    return _model


def predict_weights(feature_tensor: np.ndarray) -> np.ndarray:
    """
    Predict portfolio weights from feature tensor.

    This is the main function called by R via reticulate.

    Args:
        feature_tensor: (T, F) numpy array of normalized features
                       T = lookback window (default 60)
                       F = num_features (default 30)

    Returns:
        weights: (N,) numpy array of portfolio weights
                N = num_assets (default 5)
                Weights sum to 1, all positive
    """
    model = load_model()
    device = get_device()

    # Convert to tensor and add batch dimension
    x = torch.tensor(feature_tensor, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, T, F)
    x = x.to(device)

    # Predict
    with torch.no_grad():
        weights = model(x)

    return weights.cpu().numpy().squeeze()


def get_cached_features() -> pd.DataFrame:
    """
    Get cached normalized features, computing if needed.

    Returns:
        DataFrame of normalized features
    """
    global _features_cache

    if _features_cache is not None:
        return _features_cache

    # Load and compute features
    project_root = Path(__file__).parent.parent
    prices = load_prices(project_root / "data/raw/prices.parquet")
    features = compute_all_features(prices)
    _features_cache = zscore_normalize_rolling(features)

    return _features_cache


def predict_at_date(
    date: str,
    lookback: int = None
) -> np.ndarray:
    """
    Predict portfolio weights for a specific date.

    This function handles all feature extraction internally,
    making it easier to call from R.

    Args:
        date: Date string for prediction (e.g., "2023-06-15")
        lookback: Number of days to look back (uses config default if None)

    Returns:
        weights: (N,) numpy array of portfolio weights
    """
    config = get_config()
    if lookback is None:
        lookback = config['model']['lookback_window']

    # Get cached features
    features = get_cached_features()

    # Extract feature tensor for the date
    feature_tensor = get_feature_tensor(features, date, lookback)

    return predict_weights(feature_tensor)


def predict_at_index(
    idx: int,
    lookback: int = None
) -> np.ndarray:
    """
    Predict portfolio weights for a specific index in the feature DataFrame.

    Useful for backtesting when iterating through dates.

    Args:
        idx: Index position in the feature DataFrame
        lookback: Number of days to look back

    Returns:
        weights: (N,) numpy array of portfolio weights
    """
    config = get_config()
    if lookback is None:
        lookback = config['model']['lookback_window']

    features = get_cached_features()

    if idx < lookback:
        raise ValueError(f"Index {idx} is too small for lookback={lookback}")

    if idx >= len(features):
        raise ValueError(f"Index {idx} out of bounds (max: {len(features) - 1})")

    # Extract feature tensor
    start_idx = idx - lookback + 1
    feature_tensor = features.iloc[start_idx:idx + 1].values.astype(np.float32)

    return predict_weights(feature_tensor)


def get_tickers() -> list:
    """
    Return list of tickers for R to know asset order.

    Returns:
        List of ticker symbols in the order they appear in weights
    """
    config = get_config()
    return config['assets']['tickers']


def get_feature_dates() -> list:
    """
    Get all available dates for prediction.

    Returns:
        List of date strings
    """
    features = get_cached_features()
    return [str(d.date()) for d in features.index]


def get_valid_prediction_range() -> tuple:
    """
    Get the valid date range for predictions.

    Returns:
        (start_date, end_date) tuple of strings
    """
    config = get_config()
    lookback = config['model']['lookback_window']

    features = get_cached_features()

    # First valid prediction date (need lookback history)
    start_idx = lookback - 1
    start_date = str(features.index[start_idx].date())
    end_date = str(features.index[-1].date())

    return start_date, end_date


def clear_cache():
    """Clear all cached data. Useful for testing or memory management."""
    global _model, _device, _config, _features_cache
    _model = None
    _device = None
    _config = None
    _features_cache = None
    print("Cache cleared")


def reload_model(model_path: str = None):
    """Force reload the model from disk."""
    global _model
    _model = None
    return load_model(model_path)


if __name__ == "__main__":
    print("Testing predict module...")
    print("=" * 50)

    # Load model
    try:
        model = load_model()
        print("\n1. Model loaded successfully")

        # Get tickers
        tickers = get_tickers()
        print(f"\n2. Tickers: {tickers}")

        # Get valid date range
        start_date, end_date = get_valid_prediction_range()
        print(f"\n3. Valid prediction range: {start_date} to {end_date}")

        # Test prediction at a date
        test_date = start_date
        print(f"\n4. Testing prediction at {test_date}...")
        weights = predict_at_date(test_date)
        print(f"   Weights: {dict(zip(tickers, weights.round(4)))}")
        print(f"   Sum: {weights.sum():.6f}")
        print(f"   All positive: {(weights >= 0).all()}")

        # Test prediction with raw tensor
        print("\n5. Testing raw tensor prediction...")
        features = get_cached_features()
        feature_tensor = features.iloc[:60].values.astype(np.float32)
        weights = predict_weights(feature_tensor)
        print(f"   Input shape: {feature_tensor.shape}")
        print(f"   Output shape: {weights.shape}")
        print(f"   Weights: {weights.round(4)}")

        print("\nAll tests passed!")

    except FileNotFoundError as e:
        print(f"\nModel not found: {e}")
        print("Please run train.py first to train the model.")

        # Test with dummy model
        print("\nTesting with dummy predictions...")
        config = get_config()
        tickers = config['assets']['tickers']
        n_assets = len(tickers)

        # Equal weight as dummy
        weights = np.ones(n_assets) / n_assets
        print(f"Dummy equal weights: {dict(zip(tickers, weights.round(4)))}")
