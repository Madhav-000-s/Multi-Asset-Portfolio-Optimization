"""
FinBERT Sentiment Scoring Pipeline

Loads headlines from data/raw/headlines.csv, scores each one with
ProsusAI/finbert, aggregates to a daily per-ticker alpha signal, and
saves the result to data/processed/sentiment_scores.parquet.

Alpha signal = P(positive) − P(negative)  ∈ [−1, +1]
  +1 = fully positive   0 = neutral   −1 = fully negative

Missing days are forward-filled (sentiment persists until updated) and
any remaining NaN (beginning of history) is filled with 0.0 (neutral).

Usage:
  python sentiment.py           # uses config.yaml assets + default paths
  python sentiment.py --help    # see options
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent


# ── FinBERT loader ────────────────────────────────────────────────────────────

def load_finbert(model_name: str = "ProsusAI/finbert", device: str = None):
    """Download (first time) and load FinBERT tokenizer + model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"FinBERT ready — {n_params / 1e6:.1f}M parameters")
    return tokenizer, model, device


# ── Batch scoring ─────────────────────────────────────────────────────────────

@torch.no_grad()
def score_headlines(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Score a list of headline strings with FinBERT.

    ProsusAI/finbert label order:  0 = positive · 1 = negative · 2 = neutral

    Returns:
        np.ndarray of shape (N, 3): [P(positive), P(negative), P(neutral)]
    """
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

        done = min(start + batch_size, len(texts))
        print(f"  Scored {done}/{len(texts)} headlines...", end="\r", flush=True)

    print()  # newline after progress bar
    return np.vstack(all_probs)  # (N, 3)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def compute_sentiment_scores(
    headlines_path: str = None,
    output_path: str = None,
    assets: list[str] = None,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Full pipeline: load headlines → FinBERT → aggregate → save.

    Args:
        headlines_path: Path to headlines.csv  (default: data/raw/headlines.csv)
        output_path:    Path to save parquet   (default: data/processed/sentiment_scores.parquet)
        assets:         Ordered ticker list; missing columns are added with 0.0
        model_name:     HuggingFace model identifier
        batch_size:     Headlines per FinBERT forward pass

    Returns:
        DataFrame (DatetimeIndex × tickers) with daily sentiment scores in [−1, +1]
    """
    if headlines_path is None:
        headlines_path = ROOT / "data/raw/headlines.csv"
    if output_path is None:
        output_path = ROOT / "data/processed/sentiment_scores.parquet"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(headlines_path).exists():
        raise FileNotFoundError(
            f"Headlines file not found: {headlines_path}\n"
            "Run  python scraper.py  first to download headlines."
        )

    # ── Load & clean headlines ────────────────────────────────────────────────
    df = pd.read_csv(headlines_path, parse_dates=["date"])
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df = df.dropna(subset=["headline"])
    df = df[df["headline"].str.strip() != ""].copy()

    print(f"Loaded {len(df):,} headlines  |  {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── Score with FinBERT ────────────────────────────────────────────────────
    tokenizer, model, device = load_finbert(model_name)
    probs = score_headlines(df["headline"].tolist(), tokenizer, model, device, batch_size)

    # ProsusAI/finbert: label 0 = positive, 1 = negative, 2 = neutral
    df["p_pos"] = probs[:, 0]
    df["p_neg"] = probs[:, 1]
    df["sentiment"] = df["p_pos"] - df["p_neg"]  # alpha signal ∈ [−1, +1]

    # ── Aggregate: mean sentiment per (date, ticker) ──────────────────────────
    daily = (
        df.groupby(["date", "ticker"])["sentiment"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="ticker", values="sentiment")
    )
    daily.index = pd.to_datetime(daily.index)
    daily.columns.name = None
    daily = daily.sort_index()

    # Ensure all asset columns exist (add NaN for tickers with no headlines)
    if assets:
        for t in assets:
            if t not in daily.columns:
                daily[t] = np.nan
        daily = daily[assets]  # canonical order

    # Forward-fill within each asset, then neutral-fill any remaining NaN
    daily = daily.ffill().fillna(0.0)

    # ── Save ──────────────────────────────────────────────────────────────────
    daily.to_parquet(output_path, engine="pyarrow")

    print(f"\nSentiment scores saved → {output_path}")
    print(f"  Shape      : {daily.shape}")
    print(f"  Date range : {daily.index.min().date()} → {daily.index.max().date()}")
    print(f"  Score range: [{daily.values.min():.3f}, {daily.values.max():.3f}]")
    print("  Mean per ticker:")
    for t in daily.columns:
        print(f"    {t:6s}: {daily[t].mean():+.4f}  std={daily[t].std():.4f}")

    return daily


# ── Convenience loader (used by features.py) ──────────────────────────────────

def load_sentiment_scores(path: str = None) -> pd.DataFrame:
    """
    Load pre-computed sentiment scores from parquet.
    Returns empty DataFrame if the file doesn't exist.
    """
    if path is None:
        path = ROOT / "data/processed/sentiment_scores.parquet"

    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment scoring pipeline")
    parser.add_argument("--headlines", default=None, help="Path to headlines.csv")
    parser.add_argument("--output", default=None, help="Path to save sentiment_scores.parquet")
    parser.add_argument("--model", default="ProsusAI/finbert", help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=32, help="FinBERT batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("DSR Portfolio Optimizer — FinBERT Sentiment Scoring")
    print("=" * 60)

    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assets = config["assets"]["tickers"]
    model_name = config.get("sentiment", {}).get("model_name", args.model)
    batch_size = config.get("sentiment", {}).get("batch_size", args.batch_size)

    daily = compute_sentiment_scores(
        headlines_path=args.headlines,
        output_path=args.output,
        assets=assets,
        model_name=model_name,
        batch_size=batch_size,
    )

    print("\nSample scores (last 10 days):")
    print(daily.tail(10).round(4).to_string())
