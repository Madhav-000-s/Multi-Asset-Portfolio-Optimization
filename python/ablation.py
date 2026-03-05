"""
Ablation Study: DSR-LSTM With vs Without FinBERT Sentiment

Trains two models back-to-back on identical data splits:
  - Baseline : 6 features/asset (no sentiment), input_size=30
  - Sentiment : 7 features/asset (+ FinBERT alpha signal), input_size=35

Saves results to results/ablation.csv and prints a comparison table.

Prerequisites:
  python scraper.py              # download headlines
  python sentiment.py            # run FinBERT → sentiment_scores.parquet
  python ablation.py             # run this script

The baseline model re-uses models/best_model.pt if it already exists
(set --retrain-baseline to force a fresh training run).
"""

import argparse
import copy
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).parent.parent


def _run_train(config: dict, label: str) -> dict:
    """Train one model variant and return metrics dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from features import prepare_data
    from model import create_model_from_config
    from dsr_loss import DSRLoss, compute_sharpe_ratio
    from train import (
        create_sequential_dataloader,
        train_epoch,
        validate,
        EarlyStopping,
    )
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  Sentiment enabled: {config.get('sentiment', {}).get('enabled', False)}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds, val_ds, test_ds, _ = prepare_data(config)
    train_loader = create_sequential_dataloader(
        train_ds, batch_size=config["training"]["batch_size"]
    )
    val_loader = create_sequential_dataloader(
        val_ds, batch_size=config["training"]["batch_size"]
    )
    test_loader = create_sequential_dataloader(
        test_ds, batch_size=config["training"]["batch_size"]
    )

    model = create_model_from_config(config).to(device)
    input_size = next(model.parameters()).shape[-1] if False else \
        (7 if config.get("sentiment", {}).get("enabled", False) else 6) * len(config["assets"]["tickers"])
    print(f"Model input_size: {input_size}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = DSRLoss(eta=config["model"]["dsr_eta"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    ckpt_name = "best_model_sentiment.pt" if config.get("sentiment", {}).get("enabled") else "best_model.pt"
    ckpt_path = models_dir / ckpt_name

    best_val_dsr = float("-inf")
    t0 = time.time()

    for epoch in range(config["training"]["epochs"]):
        train_dsr, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        val_dsr, _, val_sharpe = validate(model, val_loader, criterion, device)
        scheduler.step(val_dsr)

        improved = ""
        if val_dsr > best_val_dsr:
            best_val_dsr = val_dsr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dsr": val_dsr,
                    "val_sharpe": val_sharpe,
                    "config": config,
                },
                ckpt_path,
            )
            improved = " *"

        print(
            f"  Epoch {epoch+1:3d} | train_dsr={train_dsr:8.5f} | "
            f"val_dsr={val_dsr:8.5f} | val_sharpe={val_sharpe:6.3f}{improved}"
        )

        if early_stopping(val_dsr, epoch):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    # Load best & evaluate on test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_dsr, _, test_sharpe = validate(model, test_loader, criterion, device)

    return {
        "variant":       label,
        "sentiment":     config.get("sentiment", {}).get("enabled", False),
        "input_size":    input_size,
        "best_val_dsr":  round(best_val_dsr, 6),
        "test_dsr":      round(test_dsr, 6),
        "test_sharpe":   round(float(test_sharpe), 4),
        "train_time_s":  round(train_time, 1),
        "checkpoint":    str(ckpt_path),
    }


def run_ablation(retrain_baseline: bool = False) -> pd.DataFrame:
    """
    Run both training variants and return a comparison DataFrame.

    Args:
        retrain_baseline: If False and models/best_model.pt exists, skip baseline training.
    """
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    results = []

    # ── Variant 1: Baseline (no sentiment) ───────────────────────────────────
    baseline_ckpt = ROOT / "models/best_model.pt"
    if baseline_ckpt.exists() and not retrain_baseline:
        print(f"\nBaseline checkpoint found at {baseline_ckpt}.")
        print("Loading existing metrics (use --retrain-baseline to force re-training).")
        ckpt = torch.load(baseline_ckpt, map_location="cpu")
        results.append({
            "variant":      "Baseline (no sentiment)",
            "sentiment":    False,
            "input_size":   6 * len(base_config["assets"]["tickers"]),
            "best_val_dsr": round(ckpt.get("val_dsr", float("nan")), 6),
            "test_dsr":     float("nan"),
            "test_sharpe":  round(float(ckpt.get("val_sharpe", float("nan"))), 4),
            "train_time_s": None,
            "checkpoint":   str(baseline_ckpt),
        })
    else:
        cfg_baseline = copy.deepcopy(base_config)
        cfg_baseline.setdefault("sentiment", {})["enabled"] = False
        results.append(_run_train(cfg_baseline, "Baseline (no sentiment)"))

    # ── Variant 2: Sentiment-augmented ───────────────────────────────────────
    cfg_sentiment = copy.deepcopy(base_config)
    cfg_sentiment.setdefault("sentiment", {})["enabled"] = True
    results.append(_run_train(cfg_sentiment, "Sentiment-augmented"))

    # ── Save & print ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_path = ROOT / "results/ablation.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    cols = ["variant", "input_size", "best_val_dsr", "test_dsr", "test_sharpe", "train_time_s"]
    print(df[cols].to_string(index=False))
    print("=" * 60)

    if len(df) == 2:
        dsr_delta = df.iloc[1]["best_val_dsr"] - df.iloc[0]["best_val_dsr"]
        sharpe_delta = df.iloc[1]["test_sharpe"] - df.iloc[0]["test_sharpe"]
        sign_dsr = "+" if dsr_delta >= 0 else ""
        sign_sh = "+" if sharpe_delta >= 0 else ""
        print(f"\nSentiment impact:")
        print(f"  Val DSR  : {sign_dsr}{dsr_delta:.6f}  ({'improved' if dsr_delta > 0 else 'degraded'})")
        print(f"  Test Sharpe: {sign_sh}{sharpe_delta:.4f}  ({'improved' if sharpe_delta > 0 else 'degraded'})")

    print(f"\nFull results saved → {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation: DSR-LSTM with/without sentiment")
    parser.add_argument(
        "--retrain-baseline",
        action="store_true",
        help="Re-train the baseline even if models/best_model.pt exists",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DSR Portfolio Optimizer — Sentiment Ablation Study")
    print("=" * 60)

    run_ablation(retrain_baseline=args.retrain_baseline)
