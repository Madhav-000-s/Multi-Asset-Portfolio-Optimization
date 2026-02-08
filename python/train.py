"""
Training Module for DSR Portfolio Optimizer

Implements the training loop with:
- Sequential processing (required for DSR state)
- Adam optimizer with ReduceLROnPlateau
- Early stopping on validation DSR
- Gradient clipping for stability
- Model checkpointing
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import time
from datetime import datetime

from features import prepare_data, load_config
from model import PortfolioLSTM, create_model_from_config
from dsr_loss import DSRLoss, compute_sharpe_ratio


class EarlyStopping:
    """
    Stop training when validation DSR stops improving.

    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        best_score: Best validation DSR seen
        counter: Epochs since last improvement
        should_stop: Whether training should stop
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement required
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('-inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, val_dsr: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            val_dsr: Validation DSR (higher is better)
            epoch: Current epoch number

        Returns:
            True if should stop, False otherwise
        """
        if val_dsr > self.best_score + self.min_delta:
            self.best_score = val_dsr
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


def create_sequential_dataloader(
    dataset,
    batch_size: int,
    drop_last: bool = False
) -> DataLoader:
    """
    Create DataLoader that preserves temporal order.

    CRITICAL: shuffle=False to maintain sequence order for DSR.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # MUST be False for DSR
        drop_last=drop_last,
        num_workers=0  # Avoid multiprocessing issues with sequential data
    )


def train_epoch(
    model: PortfolioLSTM,
    dataloader: DataLoader,
    criterion: DSRLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> tuple[float, float]:
    """
    Train for one epoch.

    CRITICAL: Data is processed SEQUENTIALLY. DSR state persists across batches.

    Args:
        model: The PortfolioLSTM model
        dataloader: Training data loader (must be sequential)
        criterion: DSR loss function
        optimizer: Optimizer
        device: Device to use
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        (mean_dsr, mean_loss): Average DSR and loss over epoch
    """
    model.train()
    criterion.reset_state()  # Reset A, B at epoch start

    total_loss = 0.0
    total_dsr = 0.0
    num_batches = 0

    for batch_idx, (features, next_returns) in enumerate(dataloader):
        features = features.to(device)
        next_returns = next_returns.to(device)

        optimizer.zero_grad()

        # Forward pass
        weights = model(features)  # (batch, N)

        # Compute DSR loss
        loss = criterion(weights, next_returns)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_dsr += -loss.item()  # DSR is negative of loss
        num_batches += 1

    mean_loss = total_loss / num_batches
    mean_dsr = total_dsr / num_batches

    return mean_dsr, mean_loss


def validate(
    model: PortfolioLSTM,
    dataloader: DataLoader,
    criterion: DSRLoss,
    device: torch.device
) -> tuple[float, float, float]:
    """
    Validate model on held-out data.

    Args:
        model: The PortfolioLSTM model
        dataloader: Validation data loader
        criterion: DSR loss function
        device: Device to use

    Returns:
        (mean_dsr, mean_loss, sharpe_estimate): Validation metrics
    """
    model.eval()
    criterion.reset_state()

    total_loss = 0.0
    total_dsr = 0.0
    num_batches = 0
    all_portfolio_returns = []

    with torch.no_grad():
        for features, next_returns in dataloader:
            features = features.to(device)
            next_returns = next_returns.to(device)

            weights = model(features)
            loss = criterion(weights, next_returns)

            # Track portfolio returns for Sharpe calculation
            portfolio_returns = (weights * next_returns).sum(dim=1)
            all_portfolio_returns.append(portfolio_returns)

            total_loss += loss.item()
            total_dsr += -loss.item()
            num_batches += 1

    mean_loss = total_loss / num_batches
    mean_dsr = total_dsr / num_batches

    # Compute Sharpe ratio from all returns
    all_returns = torch.cat(all_portfolio_returns)
    sharpe = compute_sharpe_ratio(all_returns, annualize=True)

    return mean_dsr, mean_loss, sharpe


def train(
    config: dict = None,
    verbose: bool = True
) -> tuple[PortfolioLSTM, dict]:
    """
    Main training function.

    Args:
        config: Configuration dictionary (loads from file if None)
        verbose: Whether to print progress

    Returns:
        (model, history): Trained model and training history
    """
    if config is None:
        config = load_config()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # Prepare data
    if verbose:
        print("\nPreparing data...")
    train_dataset, val_dataset, test_dataset, _ = prepare_data(config)

    # Create dataloaders (NO SHUFFLING)
    train_loader = create_sequential_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size']
    )
    val_loader = create_sequential_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )

    # Initialize model
    model = create_model_from_config(config).to(device)
    if verbose:
        print(f"\n{model.summary()}")

    # Loss, optimizer, scheduler
    criterion = DSRLoss(eta=config['model']['dsr_eta'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize DSR
        factor=0.5,
        patience=5
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience']
    )

    # Create models directory
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Training history
    history = {
        'train_dsr': [],
        'train_loss': [],
        'val_dsr': [],
        'val_loss': [],
        'val_sharpe': [],
        'lr': []
    }

    best_val_dsr = float('-inf')
    best_model_path = models_dir / "best_model.pt"

    if verbose:
        print(f"\nStarting training for {config['training']['epochs']} epochs...")
        print("-" * 70)

    start_time = time.time()

    for epoch in range(config['training']['epochs']):
        epoch_start = time.time()

        # Train
        train_dsr, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_dsr, val_loss, val_sharpe = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_dsr)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_dsr'].append(train_dsr)
        history['train_loss'].append(train_loss)
        history['val_dsr'].append(val_dsr)
        history['val_loss'].append(val_loss)
        history['val_sharpe'].append(val_sharpe)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Log progress
        if verbose:
            improved = ""
            if val_dsr > best_val_dsr:
                improved = " *"

            print(
                f"Epoch {epoch + 1:3d}/{config['training']['epochs']} | "
                f"Train DSR: {train_dsr:8.5f} | "
                f"Val DSR: {val_dsr:8.5f} | "
                f"Val Sharpe: {val_sharpe:6.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s{improved}"
            )

        # Save best model
        if val_dsr > best_val_dsr:
            best_val_dsr = val_dsr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsr': val_dsr,
                'val_sharpe': val_sharpe,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }, best_model_path)

        # Early stopping check
        if early_stopping(val_dsr, epoch):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best validation DSR: {best_val_dsr:.6f} at epoch {early_stopping.best_epoch + 1}")
            break

    total_time = time.time() - start_time

    if verbose:
        print("-" * 70)
        print(f"Training complete in {total_time / 60:.1f} minutes")
        print(f"Best validation DSR: {best_val_dsr:.6f}")
        print(f"Model saved to: {best_model_path}")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history


def evaluate_on_test(
    model: PortfolioLSTM,
    config: dict = None,
    verbose: bool = True
) -> dict:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained PortfolioLSTM
        config: Configuration dictionary
        verbose: Whether to print results

    Returns:
        Dictionary with test metrics
    """
    if config is None:
        config = load_config()

    device = next(model.parameters()).device

    # Prepare test data
    _, _, test_dataset, _ = prepare_data(config)
    test_loader = create_sequential_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size']
    )

    criterion = DSRLoss(eta=config['model']['dsr_eta'])

    # Evaluate
    test_dsr, test_loss, test_sharpe = validate(
        model, test_loader, criterion, device
    )

    results = {
        'test_dsr': test_dsr,
        'test_loss': test_loss,
        'test_sharpe': test_sharpe
    }

    if verbose:
        print("\nTest Set Results:")
        print(f"  DSR: {test_dsr:.6f}")
        print(f"  Sharpe Ratio (annualized): {test_sharpe:.3f}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("DSR Portfolio Optimizer - Training")
    print("=" * 70)

    # Load config
    config = load_config()
    print(f"\nConfiguration:")
    print(f"  Assets: {config['assets']['tickers']}")
    print(f"  Lookback: {config['model']['lookback_window']} days")
    print(f"  LSTM: {config['model']['lstm_hidden_1']} -> {config['model']['lstm_hidden_2']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Max epochs: {config['training']['epochs']}")

    # Train
    model, history = train(config, verbose=True)

    # Evaluate on test set
    evaluate_on_test(model, config, verbose=True)

    print("\nTraining complete!")
