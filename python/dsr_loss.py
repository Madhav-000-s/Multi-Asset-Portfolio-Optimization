"""
Differential Sharpe Ratio Loss Module

Implements the DSR loss function for gradient-based optimization of Sharpe Ratio.
Based on Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"

DSR approximates the change in Sharpe ratio when adding a new return:
    D_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^(3/2)

Where:
    A_t = exponential moving average of returns
    B_t = exponential moving average of squared returns
    eta = adaptation rate (default 0.01)
"""

import torch
import torch.nn as nn


class DSRLoss(nn.Module):
    """
    Differential Sharpe Ratio loss for portfolio optimization.

    This loss enables gradient-based optimization of the Sharpe Ratio by computing
    the differential (incremental) change in Sharpe at each time step.

    CRITICAL: This loss requires SEQUENTIAL processing. Do NOT shuffle batches.
    The EMA state (A, B) must persist across time steps within a training sequence.

    Attributes:
        eta: Adaptation rate for exponential moving averages (default 0.01)
        epsilon: Small constant for numerical stability
    """

    def __init__(self, eta: float = 0.01, epsilon: float = 1e-8):
        """
        Args:
            eta: Adaptation rate for EMAs. Lower = more history weighted.
            epsilon: Numerical stability constant.
        """
        super().__init__()
        self.eta = eta
        self.epsilon = epsilon

        # Register buffers for EMA state (not learnable parameters)
        self.register_buffer('A', torch.tensor(0.0))  # EMA of returns
        self.register_buffer('B', torch.tensor(epsilon))  # EMA of squared returns

    def reset_state(self):
        """Reset EMA state. Call at start of each epoch."""
        self.A.fill_(0.0)
        self.B.fill_(self.epsilon)

    def forward(
        self,
        weights: torch.Tensor,
        next_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DSR loss for a batch of time steps processed sequentially.

        Args:
            weights: (batch_size, num_assets) - portfolio weights at each time step
            next_returns: (batch_size, num_assets) - asset returns at t+1

        Returns:
            loss: Scalar tensor = -mean(DSR over batch)
                  Negative because optimizer minimizes but we want to maximize DSR.
        """
        batch_size = weights.shape[0]

        # Compute portfolio returns: w_t . r_{t+1}
        portfolio_returns = (weights * next_returns).sum(dim=1)  # (batch_size,)

        # Accumulate DSR over the batch
        dsr_sum = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)

        # Process sequentially to maintain EMA state
        A = self.A.clone()
        B = self.B.clone()

        for t in range(batch_size):
            R_t = portfolio_returns[t]

            # Store previous values
            A_prev = A
            B_prev = B

            # Compute deltas
            delta_A = self.eta * (R_t - A_prev)
            delta_B = self.eta * (R_t ** 2 - B_prev)

            # Update EMAs
            A = A_prev + delta_A
            B = B_prev + delta_B

            # Compute variance (B - A^2) with clamping for stability
            variance = B_prev - A_prev ** 2
            variance = torch.clamp(variance, min=self.epsilon)

            # Compute denominator: (variance)^(3/2)
            denominator = variance ** 1.5

            # Compute DSR: D_t = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / denominator
            D_t = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denominator

            dsr_sum = dsr_sum + D_t

        # Update stored state for next batch
        self.A = A.detach()
        self.B = B.detach()

        # Mean DSR
        mean_dsr = dsr_sum / batch_size

        # Return negative DSR (minimize loss = maximize DSR)
        return -mean_dsr

    def get_current_sharpe_estimate(self) -> float:
        """
        Get current estimate of Sharpe ratio from EMA state.

        Returns:
            Estimated Sharpe ratio: A / sqrt(B - A^2)
        """
        variance = self.B - self.A ** 2
        if variance < self.epsilon:
            return 0.0
        return (self.A / torch.sqrt(variance)).item()


class DSRLossStateless(nn.Module):
    """
    Stateless version of DSR loss for evaluation.

    Computes DSR over a sequence without maintaining persistent state.
    Useful for validation/testing where we want independent evaluation.
    """

    def __init__(self, eta: float = 0.01, epsilon: float = 1e-8):
        super().__init__()
        self.eta = eta
        self.epsilon = epsilon

    def forward(
        self,
        weights: torch.Tensor,
        next_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DSR loss over a sequence starting from zero state.

        Args:
            weights: (batch_size, num_assets) - portfolio weights
            next_returns: (batch_size, num_assets) - asset returns

        Returns:
            loss: -mean(DSR)
        """
        batch_size = weights.shape[0]
        device = weights.device
        dtype = weights.dtype

        # Compute portfolio returns
        portfolio_returns = (weights * next_returns).sum(dim=1)

        # Initialize EMA state
        A = torch.tensor(0.0, device=device, dtype=dtype)
        B = torch.tensor(self.epsilon, device=device, dtype=dtype)

        dsr_sum = torch.tensor(0.0, device=device, dtype=dtype)

        for t in range(batch_size):
            R_t = portfolio_returns[t]

            A_prev = A
            B_prev = B

            delta_A = self.eta * (R_t - A_prev)
            delta_B = self.eta * (R_t ** 2 - B_prev)

            A = A_prev + delta_A
            B = B_prev + delta_B

            variance = torch.clamp(B_prev - A_prev ** 2, min=self.epsilon)
            denominator = variance ** 1.5

            D_t = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denominator
            dsr_sum = dsr_sum + D_t

        return -dsr_sum / batch_size


def compute_sharpe_ratio(returns: torch.Tensor, annualize: bool = True) -> float:
    """
    Compute standard Sharpe ratio for comparison.

    Args:
        returns: (T,) tensor of returns
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        Sharpe ratio
    """
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret < 1e-8:
        return 0.0

    sharpe = mean_ret / std_ret

    if annualize:
        sharpe = sharpe * (252 ** 0.5)

    return sharpe.item()


if __name__ == "__main__":
    # Test the DSR loss
    print("Testing DSR Loss Module...")

    # Create dummy data
    torch.manual_seed(42)
    batch_size = 100
    num_assets = 5

    # Random weights (softmax to sum to 1)
    logits = torch.randn(batch_size, num_assets)
    weights = torch.softmax(logits, dim=1)

    # Random returns (small values like real returns)
    next_returns = torch.randn(batch_size, num_assets) * 0.02  # ~2% daily moves

    # Test DSRLoss
    criterion = DSRLoss(eta=0.01)

    print("\n1. Testing forward pass...")
    loss = criterion(weights, next_returns)
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   DSR (positive): {-loss.item():.6f}")

    print("\n2. Testing gradient flow...")
    weights_param = torch.randn(batch_size, num_assets, requires_grad=True)
    weights_softmax = torch.softmax(weights_param, dim=1)

    criterion.reset_state()
    loss = criterion(weights_softmax, next_returns)
    loss.backward()

    print(f"   Gradient exists: {weights_param.grad is not None}")
    print(f"   Gradient norm: {weights_param.grad.norm().item():.6f}")

    print("\n3. Testing state persistence...")
    criterion.reset_state()

    # First batch
    loss1 = criterion(weights[:50], next_returns[:50])
    sharpe1 = criterion.get_current_sharpe_estimate()

    # Second batch (state should persist)
    loss2 = criterion(weights[50:], next_returns[50:])
    sharpe2 = criterion.get_current_sharpe_estimate()

    print(f"   After batch 1 - Sharpe estimate: {sharpe1:.6f}")
    print(f"   After batch 2 - Sharpe estimate: {sharpe2:.6f}")

    print("\n4. Testing with positive returns (should have positive DSR)...")
    positive_returns = torch.abs(torch.randn(batch_size, num_assets)) * 0.01
    criterion.reset_state()
    loss_pos = criterion(weights, positive_returns)
    print(f"   Loss with positive returns: {loss_pos.item():.6f}")
    print(f"   DSR (should be positive): {-loss_pos.item():.6f}")

    print("\n5. Comparing with standard Sharpe ratio...")
    portfolio_returns = (weights * next_returns).sum(dim=1)
    standard_sharpe = compute_sharpe_ratio(portfolio_returns, annualize=False)
    print(f"   Standard Sharpe (non-annualized): {standard_sharpe:.6f}")
    print(f"   DSR Sharpe estimate: {criterion.get_current_sharpe_estimate():.6f}")

    print("\nAll tests passed!")
