"""
Portfolio LSTM Model Module

Implements a 2-layer LSTM network that outputs portfolio weights.

Architecture:
    Input: (batch, T=60, F=30) - 60 timesteps, 30 features (6 per asset * 5 assets)
    LSTM Layer 1: hidden_size=128, dropout=0.3
    LSTM Layer 2: hidden_size=64, dropout=0.3
    Linear: 64 -> num_assets
    Softmax: Ensure weights sum to 1, all positive (long-only)
    Output: (batch, N=5) - portfolio weights
"""

import torch
import torch.nn as nn


class PortfolioLSTM(nn.Module):
    """
    2-layer LSTM network for portfolio weight prediction.

    The model takes a sequence of feature vectors and outputs portfolio weights
    that are guaranteed to be:
    - Positive (long-only constraint)
    - Sum to 1 (fully invested)

    Attributes:
        input_size: Number of input features (F)
        hidden_size_1: Hidden size of first LSTM layer
        hidden_size_2: Hidden size of second LSTM layer
        num_assets: Number of assets to allocate weights to
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64,
        num_assets: int = 5,
        dropout: float = 0.3
    ):
        """
        Args:
            input_size: Number of input features (F = 6 features * num_assets)
            hidden_size_1: Hidden dimension of first LSTM layer
            hidden_size_2: Hidden dimension of second LSTM layer
            num_assets: Number of assets in portfolio (output dimension)
            dropout: Dropout probability between layers
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_assets = num_assets
        self.dropout_prob = dropout

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Output layer: maps hidden state to asset weights
        self.fc = nn.Linear(hidden_size_2, num_assets)

        # Softmax for valid portfolio weights
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and linear layer weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal (helps with gradient flow)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: zeros, except forget gate bias = 1 (helps learning)
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for LSTM
                if 'bias_ih' in name or 'bias_hh' in name:
                    # LSTM has 4 gates, forget gate is the second one
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)
            elif 'fc.weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'fc.bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: (batch, T, F) input tensor
               - batch: batch size
               - T: sequence length (lookback window)
               - F: number of features

        Returns:
            weights: (batch, N) portfolio weights
                    - All positive (long-only)
                    - Sum to 1 per sample
        """
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)  # (batch, T, hidden_size_1)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, T, hidden_size_2)

        # Take output from last timestep only
        last_hidden = lstm2_out[:, -1, :]  # (batch, hidden_size_2)
        last_hidden = self.dropout2(last_hidden)

        # Linear projection to asset space
        logits = self.fc(last_hidden)  # (batch, num_assets)

        # Softmax ensures valid portfolio weights
        weights = self.softmax(logits)  # (batch, num_assets)

        return weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference mode prediction (no dropout, no gradients).

        Args:
            x: (batch, T, F) or (T, F) input tensor

        Returns:
            weights: (batch, N) or (N,) portfolio weights
        """
        self.eval()

        # Handle single sample input
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        with torch.no_grad():
            weights = self.forward(x)

        if squeeze_output:
            weights = weights.squeeze(0)

        return weights

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a summary of the model architecture."""
        lines = [
            "PortfolioLSTM Summary",
            "=" * 50,
            f"Input size: {self.input_size}",
            f"LSTM Layer 1: {self.input_size} -> {self.hidden_size_1}",
            f"LSTM Layer 2: {self.hidden_size_1} -> {self.hidden_size_2}",
            f"Output Layer: {self.hidden_size_2} -> {self.num_assets}",
            f"Dropout: {self.dropout_prob}",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 50,
        ]
        return "\n".join(lines)


def create_model_from_config(config: dict) -> PortfolioLSTM:
    """
    Create a PortfolioLSTM model from configuration dictionary.

    Args:
        config: Configuration dict with 'model' and 'assets' sections

    Returns:
        Initialized PortfolioLSTM model
    """
    num_assets = len(config['assets']['tickers'])
    num_features = 6 * num_assets  # 6 features per asset

    model = PortfolioLSTM(
        input_size=num_features,
        hidden_size_1=config['model']['lstm_hidden_1'],
        hidden_size_2=config['model']['lstm_hidden_2'],
        num_assets=num_assets,
        dropout=config['model']['dropout']
    )

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing PortfolioLSTM...")

    # Create model with default params
    model = PortfolioLSTM(
        input_size=30,  # 6 features * 5 assets
        hidden_size_1=128,
        hidden_size_2=64,
        num_assets=5,
        dropout=0.3
    )

    print(model.summary())

    # Test forward pass
    print("\n1. Testing forward pass...")
    batch_size = 32
    seq_len = 60
    num_features = 30

    x = torch.randn(batch_size, seq_len, num_features)
    weights = model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {weights.shape}")
    print(f"   Sample weights: {weights[0].detach().numpy()}")
    print(f"   Weights sum: {weights[0].sum().item():.6f}")
    print(f"   All positive: {(weights >= 0).all().item()}")

    # Test predict mode
    print("\n2. Testing predict mode...")
    model.eval()
    single_x = torch.randn(seq_len, num_features)
    single_weights = model.predict(single_x)

    print(f"   Single input shape: {single_x.shape}")
    print(f"   Single output shape: {single_weights.shape}")
    print(f"   Weights: {single_weights.numpy()}")

    # Test gradient flow
    print("\n3. Testing gradient flow...")
    model.train()
    x = torch.randn(batch_size, seq_len, num_features, requires_grad=True)
    weights = model(x)
    loss = weights.sum()
    loss.backward()

    print(f"   Input gradient exists: {x.grad is not None}")
    print(f"   Input gradient shape: {x.grad.shape}")

    # Test with different batch sizes
    print("\n4. Testing various batch sizes...")
    for bs in [1, 16, 64]:
        x = torch.randn(bs, seq_len, num_features)
        w = model(x)
        assert w.shape == (bs, 5), f"Wrong output shape for batch_size={bs}"
        assert torch.allclose(w.sum(dim=1), torch.ones(bs)), f"Weights don't sum to 1"
    print("   All batch sizes work correctly!")

    print("\nAll tests passed!")
