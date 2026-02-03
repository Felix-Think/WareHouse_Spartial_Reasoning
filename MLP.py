from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
from typing import Iterable, Sequence, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Standard MLP: (Linear -> GELU -> Dropout) * (n-1) -> Linear(out_dim)

    Args:
        input_dim: input feature dimension
        hidden_dims: list/tuple of hidden layer dims, e.g. (128, 64)
        out_dim: output dimension
        dropout: dropout probability (0.0 = no dropout)
        bias: whether Linear layers use bias
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        out_dim: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        hidden_dims = tuple(int(d) for d in hidden_dims)
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty (e.g. (128, 64))")
        if any(d <= 0 for d in hidden_dims):
            raise ValueError("all hidden_dims must be > 0")
        if out_dim <= 0:
            raise ValueError("out_dim must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        dims = (input_dim, *hidden_dims, out_dim)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        # last linear (no activation by default)
        layers.append(nn.Linear(dims[-2], dims[-1], bias=bias))

        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A sensible default init for MLP + GELU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim)
        return self.net(x)

def build_mlp(
    input_dim: int = 10,
    hidden_dims: Iterable[int] = (128, 64),
    out_dim: int = 1,
    dropout: float = 0.1,
    bias: bool = True,
) -> MLP:
    """Helper to build an MLP with the given configuration."""
    return MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        out_dim=out_dim,
        dropout=dropout,
        bias=bias,
    )
if __name__ == "__main__":
    model = MLP(input_dim=10)
    sample_input = torch.randn(4, 10)  # Batch size of 4, input dimension of 10
    output = model(sample_input)
    print(output)  # Should be (4,2)
