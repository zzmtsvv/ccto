from typing import Optional
import torch
from torch import nn


class Flatten(nn.Module):
    def __init__(self,
                 start_dim: int = 0,
                 end_dim: int = -1) -> None:
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, self.start_dim, self.end_dim)


class Clamper(nn.Module):
    def __init__(self,
                 min: Optional[float] = None,
                 max: Optional[float] = None) -> None:
        super().__init__()

        self.min = min
        self.max = max
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min, self.max)


class ResidualBlock(nn.Module):
    """
        https://arxiv.org/abs/1603.05027
    """

    def __init__(self,
                 num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // 4, num_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
