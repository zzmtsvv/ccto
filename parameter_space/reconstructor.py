# source: https://arxiv.org/abs/2011.13786
from typing import Tuple
import torch
from torch import nn
from seresnet18 import make_seresnet18


def save_hook(module, input, output):
    setattr(module, 'output', output)


class Reconstructor(nn.Module):
    def __init__(self,
                 num_directions: int) -> None:
        super().__init__()

        self.feature_extractor = make_seresnet18()

        self.feature_extractor.layer0.conv1 = nn.Conv2d(in_channels=2,
                                                 out_channels=64,
                                                 kernel_size=7,
                                                 stride=2,
                                                 padding=3,
                                                 bias=False)
        nn.init.kaiming_normal_(self.feature_extractor.layer0.conv1.weight,
                                mode="fan_out",
                                nonlinearity="relu")
        
        self.features = self.feature_extractor.avg_pool
        self.features.register_forward_hook(save_hook)

        self.direction_estimator = nn.Linear(self.feature_extractor.num_features, num_directions)
        self.shift_estimator = nn.Linear(self.feature_extractor.num_features, 1)
    
    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x1.shape[0]

        self.feature_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.direction_estimator(features)
        shift_magnitude = self.shift_estimator(features)

        return logits, shift_magnitude.squeeze()


if __name__ == "__main__":
    x1 = torch.rand(16, 1, 64, 64)
    x2 = torch.rand(16, 1, 64, 64)

    model = Reconstructor(5)

    logits, shifts = model(x1, x2)

    print(logits.shape, shifts.shape)
