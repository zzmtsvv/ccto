from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class Flatten(nn.Module):
    def __init__(self,
                 start_dim: int = 0,
                 end_dim: int = -1) -> None:
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, self.start_dim, self.end_dim)


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


class Classifier(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 256) -> None:
        super().__init__()

        self.projection_net = nn.Sequential(
            self.make_block(in_channels, hidden_channels // 16),
            self.make_block(hidden_channels // 16, hidden_channels // 8),
            self.make_block(hidden_channels // 8, hidden_channels // 4),
            self.make_block(hidden_channels // 4, hidden_channels // 2),
            self.make_block(hidden_channels // 2, hidden_channels),
            Flatten(1),
            nn.ReLU(),
            nn.Linear(1024, 3)
        )
        # self.head = nn.Linear(3, 1)

    @staticmethod
    def make_block(in_channels: int,
                   out_channels: int,
                   kernel_size: int = 4,
                   stride: int = 2,
                   padding: int = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ResidualBlock(out_channels)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.projection_net(x)
        # scores = self.head(embeddings)

        return None, embeddings

    def get_model_size(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class TripletEmbeddingLoss(_Loss):
    '''
        Triplet loss realized over embeddings of some representative network
    '''
    def __init__(self,
                 feature_extractor: nn.Module,
                 margin: float = 1.0,
                 swap: bool = True,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.feature_extractor = feature_extractor
        self.disable_feature_extractor_grads()

        self.margin = margin
        self.swap = swap
    
    def disable_feature_extractor_grads(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
    
    def calc_euclidean(self,
                       x1: torch.Tensor,
                       x2: torch.Tensor) -> torch.Tensor:
        _, x1 = self.feature_extractor(x1)
        _, x2 = self.feature_extractor(x2)

        return (x1 - x2).pow(2).sum(1)
    
    def casual_loss(self,
                    anchor: torch.Tensor,
                    positive: torch.Tensor,
                    negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)

        # this ensures that the hardest negative inside the triplet is used for backpropagation
        if self.swap:
            distance_negative_a = self.calc_euclidean(anchor, negative)
            distance_negative_p = self.calc_euclidean(positive, negative)
            
            distance_negative = torch.minimum(distance_negative_a, distance_negative_p)
        else:
            distance_negative = self.calc_euclidean(anchor, negative)


        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses
    
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        
        loss = self.casual_loss(anchor, positive, negative)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()


class TripletImageLoss(_Loss):
    '''
        Triplet loss upon generations grom the generator.
    '''
    def __init__(self,
                 feature_extractor=None,
                 margin: float = 1.0,
                 swap: bool = True,
                 use_gram_matrix: bool = False,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.margin = margin
        self.use_gram_matrix = use_gram_matrix
        self.swap = swap
        
        self.loss_fn = self.calc_euclidean
        if use_gram_matrix:
            self.loss_fn = self.gram_matrix_loss
    
    @staticmethod
    def calc_euclidean(x1: torch.Tensor,
                       x2: torch.Tensor,
                       dims: Tuple[int] = (1, 2, 3)) -> torch.Tensor:
        # dims = list(range(1, len(x1.shape)))
        return (x1 - x2).pow(2).sum(dim=dims).unsqueeze(1)
    
    def casual_loss(self,
                    anchor: torch.Tensor,
                    positive: torch.Tensor,
                    negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.loss_fn(anchor, positive)

        # this ensures that the hardest negative inside the triplet is used for backpropagation
        if self.swap:
            distance_negative_a = self.loss_fn(anchor, negative)
            distance_negative_p = self.loss_fn(positive, negative)
            
            distance_negative = torch.minimum(distance_negative_a, distance_negative_p)
        else:
            distance_negative = self.loss_fn(anchor, negative)


        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses

    @staticmethod
    def compute_gram_matrix(y: torch.Tensor) -> torch.Tensor:
        b, с, h, w = y.shape
        return torch.einsum('bchw,bdhw->bcd', [y, y]) / (h * w)
    
    def gram_matrix_loss(self,
                         x: torch.Tensor,
                         y: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4
        # batch_size, c, h, w = x.shape

        G = self.compute_gram_matrix(x)
        A = self.compute_gram_matrix(y)
        #return A.shape

        return (G - A).pow(2).sum(1)


if __name__ == "__main__":
    a = torch.randn(16, 1, 64, 64).requires_grad_(True)
    b = torch.randn(16, 1, 64, 64).requires_grad_(True)
    loss = TripletImageLoss()

    l = loss.calc_euclidean(a, b)
    print(l.shape)
