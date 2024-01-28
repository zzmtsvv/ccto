import torch
from torch import nn
from torch import distributions
from .utils import Clamper, ResidualBlock


class  VAECriterion(nn.Module):
    def __init__(self,
                 beta: float = 1.0,
                 likelihood: str = "normal") -> None:
        super().__init__()

        self.beta = beta

        if likelihood not in ("bernoulli", "normal"):
            raise ValueError("Unknown likelihood")
        
        self.likelihood = likelihood
    
    @staticmethod
    def kl_divergence(q_mean: torch.Tensor,
                      q_std: torch.Tensor,
                      p_mean: torch.Tensor,
                      p_std: torch.Tensor) -> torch.Tensor:
        """
        Compute KL-divergence KL(q || p) between n pairs of Gaussians
        with diagonal covariance matrices (MultivariateNormal)

        Shape of all inputs is (batch_size x dim)
        """
        
        diff_mean = p_mean - q_mean

        q_std = q_std ** 2
        p_std = p_std ** 2
        
        kl = (torch.log(p_std) - torch.log(q_std)).sum(dim=-1, keepdim=True) - p_std.shape[-1]
        kl += (torch.reciprocal(p_std) * q_std).sum(dim=-1, keepdim=True)
        kl += (diff_mean * torch.reciprocal(p_mean) * diff_mean).sum(dim=-1, keepdim=True)

        return kl / 2

    def forward(self,
                target: torch.Tensor,
                proposal: distributions.Normal,
                reconstruction: torch.Tensor) -> torch.Tensor:

        if self.likelihood == 'bernoulli':
            likelihood = distributions.Bernoulli(probs=reconstruction)
        else:
            likelihood = distributions.Normal(reconstruction, torch.ones_like(reconstruction))

        likelihood = distributions.Independent(likelihood, reinterpreted_batch_ndims=-1)
        reconstruction_score = likelihood.log_prob(target).mean()

        assert proposal.loc.dim() == 2, "proposal.shape == [*, dim], dim is shape of isotopic gaussian"

        prior = distributions.Normal(torch.zeros_like(proposal.loc), torch.ones_like(proposal.scale))
        regularization = distributions.kl_divergence(proposal,
                                                     prior).sum(dim=-1).mean()

        # evidence lower bound (maximize)
        total_score = reconstruction_score - self.beta * regularization

        return -total_score, -reconstruction_score, regularization


class GumbelCriterion(VAECriterion):
    def __init__(self,
                 beta: float = 1,
                 likelihood: str = "normal") -> None:
        super().__init__(beta, likelihood)

    def forward(self,
                target: torch.Tensor,
                proposal: distributions.RelaxedOneHotCategorical,
                proposal_sample: torch.Tensor,
                reconstruction: torch.Tensor) -> torch.Tensor:
        
        if self.likelihood == "bernoulli":
            likelihood = distributions.Bernoulli(probs=reconstruction)
        else:
            likelihood = distributions.Normal(reconstruction, torch.ones_like(reconstruction))
        
        likelihood = distributions.Independent(likelihood, reinterpreted_batch_ndims=-1)
        reconstruction_score = likelihood.log_prob(target).mean()

        assert proposal.logits.dim() == 2, "proposal.shape == [*, dim], dim is shape of isotopic gaussian"

        prior = distributions.RelaxedOneHotCategorical(
            proposal.temperature,
            logits=torch.ones_like(proposal.logits)
        )
        regularization = (proposal.log_prob(proposal_sample) - prior.log_prob(proposal_sample)).mean()

        # evidence lower bound (maximize)
        total_score = reconstruction_score - self.beta * regularization
        return -total_score, -reconstruction_score, regularization


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self,
                 out_channels: int,
                 hidden_channels: int,
                 latent_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, hidden_channels,
                               kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels,
                               kernel_size=4, stride=2, padding=1),
            Clamper(-10, 10),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=4, stride=2, padding=1),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=4, stride=2, padding=1),
            Clamper(-10, 10),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    test_encoder = torch.rand(1, 1, 64, 64)
    test_decoder = torch.rand(1, 256, 16, 16)
    # pass

    enc = Encoder(1, 256)
    res_enc = ResidualEncoder(1, 256)
    res_dec = ResidualDecoder(256, 1)
    # dec = Decoder(1, 256, )
    print(enc(test_encoder).shape, res_enc(test_encoder).shape)
    print(res_dec(test_decoder).shape)
