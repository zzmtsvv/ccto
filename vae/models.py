from typing import Tuple
import torch
from torch import distributions
from torch import nn
from torch.nn import functional as F

from .blocks import Encoder, Decoder, ResidualEncoder, ResidualDecoder
from .utils import Flatten


class VanillaVAE(nn.Module):
    # https://arxiv.org/abs/1906.02691
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 latent_dim: int,
                 num_latents: int,
                 temperature: torch.Tensor) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        self.proposal_network = Encoder(in_channels, hidden_channels)
        self.prenet = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, latent_dim * num_latents, kernel_size=1),
            Flatten(1)
        )

        self.proposal_mu_head = nn.Linear(latent_dim * num_latents, latent_dim * num_latents)
        self.proposal_sigma_head = nn.Sequential(
            nn.Linear(latent_dim * num_latents, latent_dim * num_latents),
            nn.Softplus()
        )

        self.generative_network = Decoder(in_channels, hidden_channels, latent_dim * num_latents)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, distributions.Normal]:
        encoder_output = self.proposal_network(x)

        bottleneck_resolution = encoder_output.size()[-2:]
        encoder_output = self.prenet(encoder_output)

        assert encoder_output.size(-1) == self.latent_dim * self.num_latents

        proposal_mu = self.proposal_mu_head(encoder_output).reshape(-1, self.num_latents, self.latent_dim)
        proposal_mu = proposal_mu.flatten(end_dim=-2)

        proposal_sigma = self.proposal_sigma_head(encoder_output).reshape(-1, self.num_latents, self.latent_dim)
        proposal_sigma = proposal_sigma.flatten(end_dim=-2)


        proposal_distribution = distributions.Normal(proposal_mu, proposal_sigma)
        proposal_sample = proposal_distribution.rsample().reshape(-1, self.num_latents * self.latent_dim, 1, 1)
        proposal_sample = F.interpolate(proposal_sample, bottleneck_resolution)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution
    
    def generate(self,
                 num_samples: int,
                 latent_resolution: int,
                 device: str = "cpu") -> torch.Tensor:
        prior = torch.randn(num_samples * self.num_latents, self.latent_dim).to(device)
        prior = prior.reshape(-1, self.num_latents * self.latent_dim, 1, 1)

        generated = self.generative_network(prior)
        return generated

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 latent_dim: int,
                 num_latents: int,
                 temperature: torch.Tensor) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        self.proposal_network = nn.Sequential(
            ResidualEncoder(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=1)
        )

        self.proposal_mu_head = nn.Linear(latent_dim, latent_dim)
        self.proposal_sigma_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus()
        )

        self.generative_network = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, kernel_size=3, padding=1),
            ResidualDecoder(hidden_channels, in_channels)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, distributions.Normal]:
        encoder_output = self.proposal_network(x)

        encoded_shape = encoder_output.shape
        encoded_resolution = encoded_shape[-1] * encoded_shape[-2]

        # rearrange feature maps into "latent space"
        encoder_output = encoder_output.flatten(start_dim=-2).transpose(-1, -2).flatten(end_dim=1)

        proposal_mu = self.proposal_mu_head(encoder_output)
        proposal_sigma = self.proposal_sigma_head(encoder_output)

        proposal_distribution = distributions.Normal(proposal_mu, proposal_sigma)
        proposal_sample = proposal_distribution.rsample().reshape(-1, encoded_resolution, self.latent_dim)
        proposal_sample = proposal_sample.transpose(-1, -2).reshape(encoded_shape)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution
    
    @torch.no_grad()
    def generate(self,
                 num_samples: int,
                 latent_resolution: int,
                 device: str = "cpu") -> torch.Tensor:
        
        prior = torch.randn(num_samples * latent_resolution * latent_resolution, self.latent_dim)
        prior = prior.reshape(-1, latent_resolution * latent_resolution, self.latent_dim).transpose(-1, -2)
        prior = prior.reshape(-1, self.latent_dim, latent_resolution, latent_resolution).to(device)

        generated = self.generative_network(prior)
        return generated

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(VanillaVAE(1, 256, 10, 10, torch.tensor([0.6])).get_model_size())
    print(ResidualVAE(1, 256, 10, 10, torch.tensor([0.6])).get_model_size())
