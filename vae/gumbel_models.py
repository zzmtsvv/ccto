from typing import Tuple
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
from .utils import Flatten
from .blocks import Encoder, Decoder, ResidualEncoder, ResidualDecoder


class GumbelVAE(nn.Module):
    # https://arxiv.org/abs/1611.01144
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
        self.temperature = temperature

        self.proposal_network = Encoder(in_channels, hidden_channels)
        self.prenet = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, latent_dim * num_latents, kernel_size=1),
            Flatten(1)
        )

        # predict parameters for Categorical distributions
        self.proposal_logits_head = nn.Linear(latent_dim * num_latents, latent_dim * num_latents)

        self.generative_network = Decoder(in_channels, hidden_channels, latent_dim * num_latents)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, distributions.RelaxedOneHotCategorical, torch.Tensor]:
        encoder_output = self.proposal_network(x)
        bottleneck_resolution = encoder_output.size()[-2:]
        encoder_output = self.prenet(encoder_output)

        assert encoder_output.size(-1) == self.latent_dim * self.num_latents

        proposal_logits = self.proposal_logits_head(encoder_output)
        proposal_logits = proposal_logits.reshape(-1, self.num_latents, self.latent_dim).flatten(end_dim=-2)

        proposal_distribution = distributions.RelaxedOneHotCategorical(self.temperature, logits=proposal_logits)
        proposal_sample = proposal_distribution.rsample()
        proposal_sample_copy = proposal_sample
        proposal_sample = proposal_sample.reshape(-1, self.num_latents * self.latent_dim, 1, 1)
        proposal_sample = F.interpolate(proposal_sample, bottleneck_resolution)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution, proposal_sample_copy

    def generate(self,
                 num_samples: int,
                 latent_resolution: int,
                 device: str = "cpu"):
        prior_distribution = distributions.RelaxedOneHotCategorical(
            temperature=self.temperature,
            logits=torch.ones(num_samples * self.num_latents, self.latent_dim).to(device)
        )

        prior = prior_distribution.sample().reshape(-1, self.num_latents * self.latent_dim, 1, 1)

        generated = self.generative_network(prior)
        return generated

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def load(self, filename: str, map_location: str = "cpu"):
        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict)


class ResidualGumbelVAE(nn.Module):
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
        self.temperature = temperature

        self.proposal_network = nn.Sequential(
            ResidualEncoder(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=1),
        )

        self.proposal_logits_head = nn.Linear(latent_dim, latent_dim)

        self.generative_network = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, kernel_size=3, padding=1),
            ResidualDecoder(hidden_channels, in_channels),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, distributions.RelaxedOneHotCategorical, torch.Tensor]:
        encoder_output = self.proposal_network(x)

        encoded_shape = encoder_output.shape
        encoded_resolution = encoded_shape[-1] * encoded_shape[-2]

        # rearrange feature maps into "latent space"
        encoder_output = encoder_output.flatten(start_dim=-2).transpose(-1, -2).flatten(end_dim=1)

        proposal_logits = self.proposal_logits_head(encoder_output)

        proposal_distribution = distributions.RelaxedOneHotCategorical(self.temperature, logits=proposal_logits)
        proposal_sample = proposal_distribution.rsample()
        proposal_sample_copy = proposal_sample
        proposal_sample = proposal_sample.reshape(-1, encoded_resolution, self.latent_dim).transpose(-1, -2)
        proposal_sample = proposal_sample.reshape(encoded_shape)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution, proposal_sample_copy

    @torch.no_grad()
    def generate(self,
                 num_samples: int,
                 latent_resolution: int,
                 device: str = "cpu") -> torch.Tensor:
        prior_distribution = torch.distributions.RelaxedOneHotCategorical(
            temperature=self.temperature,
            logits=torch.ones(num_samples * latent_resolution * latent_resolution, self.latent_dim).to(device)
        )

        prior = prior_distribution.sample().reshape(-1, latent_resolution * latent_resolution, self.latent_dim)
        prior = prior.transpose(-1, -2).reshape(-1, self.latent_dim, latent_resolution, latent_resolution)

        generated = self.generative_network(prior)

        return generated

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load(self, filename: str, map_location: str = "cpu"):
        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    print(GumbelVAE(1, 256, 10, 10, torch.tensor([0.6])).get_model_size())
    print(ResidualGumbelVAE(1, 256, 10, 10, torch.tensor([0.6])).get_model_size())
