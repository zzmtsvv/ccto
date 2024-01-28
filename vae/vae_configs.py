import torch
from dataclasses import dataclass


@dataclass
class vae_config:
    random_seed: int = 42

    height_dim: int = 64 # 32
    width_dim: int = 64 # 32
    in_channels: int = 1

    test_ratio: float = 0.2
    batch_size: int = 128
    num_epochs: int = 100
    dataset_path: str = "clean_dataset.pt"  # dataset.pt

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr: float = 3e-4
    model_type: str = "vanilla"  # [vanilla, residual, vanilla-gumbel, residual-gumbel]
    likelihood_type: str = "bernoulli"  # [bernoulli, normal]
    hidden_channels: int = 256
    latent_dim: int = 10
    num_latents: int = 10
    beta: float = 1.0
    temperature_float: float = 0.6
    temperature: torch.Tensor = torch.Tensor([temperature_float]).to(device)

    checkpoint_dir: str = "vae_checkpoints"
    img_dir: str = "vae_generations"

    num_workers: int = 0

    descritption: str = '''desc'''


if __name__ == "__main__":
    print(vae_config.dataset_path[:-3])

    test = torch.rand(2, 1, 64, 64)
    for t in test:
        print(t.shape)
