from typing import Optional
import torch
from torch import nn
from typing import List, Tuple

try:
    from sn_modules import SNConv2d, SNLinear
except ModuleNotFoundError:
    from .sn_modules import SNConv2d, SNLinear


class Reshape(nn.Module):
    def __init__(self, shape: Tuple[int, int, int, int]) -> None:
        super().__init__()

        self.shape = shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)


class GeneratorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out += self.skip_connection(x)
        return out


class SNDiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool = False) -> None:
        super().__init__()

        self.first_activation = nn.ReLU()

        self.block = nn.Sequential(
            SNConv2d(in_channels, out_channels, 3, 1 + downsample, 1),
            nn.ReLU(),
            SNConv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels == out_channels and not downsample:
            self.skip_connection = nn.Identity()
        else:
            skip = [nn.AvgPool2d(2)] if downsample else []
            skip.append(SNConv2d(in_channels, out_channels, 1, 1, 0))
            self.skip_connection = nn.Sequential(*skip)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(self.first_activation(x))
        out += self.skip_connection(x)
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool = False) -> None:
        super().__init__()

        self.first_activation = nn.ReLU()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1 + downsample, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels == out_channels and not downsample:
            self.skip_connection = nn.Identity()
        else:
            skip = [nn.AvgPool2d(2)] if downsample else []
            skip.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
            self.skip_connection = nn.Sequential(*skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(self.first_activation(x))
        out += self.skip_connection(x)
        return out


class Generator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 latent_dim: int = 128,
                 num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256 * base_channels, bias=False),
            Reshape((-1, 16 * base_channels, 4, 4)),
            self.make_layer(16 * base_channels, 8 * base_channels, num_blocks[0]),
            self.make_layer(8 * base_channels, 4 * base_channels, num_blocks[1]),
            self.make_layer(4 * base_channels, 2 * base_channels, num_blocks[2]),
            self.make_layer(2 * base_channels, base_channels, num_blocks[3]),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, 1, 1, 0),
            nn.Tanh()
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode='fan_in', nonlinearity="leaky_relu"
                )
    
    @staticmethod
    def make_layer(in_channels: int,
                   out_channels: int,
                   num_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            GeneratorBlock(in_channels, out_channels),
            *[GeneratorBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out
    
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device

        if noise is None:
            noise = torch.randn((num_samples, self.latent_dim))
        
        return self.forward(noise.to(device))

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SNDiscriminator(nn.Module):
    # https://arxiv.org/abs/1802.05957v1
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__()

        first_dis_block = SNDiscriminatorBlock(in_channels, base_channels, downsample=True)
        first_dis_block.first_activation = nn.Identity()

        self.net = nn.Sequential(
            first_dis_block,
            self.make_layer(base_channels, 2 * base_channels, num_blocks[0]),
            self.make_layer(2 * base_channels, 4 * base_channels, num_blocks[1]),
            self.make_layer(4 * base_channels, 8 * base_channels, num_blocks[2]),
            self.make_layer(8 * base_channels, 16 * base_channels, num_blocks[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            SNLinear(16 * base_channels, 1)
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (SNLinear, SNConv2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode='fan_in', nonlinearity="leaky_relu"
                )
                if m.use_gamma:
                    m._init_gamma()

    @staticmethod
    def make_layer(in_channels: int,
                   out_channels: int,
                   num_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            SNDiscriminatorBlock(in_channels, out_channels, downsample=True),
            *[SNDiscriminatorBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return score
    
    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad = False

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PerceptualDiscriminator(SNDiscriminator):
    def __init__(self, in_channels: int, base_channels: int = 64, num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__(in_channels, base_channels, num_blocks)
    
    @staticmethod
    def is_for_perception_loss(layer: nn.Module) -> bool:
        return isinstance(layer, nn.Sequential) or isinstance(layer, SNDiscriminatorBlock)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []

        for layer in self.net:
            x = layer(x)
            if self.is_for_perception_loss(layer):
                #print(layer.__class__.__name__)
                features.append(x)
        
        return x, features


class MixupDiscriminator(PerceptualDiscriminator):
    def __init__(self, in_channels: int, base_channels: int = 64, num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__(in_channels, base_channels, num_blocks)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []

        for layer in self.net:
            x = layer(x)
            if self.is_for_perception_loss(layer):
                features.append(x)
        
        return torch.sigmoid(x), features


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__()

        first_dis_block = DiscriminatorBlock(in_channels, base_channels, downsample=True)
        first_dis_block.first_activation = nn.Identity()

        self.net = nn.Sequential(
            first_dis_block,
            self.make_layer(base_channels, 2 * base_channels, num_blocks[0]),
            self.make_layer(2 * base_channels, 4 * base_channels, num_blocks[1]),
            self.make_layer(4 * base_channels, 8 * base_channels, num_blocks[2]),
            self.make_layer(8 * base_channels, 16 * base_channels, num_blocks[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * base_channels, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode='fan_in', nonlinearity="leaky_relu"
                )

    @staticmethod
    def make_layer(in_channels: int,
                   out_channels: int,
                   num_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            DiscriminatorBlock(in_channels, out_channels, downsample=True),
            *[DiscriminatorBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return score

    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad = True

    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad = False

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WassersteinDiscriminator(Discriminator):
    # https://arxiv.org/abs/1701.07875
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_blocks: List[int] = [1, 1, 1, 1],
                 clip_grad_value: float = 0) -> None:
        super().__init__(in_channels, base_channels, num_blocks)

        self.clip_value = clip_grad_value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
    
    def clip_weights(self) -> None:
        for param in self.parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)
    

class Encoder(nn.Module):
    '''
        Architecture is the same as in Discriminator except the head projects to the latent space with latent_dim
    '''
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 latent_dim: int = 128,
                 num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        first_dis_block = DiscriminatorBlock(in_channels, base_channels, downsample=True)
        first_dis_block.first_activation = nn.Identity()

        self.net = nn.Sequential(
            first_dis_block,
            self.make_layer(base_channels, 2 * base_channels, num_blocks[0]),
            self.make_layer(2 * base_channels, 4 * base_channels, num_blocks[1]),
            self.make_layer(4 * base_channels, 8 * base_channels, num_blocks[2]),
            self.make_layer(8 * base_channels, 16 * base_channels, num_blocks[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * base_channels, latent_dim)
        )

        self.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode='fan_in', nonlinearity="leaky_relu"
                )
    
    @staticmethod
    def make_layer(in_channels: int,
                   out_channels: int,
                   num_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            DiscriminatorBlock(in_channels, out_channels, downsample=True),
            *[DiscriminatorBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )
    
    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Codecriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Decoder(Generator):
    pass


class AAEDiscriminator(Codecriminator):
    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad = False


if __name__ == "__main__":
    generator = Generator(1, num_blocks=[2, 2, 2, 2])

    first_gen_block = generator.net[2][1].block
    first_conv = first_gen_block[2]

    print(first_conv.weight.data.shape)
