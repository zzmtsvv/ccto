from typing import List, Tuple
import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16

try:
    from sn_modules import PseudoSobel
    from gan_config import tox_config
except ModuleNotFoundError:
    from .sn_modules import PseudoSobel
    from .gan_config import tox_config


class IntensityLoss(_Loss):
    '''
        loss that considers generations to be contrastive, unblurred and highly intensive
        in order to generate more porous-like or stable structures
        
        Uses edge detection differentiable Sobel operator for this

        possible modes:
            - mean: drive predictions to have L2 norm of the output of the Sobel operator 
                    to be close to the mean norm of the output of the Sobel operator of the clean dataset

            - max: drive predictions to have L2 norm of the output of the Sobel operator
                    to be as large as it can be (and have a spider_web-like structure)
    '''
    def __init__(self,
                 device: torch.device,
                 reduction: str = 'mean',
                 mode: str = "mean") -> None:
        super().__init__()

        assert reduction in ('mean', 'sum')
        self.reduction = reduction
        
        assert mode in ("mean", "max")
        self.mode = mode

        self.operator = PseudoSobel().to(device)
        self.mean = torch.Tensor([self.operator.mean]).to(device)
    
    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        assert len(prediction.shape) == 4, "[batch_size, channels, height, width]"

        score = self.operator(prediction)
        score = torch.norm(score, dim=(2, 3))
        
        if self.mode == "mean":
            return F.mse_loss(score, self.mean, reduction=self.reduction)
        
        if self.reduction == "mean":
            return -score.mean()
        
        return -score.sum()


class MaterialPercentLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.loss = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)
    
    def calc_percent(self, images: torch.Tensor) -> torch.Tensor:
        material = images.sum(dim=(1, 2, 3)) / (64 * 64)

        return material.unsqueeze(-1)
    
    def forward(self,
                target_percent: torch.Tensor,
                generated_images: torch.Tensor) -> torch.Tensor:
        gen_material = self.calc_percent(generated_images)

        assert gen_material.size() == target_percent.size()

        return self.loss(gen_material, target_percent)


class WassersteinDiscriminatorLoss(_Loss):
    # https://arxiv.org/abs/1701.07875
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, real_scores: torch.Tensor, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == len(real_scores.shape) == 2  # [batch_size, 1]

        loss = -torch.mean(real_scores) + torch.mean(generator_scores)
        return loss


class WassersteinGeneratorLoss(_Loss):
    # https://arxiv.org/abs/1701.07875
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == 2  # [batch_size, 1]

        loss = -torch.mean(generator_scores)
        return loss


class RelativisticDiscriminatorLoss(_Loss):
    # https://arxiv.org/abs/1807.00734v3
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_scores: torch.Tensor, generator_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(real_scores)
        labels_fake = torch.zeros_like(generator_scores)

        real_loss = self.loss(real_scores - generator_scores, labels_true)
        fake_loss = self.loss(generator_scores - real_loss, labels_fake)
        loss = (fake_loss + real_loss) / 2

        return loss


class RelativisticGeneratorLoss(_Loss):
    # https://arxiv.org/abs/1807.00734v3
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_scores: torch.Tensor, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == len(real_scores.shape) == 2  # [batch_size, 1]

        target = torch.ones_like(generator_scores)
        loss = self.loss(generator_scores - real_scores, target)
        return loss


class DiscriminatorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.MSELoss()
    
    def forward(self, real_scores: torch.Tensor, generated_scores: torch.Tensor) -> torch.Tensor:
        loss = 0

        labels_true = torch.ones_like(real_scores)
        labels_fake = torch.zeros_like(generated_scores)

        loss += self.loss(real_scores, labels_true) + self.loss(generated_scores, labels_fake)

        return loss


class MixupLoss(_Loss):
    def __init__(self, loss: nn.Module) -> None:
        super().__init__()

        self.loss = loss()
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(prediction.shape) == len(target.shape) == 2  # [batch_size, 1]

        prediction = torch.sigmoid(prediction)

        return self.loss(prediction, target)


class VGG16FeatureLoss(nn.Module):
    def __init__(self,
                 p_norm: int) -> None:
        super().__init__()

        if p_norm == 1:
            self.loss = nn.L1Loss()
        elif p_norm == 2:
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"invalid p_norm {p_norm}")
        
        features = vgg16(pretrained=True).features

        self.encoder1 = nn.Sequential(*features[:5])
        self.encoder2 = nn.Sequential(*features[5:10])
        self.encoder3 = nn.Sequential(*features[10:])

        for i in range(1, 4):
            for param in getattr(self, f"encoder{i}").parameters():
                param.requires_grad = False
    
    @staticmethod
    def make_3_channels(img: torch.Tensor) -> torch.Tensor:
        return torch.tile(img, (3, 1, 1))
    
    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loss = 0
        prediction, target = self.make_3_channels(prediction), self.make_3_channels(target)

        for i in range(1, 4):
            func = getattr(self, f"encoder{i}")
            
            prediction, target = func(prediction), func(target)
            loss += self.loss(prediction, target)
        
        return loss


class PerceptualLoss(_Loss):
    def __init__(self,
                 use_gram_matrix: bool = True) -> None:
        super().__init__()

        self.loss = self.gram_matrix_loss if use_gram_matrix else nn.L1Loss()
    
    def forward(self, features_real: List[torch.Tensor], features_generated: List[torch.Tensor]) -> torch.Tensor:
        loss = 0

        for real, generated in zip(features_real, features_generated):
            loss += self.loss(real, generated)
        
        return loss

    @staticmethod
    def compute_gram_matrix(y: torch.Tensor) -> torch.Tensor:
        b, с, h, w = y.shape
        return torch.einsum('bchw,bdhw->bcd', [y, y]) / (h * w)
    
    def gram_matrix_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4
        batch_size, c, h, w = x.shape

        G = self.compute_gram_matrix(x)
        A = self.compute_gram_matrix(y)
        #return A.shape

        return F.mse_loss(G, A)


class GeneratorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.base_loss = nn.MSELoss()

    def forward(self, generator_score: torch.Tensor) -> torch.Tensor:
        assert len(generator_score.shape) == 2  # [batch_size, 1]

        target = torch.ones_like(generator_score)
        loss = self.base_loss(generator_score, target)
        return loss


class GeneratorReconstructionLoss(_Loss):
    height_dim = 64
    width_dim = 64

    def __init__(self, loss: str, weights: List[float] = tox_config.generator_loss_weights, eps: float = 1e-8) -> None:
        '''
            weights - a List of values sum up to one that identify proportion of tiling losses and general loss to the overall loss
            1st weight corresponds to the general loss (on the level of labels from discriminator)
            2nd weight - vertical tiling loss
            3rd weight - horizontal tiling loss
        '''
        super().__init__()

        assert 1 - eps <= sum(weights) <= 1 + eps and len(weights) == 3, "Incorrect weights for GeneratorReconstructionLoss"
        self.general_coef, self.vertical_coef, self.horizontal_coef = weights

        self.base_loss = str2loss[loss]()
        self.tiling_loss = nn.MSELoss()
    
    def vertical_loss(self, tile: torch.Tensor) -> torch.Tensor:
        upper_line = tile[:, :, 0, :]
        lower_line = tile[:, :, self.height_dim - 1, :]

        return self.vertical_coef * self.tiling_loss(upper_line, lower_line)

    def horizontal_loss(self, tile: torch.Tensor) -> torch.Tensor:
        left_line = tile[:, :, :, 0]
        right_line = tile[:, :, :, self.width_dim - 1]

        return self.horizontal_coef * self.tiling_loss(left_line, right_line)
    
    def forward(self, generated_image: torch.Tensor, generator_score: torch.Tensor) -> torch.Tensor:
        # return x.neg().mean()
        assert len(generated_image.shape) == 4  # [batch_size, C, H, W]
        assert len(generator_score.shape) == 2  # [batch_size, 1]
        
        # x = gen_img.detach()
        # vertical_tiling = torch.cat((gen_img, x), dim=2)
        # horizontal_tiling = torch.cat((gen_img, x), dim=3)

        target = torch.ones_like(generator_score)

        tiling_loss = self.vertical_loss(generated_image) + self.horizontal_loss(generated_image)
        loss = self.general_coef * self.base_loss(generator_score, target) + tiling_loss
        return loss


class SobolevDiscriminatorLoss(_Loss):
    '''
        alpha - Lagrange multiplier
        rho - quadratic weight penalty
    '''
    def __init__(self,
                 alpha: float = 0.0,
                 rho: float = 1e-5,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.alpha = alpha
        self.rho = rho
    
    def forward(self,
                real_image: torch.Tensor,
                generator_image: torch.Tensor,
                real_scores: torch.Tensor,
                generator_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(generator_scores.shape) == len(real_scores.shape) == 2  # [batch_size, 1]
        assert real_image.requires_grad == generator_image.requires_grad == True

        ipm_estimate = real_scores.mean() - generator_scores.mean()

        grad_real = grad(real_scores.sum(), real_image, create_graph=True)[0]
        grad_fake = grad(generator_scores.sum(), generator_image, create_graph=True)[0]
        
        grad_real = grad_real.view(grad_real.size(0), -1).pow(2).mean()
        grad_fake = grad_fake.view(grad_fake.size(0), -1).pow(2).mean()

        omega = (grad_real + grad_fake) / 2

        loss = -ipm_estimate - self.alpha * (1.0 - omega) + self.rho * (1.0 - omega).pow(2) / 2
        return ipm_estimate, loss


class SobolevGeneratorLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == 2  # [batch_size, 1]
        assert generator_scores.requires_grad

        loss = -generator_scores.mean()
        return loss


class AlphaReconstructionLoss(_Loss):
    '''
        the input is considered to be real images and reconstructed ones by generator
        that takes hidden representation of these images from the encoder.
    '''
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.L1Loss()
    
    def forward(self,
                reconstructed: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        return self.loss(reconstructed, target)


class AlphaEncoderLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, encoder_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(encoder_scores)
        labels_fake = torch.zeros_like(encoder_scores)

        loss = self.loss(encoder_scores, labels_true) - self.loss(encoder_scores, labels_fake)
        return loss


class AlphaGeneratorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                reconstructed_scores: torch.Tensor,
                generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == len(reconstructed_scores.shape) == 2  # [batch_size, 1]

        labels_true = torch.ones_like(reconstructed_scores)
        labels_fake = torch.zeros_like(generator_scores)

        recon_loss = self.loss(reconstructed_scores, labels_true) - self.loss(reconstructed_scores, labels_fake)
        gener_loss = self.loss(generator_scores, labels_true) - self.loss(generator_scores, labels_fake)
        return recon_loss + gener_loss


class AlphaDiscriminatorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                real_scores: torch.Tensor,
                reconstructed_scores: torch.Tensor,
                generated_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(real_scores)
        labels_fake = torch.zeros_like(generated_scores)

        loss = self.loss(real_scores, labels_true) + self.loss(generated_scores, labels_fake)
        loss += self.loss(reconstructed_scores, labels_fake)

        return loss


class CodecriminatorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                encoder_scores: torch.Tensor,
                target_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(encoder_scores)
        labels_fake = torch.zeros_like(target_scores)

        loss = self.loss(encoder_scores, labels_fake) + self.loss(target_scores, labels_true)
        return loss


class AAEDiscriminatorLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                target_scores: torch.Tensor,
                encoder_scores: torch.Tensor) -> torch.Tensor:
        
        labels_true = torch.ones_like(target_scores)
        labels_fake = torch.zeros_like(encoder_scores)

        loss = self.loss(target_scores, labels_true) + self.loss(encoder_scores, labels_fake)
        return loss / 2


class AAEGeneratorLoss(_Loss):
    def __init__(self,
                 adversarial_term: float,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pixelwise_loss = nn.L1Loss()
        self.alpha = adversarial_term
    
    def forward(self,
                encoder_scores: torch.Tensor,
                decoded_images: torch.Tensor,
                real_images: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(encoder_scores)

        adversarial_loss = self.adversarial_loss(encoder_scores, labels_true)
        reconstruction_term = self.pixelwise_loss(decoded_images, real_images)

        loss = self.alpha * adversarial_loss + (1 - self.alpha) * reconstruction_term
        return loss


class BaseRobustLoss(_Loss):
    def __init__(self, c=1.0, reduction='mean') -> None:
        super().__init__()

        self.c2 = c * c
        self.reduction_ = reduction
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
            arguments are tensors of labels
        '''
        x = prediction - target
        x = x.pow(2) / self.c2

        loss = self.robust_loss_fn(x)

        if self.reduction_ == 'mean':
            loss = loss.mean(dim=0)
        if self.reduction_ == 'sum':
            loss = loss.sum(dim=0)
        
        return loss


class CauchyLoss(BaseRobustLoss):
    def __init__(self, c=1.0, reduction='mean') -> None:
        super().__init__(c=c, reduction=reduction)
    
    def robust_loss_fn(self, x: torch.tensor) -> torch.Tensor:
        return torch.log1p(x)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


class GemanMcClureLoss(BaseRobustLoss):
    def __init__(self, c=1.0, reduction='mean') -> None:
        super().__init__(c, reduction)
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x / (x + 4) 

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


class WelschLoss(BaseRobustLoss):
    def __init__(self, c=1.0, reduction='mean') -> None:
        super().__init__(c, reduction)
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-x / 2)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


class AnotherSmoothL1Loss(BaseRobustLoss):
    def __init__(self, c=1.0, reduction='mean') -> None:
        super().__init__(c, reduction)
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x + 1) - 1
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


str2loss = {
    "mse_loss": nn.MSELoss,
    "cauchy_loss": CauchyLoss,
    "gemanmcclure_loss": GemanMcClureLoss,
    "welsch_loss": WelschLoss,
    "l1_loss": nn.L1Loss,
    "binary_cross_entropy_loss": nn.BCEWithLogitsLoss,
    "huber_loss": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "another_smooth_l1_loss": AnotherSmoothL1Loss
}


if __name__ == '__main__':
    # loss = GeneratorReconstructionLoss()
    # pred = torch.randn(16, 1, 64, 64).requires_grad_(True)
    # target = torch.rand(16, 1, 64, 64)
    # print(loss.gram_matrix_loss(pred, target))

    # from torchvision.utils import make_grid
    # from matplotlib import pyplot as plt
    # imgs, labels = torch.load('dataset.pt')
    # x = imgs[6500].unsqueeze(0).unsqueeze(0)
    # print(x[:, :, :, 0], x[:, :, :, 63])
    # # one_flip = torch.cat((x, x.detach()), dim=2)  # вертикаль
    # # two_flip = torch.cat((x, x), dim=3)  # горизонталь
    # # print(one_flip.size(), two_flip.size())
    # plt.imshow(x.squeeze(0).squeeze(0).numpy())
    # plt.show()

    # plt.imshow(one_flip.numpy())
    # plt.show()

    # plt.imshow(two_flip.numpy())
    # plt.show()

    # three_flip = x.flip(2, 3)
    # imgs = torch.cat([x, one_flip, two_flip, three_flip], dim=0)
    # print(imgs.shape)
    # images = make_grid(
    #     imgs, nrow=2, normalize=True).numpy().transpose(1, 2, 0)
    # plt.imshow(images)
    # plt.show()

    # weights = [2/3, 1/3, 0]
    # g = GeneratorReconstructionLoss(weights)

    # inp = torch.randn(16, 1).requires_grad_(True)
    # tgt = torch.randn(16, 1)
    # loss = CauchyLoss()
    # print(loss(inp, tgt))

    # import matplotlib.pyplot as plt
    # import numpy as np

    # a = np.array([[1,0,0,1],[1,0,1,1],[1,0,1,1],[1,0,1,0]])
    # plt.imshow(a)
    # ah = np.hstack([a,a])
    # print(ah.shape)
    # plt.imshow(ah)
    # av = np.vstack([a,a])
    # plt.imshow(av)
    # print(av.shape)


    # from sklearn.metrics import mean_absolute_error, mean_squared_error

    # print(av)
    # print(av[av.shape[0]//2], av[av.shape[0]//2-1])
    # loss_v = mean_squared_error(av[av.shape[0]//2], av[av.shape[0]//2-1])
    # print("loss_v:", loss_v)

    # loss_h = mean_squared_error(ah[:, ah.shape[1]//2], ah[:, ah.shape[1]//2-1])
    # print("loss_h:", loss_h)
    
    # loss = PerceptualLoss()
    # # n = 64
    # # test1 = torch.randn(3, 1, n, n).requires_grad_(True)
    # # test2 = torch.randn(3, 1, n, n)

    # # print(loss.gram_matrix_loss(test1, test2))

    # shapes = [
    #     torch.Size([3, 64, 32, 32]),
    #     torch.Size([3, 128, 16, 16]),
    #     torch.Size([3, 256, 8, 8]),
    #     torch.Size([3, 512, 4, 4]),
    #     torch.Size([3, 1024, 2, 2])
    # ]

    # for shape in shapes:
    #     test1 = torch.randn(shape).requires_grad_(True)
    #     test2 = torch.randn(shape)

    #     print(loss.gram_matrix_loss(test1, test2))

    test = torch.rand(4, 1, 64, 64)
    # m = vgg16().features[:5]
    # print(m(test).shape)
    print(torch.tile(test, (3, 1, 1)).shape)
