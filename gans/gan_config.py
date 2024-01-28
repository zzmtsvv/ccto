import torch
from dataclasses import dataclass


@dataclass
class tox_config:
    batch_size: int = 64
    lr: float = 3e-4
    beta1: float = 0.9  # 0.0 for Adam-like optims
    beta2: float = 0.999  # 0.9 for Adam-like optims
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_size: float = 0.1

    # base params to create generator and discriminator
    in_channels: int = 1
    base_channels: int = 64
    latent_dim: int = 128
    num_blocks = [2, 2, 2, 2]  # [1, 1, 1, 1]
    num_epochs: int = 50
    random_seed: int = 42

    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    img_dir: str = "generated"
    dataset_path: str = "triplet_extended_dataset.pt"  # dataset.pt

    generator_size: int = batch_size  # number of generated samples for evaluation (by eyes)
    num_dis: int = 1  # number of discriminator updates per iteration
    generator_batches: int = 2
    num_fakes = batch_size * generator_batches # number of fake examples for single generator update
    generator_loss: str = "mse_loss"  # ['mse_loss', 'cauchy_loss', 'gemanmcclure_loss', 'welsch_loss', 'l1_loss', 'binary_cross_entropy_loss', 'huber_loss', 'smooth_l1', 'another_smooth_l1_loss']
    generator_loss_weights = [0.4, 0.3, 0.3]
    use_gram_matrix: bool = False  # either use gram matrix for perception loss or usual L1Loss

    vgg16_feature_loss_p_norm: int = 1  # [1, 2]

    intensity_mode: str = "mean"  # [mean, max]
    intensity_reduction: str = "mean"  # [mean, sum]
    intensity_loss_coef: float = 0.001

    num_workers: int = 0

    description: str = '''aae with triplet loss upon arcface embeddings including general loss upon
                            anchors, positives and negatives (both generator & discriminator)'''
    adversarial_term: float = 0.001
    triplet_loss_term: float = 0.001

    condition_dim: int = 2

    model_path: str = "ArcFaceLoss_model.pt"




if __name__ == "__main__":
    print(tox_config.generator_loss_weights)
