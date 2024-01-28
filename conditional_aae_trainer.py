import os
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from typing import List
from utils import tox_config, seed_everything, make_dir
from dataset.dataset2 import BinarizedDataset, random_roll
from gans.conditional_models import Encoder, Decoder, AAEDiscriminator
from gans.losses import AAEDiscriminatorLoss, AAEGeneratorLoss
from itertools import chain
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm as ptsd
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


class DiscriminatorLoss(_Loss):
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


class GeneratorLoss(_Loss):
    width_dim = 64
    height_dim = 64

    def __init__(self,
                 adversarial_term: float,
                 weights: List[float],
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pixelwise_loss = nn.L1Loss()
        self.tiling_loss = nn.MSELoss()

        self.alpha = adversarial_term

        self.general_coef, self.vertical_coef, self.horizontal_coef = weights

    def forward(self,
                encoder_scores: torch.Tensor,
                decoded_images: torch.Tensor,
                real_images: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(encoder_scores)

        adversarial_loss = self.adversarial_loss(encoder_scores, labels_true)
        reconstruction_term = self.pixelwise_loss(decoded_images, real_images)

        base_loss = self.alpha * adversarial_loss + (1 - self.alpha) * reconstruction_term
        vertical_loss = self.vertical_loss(decoded_images)
        horizontal_loss = self.horizontal_loss(decoded_images)

        return self.general_coef * base_loss + vertical_loss + horizontal_loss

    def vertical_loss(self, tile: torch.Tensor) -> torch.Tensor:
        upper_line = tile[:, :, 0, :]
        lower_line = tile[:, :, self.height_dim - 1, :]

        return self.vertical_coef * self.tiling_loss(upper_line, lower_line)

    def horizontal_loss(self, tile: torch.Tensor) -> torch.Tensor:
        left_line = tile[:, :, :, 0]
        right_line = tile[:, :, :, self.width_dim - 1]

        return self.horizontal_coef * self.tiling_loss(left_line, right_line)


class AAETrainer:
    def __init__(self,
                 cfg=tox_config()) -> None:

        self.cfg = cfg
        self.device = cfg.device
        seed_everything(cfg.random_seed)
        self.checkpoint_path = os.path.join(cfg.checkpoint_dir, f"conditional_aae.pt")

        make_dir(cfg.checkpoint_dir)
        make_dir(cfg.img_dir)

        self.encoder = Encoder(cfg.in_channels, cfg.base_channels, cfg.latent_dim, cfg.num_blocks).to(self.device)
        self.generator = Decoder(cfg.condition_dim, cfg.in_channels, cfg.base_channels, cfg.latent_dim, cfg.num_blocks).to(self.device)
        self.discriminator = AAEDiscriminator(cfg.condition_dim, cfg.latent_dim).to(self.device)

        self.gen_optim = torch.optim.AdamW(chain(self.encoder.parameters(), self.generator.parameters()),
                                           lr=cfg.lr,
                                           betas=(cfg.beta1, cfg.beta2))
        self.discr_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

        #self.gen_criterion = GeneratorLoss()
        self.gen_criterion = AAEGeneratorLoss(cfg.adversarial_term, cfg.generator_loss_weights)
        self.discr_criterion = AAEDiscriminatorLoss()

        # init dataset and dataloader
        imgs, labels = torch.load(cfg.dataset_path)
        traindata = BinarizedDataset(images=imgs,
                                        labels=labels,
                                        bins_number=cfg.num_bins,
                                        transform=[random_roll,])
        self.train_loader = DataLoader(
            traindata, shuffle=True, batch_size=cfg.batch_size, drop_last=True, pin_memory=False, num_workers=cfg.num_workers
        )
        self.one_hot = traindata.one_hot

    def fit(self):
        run_name = f"conditional_tiling_bce_" + str(self.cfg.num_bins)
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project="topology_topxpy", group=f"conditional_aae", name=run_name, job_type="training"):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith('__')})

            gen_noise = torch.randn((self.cfg.generator_size, self.generator.latent_dim)).to(self.device)
            num1 = torch.rand(1).item()
            num2 = torch.randint(low=0, high=2, size=(1,))
            oh = self.one_hot(num1)
            condition = torch.cat((oh, num2), dim=-1).to(self.device)

            for e in range(self.cfg.num_epochs):
                self.encoder.train()
                self.generator.train()
                self.discriminator.train()

                total_discriminator_loss = 0
                total_generator_loss = 0
                total_discriminator_counter = 0
                total_generator_counter = 0

                with ptsd(self.train_loader, desc=f"{e + 1}/{self.cfg.num_epochs} epochs", total=self.cfg.num_epochs) as t:
                    #self.generator.train()

                    for i, (x_true, condition, label) in enumerate(self.train_loader):
                        self.encoder.train()
                        self.generator.train()
                        self.discriminator.train()

                        x_true = x_true.to(self.device)
                        condition = condition.to(self.device)
                        size = x_true.shape[0]

                        # generator step
                        self.gen_optim.zero_grad()
                        self.discriminator.disable_grads()

                        encoded_x = self.encoder(x_true)
                        decoded_x = self.generator(encoded_x, condition)

                        encoder_scores = self.discriminator(encoded_x, condition)

                        generator_loss = self.gen_criterion(encoder_scores, decoded_x, x_true)
                        generator_loss.backward()
                        self.gen_optim.step()

                        total_generator_loss += generator_loss.item() * size
                        total_generator_counter += size

                        self.discriminator.enable_grads()

                        # discriminator step
                        self.discr_optim.zero_grad()

                        z = torch.randn(size, self.cfg.latent_dim).to(self.device)

                        target_scores = self.discriminator(z, condition)
                        encoder_scores = self.discriminator(encoded_x.detach(), condition)

                        discriminator_loss = self.discr_criterion(target_scores, encoder_scores)

                        discriminator_loss.backward()
                        self.discr_optim.step()

                        total_discriminator_loss += discriminator_loss.item() * size
                        total_discriminator_counter += size

                        t.set_postfix({
                            "discr_loss": total_discriminator_loss / total_discriminator_counter,
                            "gen_loss": total_generator_loss / total_generator_counter,
                        })

                        if not i % self.cfg.checkpoint_interval:
                            wandb.log({
                                "discriminator_loss": total_discriminator_loss / total_discriminator_counter,
                                "generator_loss": total_generator_loss / total_generator_counter,
                            })

                        if i == len(self.train_loader) - 1:
                            self.generator.eval()

                            with torch.no_grad():
                                generated_images = self.generator.sample(self.cfg.generator_size, gen_noise, condition).cpu()
                            generated_images = make_grid(
                                generated_images, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
                            plt.imsave(os.path.join(self.cfg.img_dir, f"{e + 1}.jpg"), generated_images)

                            if not (e + 1) % self.cfg.checkpoint_interval:
                                self.save()

    def save(self):
        state_dict = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "generator_optim": self.gen_optim.state_dict(),
            "discriminator_optim": self.discr_optim.state_dict(),
            "encoder": self.encoder.state_dict()
            }
        torch.save(state_dict, self.checkpoint_path)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=self.device)

        self.generator.load_state_dict(state_dict["generator"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.gen_optim.load_state_dict(state_dict["generator_optim"])
        self.discr_optim.load_state_dict(state_dict["discriminator_optim"])
        self.encoder.load_state_dict(state_dict["encoder"])
