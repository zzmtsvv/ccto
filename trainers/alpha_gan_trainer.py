import os
import torch
from utils import tox_config, seed_everything, make_dir
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from dataset.dataset2 import StructuresDatasetTensor, random_roll
from gans.models import Encoder, Generator, Discriminator, Codecriminator
from gans.losses import AlphaReconstructionLoss, AlphaGeneratorLoss, AlphaDiscriminatorLoss, AlphaEncoderLoss, CodecriminatorLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


class AlphaGANTrainer:
    def __init__(self,
                 cfg=tox_config()) -> None:
        self.cfg = cfg
        self.device = cfg.device
        seed_everything(cfg.random_seed)
        self.checkpoint_path = os.path.join(cfg.checkpoint_dir, f"alpha_gan.pt")

        make_dir(cfg.checkpoint_dir)
        make_dir(cfg.img_dir)
        make_dir(cfg.recon_dir)

        self.encoder = Encoder(cfg.in_channels,
                               cfg.base_channels,
                               cfg.latent_dim,
                               cfg.num_blocks).to(self.device)
        self.generator = Generator(cfg.in_channels,
                                   cfg.base_channels,
                                   cfg.latent_dim,
                                   cfg.num_blocks).to(self.device)
        self.discriminator = Discriminator(cfg.in_channels,
                                           cfg.base_channels,
                                           cfg.num_blocks).to(self.device)
        self.codecriminator = Codecriminator(cfg.latent_dim).to(self.device)

        self.encoder_optim = AdamW(self.encoder.parameters(), lr=cfg.lr)
        self.generator_optim = AdamW(self.generator.parameters(), lr=cfg.lr)
        self.discriminator_optim = AdamW(self.discriminator.parameters(), lr=cfg.lr)
        self.codecriminator_optim = AdamW(self.codecriminator.parameters(), lr=cfg.lr)

        self.l1 = AlphaReconstructionLoss()
        self.encoder_criterion = AlphaEncoderLoss()
        self.codecriminator_criterion = CodecriminatorLoss()
        self.generator_criterion = AlphaGeneratorLoss()
        self.discriminator_criterion = AlphaDiscriminatorLoss()

        imgs, labels = torch.load(cfg.dataset_path)

        (train_imgs, train_labels), (test_imgs, test_labels) = train_test_split(imgs, labels, test_size=cfg.val_size)

        traindata = StructuresDatasetTensor(imgs=train_imgs,
                                            labels=train_labels,
                                            transform=[random_roll,])
        testdata = StructuresDatasetTensor(imgs=test_imgs,
                                           labels=test_labels)
        self.train_loader = DataLoader(
            traindata, shuffle=True, batch_size=cfg.batch_size, drop_last=True, pin_memory=False, num_workers=cfg.num_workers
        )
        self.test_loader = DataLoader(
            testdata, shuffle=False, batch_size=cfg.batch_size, drop_last=True, pin_memory=False, num_workers=cfg.num_workers
        )
    
    def fit(self):
        run_name = f"casual_alpha_gan_{self.cfg.random_seed}"
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project="topology_topxpy", group="alpha_gan", name=run_name, job_type="vanilla_training"):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith('__')})

            gen_noise = torch.randn((self.cfg.generator_size, self.generator.latent_dim)).to(self.device)
            fixed_img = next(iter(self.test_loader))
            grid = make_grid(
                 fixed_img, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
            plt.imsave("images_for_reconstruction.jpg", grid)

            for e in range(self.cfg.num_epochs):

                total_discriminator_loss = 0
                total_generator_loss = 0
                total_encoder_loss = 0
                total_codecriminator_loss = 0
                total_counter = 0

                with tqdm(self.train_loader, desc=f"{e + 1}/{self.cfg.num_epochs} epochs", total=self.cfg.num_epochs) as t:
                    #self.generator.train()
                    self.encoder.train()
                    self.generator.train()
                    self.discriminator.train()
                    self.codecriminator.train()

                    for i, (x_real, label) in enumerate(self.train_loader):

                        x_real = x_real.to(self.device)
                        size = x_real.shape[0]
                        z = torch.randn(size, self.cfg.latent_dim, device=self.device)

                        # encoder
                        z_real = self.encoder(x_real)

                        # generator - reconstruction and generation step
                        x_reconstructed = self.generator(z_real)
                        x_fake = self.generator(z)

                        # discriminator
                        real_scores = self.discriminator(x_real)
                        reconstructed_scores = self.discriminator(x_reconstructed)
                        fake_scores = self.discriminator(x_fake)

                        # codecriminator
                        code_real = self.codecriminator(z_real)
                        code_fake = self.codecriminator(z)

                        # encoder training step
                        self.encoder_optim.zero_grad()
                        l1_loss = self.cfg.reconstruction_term * self.l1(x_reconstructed, x_real)
                        encoder_loss = l1_loss + self.encoder_criterion(code_real)
                        encoder_loss.backward(retain_graph=True)
                        self.encoder_optim.step()

                        total_encoder_loss += encoder_loss.item() * size
                        total_counter += size

                        # generator training step
                        self.generator_optim.zero_grad()
                        generator_loss = l1_loss + self.generator_criterion(reconstructed_scores, fake_scores)
                        generator_loss.backward(retain_graph=True)
                        self.generator_optim.step()

                        total_generator_loss += generator_loss.item() * size

                        # discriminator training step
                        self.discriminator_optim.zero_grad()
                        discriminator_loss = self.discriminator_criterion(real_scores, reconstructed_scores, fake_scores)
                        discriminator_loss.backward(retain_graph=True)
                        self.discriminator_optim.step()

                        total_discriminator_loss += discriminator_loss.item() * size

                        # codecriminator training step
                        self.codecriminator_optim.zero_grad()
                        codecriminator_loss = self.codecriminator_criterion(code_real, code_fake)
                        codecriminator_loss.backward()
                        self.codecriminator_optim.step()

                        total_codecriminator_loss += codecriminator_loss.item() * size

                        t.set_postfix({
                            "discriminator_loss": total_discriminator_loss / total_counter,
                            "generator_loss": total_generator_loss / total_counter,
                            "encoder_loss": total_encoder_loss / total_counter,
                            "codecriminator_loss": total_codecriminator_loss / total_counter
                        })

                        if not i % self.cfg.checkpoint_interval:
                            wandb.log({
                                "discriminator_loss": total_discriminator_loss / total_counter,
                                "generator_loss": total_generator_loss / total_counter,
                                "encoder_loss": total_encoder_loss / total_counter,
                                "codecriminator_loss": total_codecriminator_loss / total_counter
                        })
                        
                        if i == len(self.train_loader) - 1:
                            self.generator.eval()
                            self.encoder.eval()
                            
                            with torch.no_grad():
                                fake_images = self.generator.sample(self.cfg.generator_size, gen_noise)
                                reconstructed_images = self.generator(self.encoder(fixed_img))

                                generated_images = fake_images.cpu()
                                reconstructed_images = reconstructed_images.cpu()
                            
                            generated_images = make_grid(
                                generated_images, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
                            plt.imsave(os.path.join(self.cfg.img_dir, f"{e + 1}.jpg"), generated_images)

                            reconstructed_images = make_grid(
                                reconstructed_images, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
                            plt.imsave(os.path.join(self.cfg.recon_dir, f"{e + 1}.jpg"), reconstructed_images)

                            if not (e + 1) % self.cfg.checkpoint_interval:
                                self.save()

    def save(self):
        state_dict = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "generator_optim": self.generator_optim.state_dict(),
            "discriminator_optim": self.discriminator_optim.state_dict(),
            "codecriminator": self.codecriminator.state_dict(),
            "codecriminator_optim": self.codecriminator_optim.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict()
            }
        torch.save(state_dict, self.checkpoint_path)

    def load(self, filename: str):
        state_dict = torch.load(filename, map_location=self.device)
        
        self.generator.load_state_dict(state_dict["generator"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.generator_optim.load_state_dict(state_dict["generator_optim"])
        self.discriminator_optim.load_state_dict(state_dict["discriminator_optim"])
        self.codecriminator.load_state_dict(state_dict["codecriminator"])
        self.codecriminator_optim.load_state_dict(state_dict["codecriminator_optim"])
        self.encoder.load_state_dict(state_dict["encoder"])
        self.encoder_optim.load_state_dict(state_dict["encoder_optim"])
