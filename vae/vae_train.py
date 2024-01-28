from tqdm import tqdm
import os
import torch
from torch import nn
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
import wandb
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

from vae.models import VanillaVAE, ResidualVAE
from vae.gumbel_models import GumbelVAE, ResidualGumbelVAE
from vae.vae_configs import vae_config
from vae.blocks import VAECriterion, GumbelCriterion
from utils import train_test_split, make_dir, seed_everything
from dataset.dataset2 import StructuresDatasetTensor, random_roll


class VAETrainer:
    str2model = {
        "vanilla": VanillaVAE,
        "residual": ResidualVAE,
        "vanilla-gumbel": GumbelVAE,
        "residual-gumbel": ResidualGumbelVAE
    }

    def __init__(self,
                 cfg=vae_config()) -> None:
        self.device = cfg.device
        self.cfg = cfg

        self.checkpoint_path = os.path.join(cfg.checkpoint_dir,
                                            f"{cfg.dataset_path[:-3]}_{cfg.likelihood_type}_{cfg.model_type}_vae.pt")
        make_dir(cfg.checkpoint_dir)
        make_dir(cfg.img_dir)

        seed_everything(cfg.random_seed)

        self.model: nn.Module = self.str2model[cfg.model_type](cfg.in_channels,
                                                               cfg.hidden_channels,
                                                               cfg.latent_dim,
                                                               cfg.num_latents,
                                                               cfg.temperature).to(self.device)
        
        self.optim = torch.optim.AdamW(self.model.parameters(), cfg.lr)
        
        self.is_gumbel = "gumbel" in cfg.model_type

        if self.is_gumbel:
            self.criterion = GumbelCriterion(cfg.beta, cfg.likelihood_type)
        else:
            self.criterion = VAECriterion(cfg.beta, cfg.likelihood_type)
        
        self.resize = Resize((cfg.height_dim, cfg.width_dim))
        
        images, labels = torch.load(cfg.dataset_path)
        (train_images, train_labels), (test_images, test_labels) = train_test_split(images, labels, cfg.test_ratio)
        train_dataset = StructuresDatasetTensor(train_images,
                                                train_labels,
                                                transform=[random_roll,])
        test_dataset = StructuresDatasetTensor(test_images,
                                               test_labels)
        
        self.train_loader = DataLoader(
            train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, cfg.batch_size // 4, num_workers=cfg.num_workers
        )
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")
        run_name = f"{self.cfg.dataset_path[:-3]}_{self.cfg.likelihood_type}_{self.cfg.model_type}" + str(self.cfg.random_seed)

        with wandb.init(project="topology_topxpy",
                        group=f"{self.cfg.dataset_path[:-3]}_{self.cfg.likelihood_type}_{self.cfg.model_type}",
                        job_type="try to overfit",
                        name=run_name):
            
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith('__')})

            if self.is_gumbel:
                self.train_gumbel()
            else:
                self.train_basic()
    
    def log_scalars(self,
                    counter,
                    total_loss,
                    reconstruction_loss,
                    total_kl,
                    images,
                    reconstruction,
                    mode="vae_train"):
            
            wandb.log({
                f"{mode}/total_loss": total_loss / counter,
                f"{mode}/reconstruction_loss": reconstruction_loss / counter,
                f"{mode}/kl_divergence": total_kl / counter,
                f"{mode}/images": wandb.Image(images.cpu()),
                f"{mode}/reconstruction": wandb.Image(reconstruction.detach().cpu())
            })
        
    
    def train_gumbel(self):
        batch_size = self.cfg.batch_size
        test_batch_size = batch_size
        
        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()

            total_loss = 0
            total_reconstruction_loss = 0
            total_kl = 0
            counter = 0

            for (images, labels) in tqdm(self.train_loader):
                images = images.to(self.device)

                reconstruction, proposal_distribution, proposal_sample = self.model(images)
                total_loss_, reconstruction_loss, kl_divergence = self.criterion(images,
                                                                                proposal_distribution,
                                                                                proposal_sample,
                                                                                reconstruction)
                self.optim.zero_grad()
                total_loss_.backward()
                self.optim.step()

                total_loss += total_loss_.item() * batch_size
                total_reconstruction_loss += reconstruction_loss.item() * batch_size
                total_kl += kl_divergence.item() * batch_size
                
                counter += batch_size
            
            self.log_scalars(counter,
                             total_loss,
                             total_reconstruction_loss,
                             total_kl,
                             images,
                             reconstruction)

            # evaluation
            self.model.eval()

            total_loss = 0
            total_reconstruction_loss = 0
            total_kl = 0
            counter = 0

            for (images, _) in tqdm(self.test_loader):
                images = images.to(self.device)

                with torch.no_grad():
                    reconstruction, proposal_distribution, proposal_sample = self.model(images)

                    total_loss_, reconstruction_loss, kl = self.criterion(images,
                                                                          proposal_distribution,
                                                                          proposal_sample,
                                                                          reconstruction)
                    
                total_loss += total_loss_.item() * test_batch_size
                total_reconstruction_loss += reconstruction_loss.item() * test_batch_size
                total_kl += kl.item() * test_batch_size
                counter += test_batch_size
            
            self.log_scalars(counter,
                             total_loss,
                             total_reconstruction_loss,
                             total_kl,
                             images,
                             reconstruction,
                             mode="vae_eval")

            generated = self.model.generate(64, 8, self.device).cpu()

            generated_images = make_grid(
                 generated, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
            plt.imsave(os.path.join(self.cfg.img_dir, f"{epoch}.jpg"), generated_images)

            self.save()

    def train_basic(self):
        batch_size = self.cfg.batch_size
        test_batch_size = batch_size
        
        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()

            total_loss = 0
            total_reconstruction_loss = 0
            total_kl = 0
            counter = 0

            for (images, labels) in tqdm(self.train_loader):
                images = images.to(self.device)

                reconstruction, proposal_distribution = self.model(images)
                total_loss_, reconstruction_loss, kl_divergence = self.criterion(images,
                                                                                proposal_distribution,
                                                                                reconstruction)
                self.optim.zero_grad()
                total_loss_.backward()
                self.optim.step()

                total_loss += total_loss_.item() * test_batch_size
                total_reconstruction_loss += reconstruction_loss.item() * test_batch_size
                total_kl += kl_divergence.item() * test_batch_size
                
                counter += test_batch_size
            
            self.log_scalars(counter,
                             total_loss,
                             total_reconstruction_loss,
                             total_kl,
                             images,
                             reconstruction)

            # evaluation
            self.model.eval()

            total_loss = 0
            total_reconstruction_loss = 0
            total_kl = 0
            counter = 0

            for (images, _) in tqdm(self.test_loader):
                images = images.to(self.device)

                with torch.no_grad():
                    reconstruction, proposal_distribution = self.model(images)

                    total_loss_, reconstruction_loss, kl = self.criterion(images,
                                                                          proposal_distribution,
                                                                          reconstruction)
                    
                total_loss += total_loss_.item() * batch_size
                total_reconstruction_loss += reconstruction_loss.item() * batch_size
                total_kl += kl.item() * batch_size
                counter += batch_size
            
            self.log_scalars(counter,
                             total_loss,
                             total_reconstruction_loss,
                             total_kl,
                             images,
                             reconstruction,
                             mode="vae_eval")

            generated = self.model.generate(64, 8, self.device).cpu()

            generated_images = make_grid(
                 generated, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
            plt.imsave(os.path.join(self.cfg.img_dir, f"{epoch}.jpg"), generated_images)

            self.save()

    def save(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)
