from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from gans.gan_config import tox_config
from gans.models import Generator, PerceptualDiscriminator
from gans.losses import GeneratorReconstructionLoss, DiscriminatorLoss, PerceptualLoss
from utils import make_dir, seed_everything
from dataset.dataset2 import StructuresDatasetTensor, random_roll
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import wandb


class PerceptionGANTrainer:
    def __init__(self,
                 cfg=tox_config()) -> None:
        
        self.cfg = cfg
        self.device = cfg.device
        seed_everything(cfg.random_seed)
        self.checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{cfg.generator_loss}_perception-top-sngan.pt")

        make_dir(cfg.checkpoint_dir)
        make_dir(cfg.img_dir)

        self.generator = Generator(cfg.in_channels, cfg.base_channels, cfg.latent_dim, cfg.num_blocks).to(self.device)
        self.discriminator = PerceptualDiscriminator(cfg.in_channels, cfg.base_channels, cfg.num_blocks).to(self.device)

        self.gen_optim = torch.optim.AdamW(self.generator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        self.discr_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

        #self.gen_criterion = GeneratorLoss()
        self.gen_criterion = GeneratorReconstructionLoss(cfg.generator_loss)
        self.discr_criterion = DiscriminatorLoss()
        self.perception_loss = PerceptualLoss(cfg.use_gram_matrix)

        # init dataset and dataloader
        imgs, labels = torch.load(cfg.dataset_path)
        traindata = StructuresDatasetTensor(imgs=imgs,
                                            labels=labels,
                                            transform=[random_roll,])
        self.train_loader = DataLoader(
            traindata, shuffle=True, batch_size=cfg.batch_size, drop_last=True, pin_memory=False, num_workers=cfg.num_workers
        )
    
    def fit(self):
        gram_matrix = "gram_matrix" if self.cfg.use_gram_matrix else ""
        run_name = f"perception_{self.cfg.generator_loss}_" + gram_matrix + str(self.cfg.random_seed)
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project="topology_topxpy", group=self.cfg.generator_loss, name=run_name, job_type="vanilla_training"):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith('__')})

            gen_noise = torch.randn((self.cfg.generator_size, self.generator.latent_dim)).to(self.device)

            for e in range(self.cfg.num_epochs):
                self.generator.train()
                self.discriminator.train()

                total_discriminator_loss = 0
                total_generator_loss = 0
                total_discriminator_counter = 0
                total_generator_counter = 0
                total_perception_loss = 0
                total_perception_loss_counter = 0

                with tqdm(self.train_loader, desc=f"{e + 1}/{self.cfg.num_epochs} epochs", total=self.cfg.num_epochs) as t:
                    #self.generator.train()

                    for i, (x_true, label) in enumerate(self.train_loader):

                        # if label[-1] == 1:
                        #     # пропускаем плохие генерации из датасета - всего 7к картинок в датасете, на которых происходит обучение
                        #     continue

                        x_true = x_true.to(self.device)
                        size = x_true.shape[0]

                        x_fake = self.generator.sample(size)

                        # discriminator step
                        self.discr_optim.zero_grad()
                        
                        true_score, _ = self.discriminator(x_true)
                        fake_score, _ = self.discriminator(x_fake.detach())

                        discr_loss = self.discr_criterion(true_score, fake_score)
                        discr_loss.backward()
                        self.discr_optim.step()

                        total_discriminator_loss += discr_loss.item() * size
                        total_discriminator_counter += size

                        # generator step
                        self.gen_optim.zero_grad()
                        self.discriminator.disable_grads()

                        true_score, true_features = self.discriminator(x_true)
                        fake_score, fake_features = self.discriminator(x_fake)

                        perception_loss = self.perception_loss(true_features, fake_features)
                        generator_loss = self.gen_criterion(x_fake, fake_score)

                        overall_generator_loss = generator_loss + perception_loss
                        overall_generator_loss.backward()
                            
                        self.gen_optim.step()

                        total_generator_loss += generator_loss.item() * size
                        total_generator_counter += size

                        total_perception_loss += perception_loss.item() * size
                        total_perception_loss_counter += size

                        self.discriminator.enable_grads()

                        t.set_postfix({
                            "dicr_loss": total_discriminator_loss / total_discriminator_counter,
                            "gen_loss": total_generator_loss / total_generator_counter,
                            "perception_loss": total_perception_loss / total_perception_loss_counter
                        })

                        if not i % self.cfg.checkpoint_interval:
                            wandb.log({
                                "discriminator_loss": total_discriminator_loss / total_discriminator_counter,
                                "generator_loss": total_generator_loss / total_generator_counter,
                                "perception_loss": total_perception_loss / total_perception_loss_counter
                            })
                        
                        if i == len(self.train_loader) - 1:
                            self.generator.eval()
                            
                            with torch.no_grad():
                                generated_images = self.generator.sample(self.cfg.generator_size, gen_noise).cpu()
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
            "discriminator_optim": self.discr_optim.state_dict()
            }
        torch.save(state_dict, self.checkpoint_path)

    def load(self, filename: str):
        state_dict = torch.load(filename, map_location=self.device)
        
        self.generator.load_state_dict(state_dict["generator"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.gen_optim.load_state_dict(state_dict["generator_optim"])
        self.discr_optim.load_state_dict(state_dict["discriminator_optim"])



if __name__ == "__main__":
    t = PerceptionGANTrainer()
    # t.fit()
