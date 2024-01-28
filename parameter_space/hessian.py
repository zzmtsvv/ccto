# source: https://arxiv.org/abs/2011.13786

from tqdm import trange
import os
from copy import deepcopy
from typing import List, Callable
import torch
from torch import nn
from lpips import LPIPS
from weight_deformator import get_conv_from_generator


def orthogonal_complement(v: torch.Tensor,
                          basis_vectors: List[torch.Tensor]) -> torch.Tensor:
    '''Gram–Schmidt process'''
    v = v.detach().clone()


    for vector in basis_vectors:
        norm = torch.sqrt(vector.pow(2).sum())
        vector /= norm

        vector = vector.to(v.device)
        dot_product = (v * vector).sum()
        v -= vector * dot_product
    
    return v


class HessianEigenvectors(nn.Module):
    def __init__(self,
                 generator: nn.Module,
                 lpips_model: LPIPS,
                 conv_indexes: List[int],
                 batch_size: int = 8,
                 cache_path: str = ".",
                 update_z: bool = False,
                 device: str = "cpu") -> None:
        super().__init__()

        self.device = device

        self.generator0 = generator.to(device).eval()
        self.generator1 = deepcopy(generator).to(device)
        self.lpips_model = lpips_model

        self.conv0 = get_conv_from_generator(self.generator0, *conv_indexes)
        self.conv1 = get_conv_from_generator(self.generator1, *conv_indexes)

        self.batch_size = batch_size
        self.cache_path = cache_path
        self.update_z = update_z
    
    def calc_g_batch(self,
               z: torch.Tensor,
               weight_diff: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img0 = self.generator0(z)
            img0 = img0.expand(-1, 3, -1, -1)
        
        self.conv1.weight = nn.Parameter(self.conv0.weight.data + weight_diff)

        img1 = self.generator1(z)
        img1 = img1.expand(-1, 3, -1, -1)

        lpips_distance = self.lpips_model(img0, img1).mean()

        self.zero_grad()
        lpips_distance.backward()

        return self.conv1.weight.grad
    
    def calc_g(self,
               z: torch.Tensor,
               weight_diff: torch.Tensor) -> torch.Tensor:
        assert not len(z) % self.batch_size

        counter = len(z) // self.batch_size

        g = 0
        for index in range(0, len(z), self.batch_size):
            z_batch = z[index: index + self.batch_size]
            g_batch = self.calc_g_batch(z_batch, weight_diff)

            g += g_batch
        
        return g / counter
    
    def forward_step(self,
                     z: torch.Tensor,
                     v: torch.Tensor,
                     epsilon: float) -> torch.Tensor:
        plus_delta = self.calc_g(z, epsilon * v)
        minus_delta = self.calc_g(z, -epsilon * v)
        norm = torch.sqrt(v.pow(2).sum())

        return (plus_delta - minus_delta) / (2 * (epsilon + 1e-14) * norm)
    
    def find_eigenvector(self,
                         z: torch.Tensor,
                         projector: Callable[[torch.Tensor], torch.Tensor],
                         max_iterations: int,
                         epsilon: float) -> torch.Tensor:
        v_current = torch.randn(self.conv0.weight.data.shape).to(self.device)
        v_current = projector(v_current)

        for i in trange(max_iterations):
            v_new = self.forward_step(z, v_current, epsilon)
            v_new = projector(v_new)

            norm_diff = torch.sqrt((v_new - v_current).pow(2).sum())
            print(f'Step: {i + 1}.\tNorm of (v_new - v_current): {norm_diff}')

            v_current = v_new
        
        return v_current
    
    def top_k_eigenvectors(self,
                           k: int = 64,
                           num_samples: int = 512,
                           max_iterations: int = 20,
                           epsilon: float = 0.1) -> List[torch.Tensor]:
        z = self.load_z(num_samples)
        eigenvectors = self.load_eigenvectors()

        for i in trange(len(eigenvectors), k):
            if self.update_z:
                z = torch.randn((num_samples, self.generator0.latent_dim)).to(self.device)
            
            print(f"Computing eigenvector {i + 1}")
            projector = lambda v: orthogonal_complement(v, eigenvectors)

            new_eigenvector = self.find_eigenvector(z, projector, max_iterations, epsilon)
            eigenvectors.append(new_eigenvector)
        
        self.save_eigenvectors(eigenvectors)

        return eigenvectors

    def load_z(self, num_samples: int) -> torch.Tensor:
        path = os.path.join(self.cache_path, "z_tmp.pt")
        try:
            z = torch.load(path).to(self.device)
            print("restored cached z")
            assert len(z) == num_samples, f"len(cached_z) != num_samples: {len(z)} != {num_samples}"
        except:
            z = torch.randn((num_samples, self.generator0.latent_dim)).to(self.device)
            torch.save(z, path)
        return z
    
    def load_eigenvectors(self) -> List[torch.Tensor]:
        path = os.path.join(self.cache_path, "eigenvectors_tmp.pt")
        try:
            eigenvectors = torch.load(path).cpu()
            print(f"Restored {len(eigenvectors)} cached eigenvectors")
        except:
            eigenvectors = []
        return eigenvectors
    
    def save_eigenvectors(self, eigenvectors: List[torch.Tensor]) -> None:
        path = os.path.join(self.cache_path, "eigenvectors_tmp.pt")
        torch.save(eigenvectors, path)
    
    def remove_cache(self):
        pass




if __name__ == "__main__":
    v = torch.rand(128)
