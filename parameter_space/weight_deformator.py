# source: https://arxiv.org/abs/2011.13786
from typing import List
import torch
from torch import nn


def get_conv_from_generator(generator: nn.Module,
                            sequential_index: int = 2,
                            block_index: int = 1,
                            conv_index: int = 2) -> nn.Conv2d:
    # works only for architecture defined in topologу_topxpy

    legal_sequential_indexes = [2, 3, 4, 5]
    legal_block_indexes = [1, 2]
    legal_conv_indexes = [2, 5]

    assert sequential_index in legal_sequential_indexes, f"Legal sequentials: {legal_sequential_indexes}"
    assert block_index in legal_block_indexes, f"Legal blocks: {legal_block_indexes}"
    assert conv_index in legal_conv_indexes, f"Legal convs: {legal_conv_indexes}"

    return generator.net[sequential_index][block_index].block[conv_index]


class ConstantWeightDeformator(nn.Module):
    def __init__(self,
                 generator: nn.Module,
                 direction: torch.Tensor,
                 indexes: List[int] = [2, 1, 2]) -> None:
        super().__init__()

        self.generator = generator
        self.direction = direction
        self.conv = get_conv_from_generator(generator, *indexes)
        self.original_weight = self.conv.weight.data
    
    def deformate(self, epsilon: float) -> None:
        # works in-place for a given in __init__ generator
        self.conv.weight = nn.Parameter(self.original_weight + epsilon * self.direction)
    
    def disable_deformation(self):
        self.deformate(0.0)
    
    def __del__(self):
        self.disable_deformation()

