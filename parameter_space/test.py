from hessian import HessianEigenvectors
from generator import Generator
from lpips import LPIPS


alexnet = LPIPS()

generator = Generator(1, num_blocks=[1, 1, 1, 1])
eigenvector_search = HessianEigenvectors(generator,
                                         alexnet,
                                         [2, 1, 2])

eigenvectors = eigenvector_search.top_k_eigenvectors()
