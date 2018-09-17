import torch
from pyro.contrib.gp.kernels import ArcCosine


X = torch.tensor([[1.0, 0.0, 1.0], [2.0, 1.0, 3.0]])
Z = torch.tensor([[4.0, 5.0, 6.0], [3.0, 1.0, 7.0], [3.0, 1.0, 2.0]])
kernel = ArcCosine(3, [1.0, 2.0])

kernel.forward(X, Z)