from torch import nn
import torch

class SimpleLinearRegressionModal(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.rand(1,dtype= torch.float),requires_grad= True)
        self.bias = nn.Parameter(torch.rand(1,dtype= torch.float),requires_grad= True)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


