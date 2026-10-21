import torch
import torch.nn as nn

class LinearRegressionModal(nn.Module):
    def __init__(self):
        super().__init__()

        self.linearLayer = nn.Linear(in_features=1,out_features=1,bias= True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linearLayer(x)