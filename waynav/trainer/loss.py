import torch.nn as nn
import torch

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.eps = 0.001

    def forward(self, input, target,):
        g = torch.log(input+self.eps) - torch.log(target+self.eps)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)