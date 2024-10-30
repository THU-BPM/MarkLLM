import torch
import torch.nn as nn
import numpy as np


class DeltaNetwork(nn.Module):
    def __init__(self, input_dim=2048, layers=2, init_val=2.0):
        super(DeltaNetwork, self).__init__()
        if layers == 2:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
            )
        elif layers == 3:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
            )
        elif layers == 5:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1)
            )

        for layer in self.delta:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.init_val = init_val
        nn.init.constant_(self.delta[-1].bias, init_val)  # Set bias to the calculated value
    def forward(self, x):
        return self.delta(x)

class GammaNetwork(nn.Module):
    def __init__(self, input_dim=2048, layers=2, init_val=0.25):
        super(GammaNetwork, self).__init__()
        if layers == 2:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        elif layers == 3:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif layers == 5:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        for layer in self.gamma:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.init_val = init_val
        nn.init.constant_(self.gamma[-2].bias, np.log(init_val / (1 - init_val)))  # Set bias to the calculated value
    def forward(self, x):
        return self.gamma(x)