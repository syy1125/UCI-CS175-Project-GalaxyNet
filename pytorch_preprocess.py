import math

import torch
import torch.nn as nn

log_255 = math.log(255)


class NormalizeImages(nn.Module):
    def forward(self, x):
        return torch.div(x, log_255)


class ColorIndex(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, 1, 255)
        log_br = torch.log(x[:, (2, 0)])
        return (log_br[:, (1,)] - log_br[:, (0,)]) / log_255


class Compose(nn.Module):
    def __init__(self, *modules):
        super().__init__()

        self.modules = modules

    def forward(self, x):
        for module in reversed(self.modules):
            x = module(x)
        return x


class Combine(nn.Module):
    def __init__(self, *modules):
        super().__init__()

        self.modules = modules

    def forward(self, x):
        return torch.cat(
            [module(x) for module in self.modules],
            dim=1
        )
