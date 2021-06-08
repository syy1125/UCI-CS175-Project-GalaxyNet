import torch.nn as nn


def random_weights(module):
    if hasattr(module, 'weights'):
        nn.init.normal_(module.weights, std=0.01)
