import numpy as np
import torch.nn as nn


class EmulateVariableStride(nn.Module):
    """
    Emulate variable-stride CNN by deleting specific rows and columns from the image.
    Place this after a stride-1 convolution layer.
    """

    def __init__(self, pattern):
        super().__init__()
        self.pattern = np.hstack([[0], np.cumsum(pattern)])

    def forward(self, x):
        return x[:, :, self.pattern][:, :, :, self.pattern]
