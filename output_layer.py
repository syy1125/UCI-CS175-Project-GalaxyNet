import torch
import torch.nn as nn


class GalaxyNetNormalizeOutput(nn.Module):
    def forward(self, x):
        self._normalize(x, 0, 3)  # class 1

        self._normalize(x, 15, 18, [0])  # class 7
        self._normalize(x, 3, 5, [1])  # class 2
        self._normalize(x, 25, 28, [3])  # class 9

        self._normalize(x, 5, 7, [4])  # class 3
        self._normalize(x, 7, 9, [5, 6])  # class 4
        self._normalize(x, 28, 31, [7])  # class 10
        self._normalize(x, 31, 37, [28, 29, 30])  # class 11
        self._normalize(x, 9, 13, [8, 31, 32, 33, 34, 35, 36])  # class 5

        # class 6 is weird. It appears that it always sums to 1 no matter what the viewer answered for Q1,
        # which is different from the flowchart.
        self._normalize(x, 13, 15)  # class 6
        self._normalize(x, 18, 25, [13])  # class 8

        return x

    def _normalize(self, x, start, end, sum_index=None):
        class_sum = torch.sum(x[:, start:end], dim=1, keepdim=True)
        non_zero = torch.where(class_sum[:, 0] > 0, True, False)
        x[non_zero, start:end] /= class_sum[non_zero]

        if sum_index is not None:
            x[:, start:end] *= torch.sum(x[:, sum_index], dim=1, keepdim=True)
