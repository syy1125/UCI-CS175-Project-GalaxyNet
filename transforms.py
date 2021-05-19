import numpy as np


def normalize_image(data: np.ndarray):
    return data.astype(np.float) / 255
