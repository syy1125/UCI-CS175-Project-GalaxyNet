import numpy as np


def normalize_image(data: np.ndarray):
    return data.astype(np.float) / 255


def preprocess(data: np.ndarray):
    # TODO: More pre-processing!
    # Some ideas:
    # Color (log of blue/red ratio)
    # Brightness only
    normalized = normalize_image(data)

    return normalized
