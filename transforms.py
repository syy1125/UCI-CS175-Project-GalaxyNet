import numpy as np


def normalize_images(data: np.ndarray):
    return data.astype(np.float) / 255


# Kinda replicates the idea of a "B-R" color index typically used in astronomy
def color_index(data: np.ndarray):
    data = np.maximum(data, 1)  # Prevent log(0) singularity
    return (np.log(data[:, 2]) - np.log(data[:, 0])) / np.log(255)


def compose(*transforms):
    """
    Compose the transforms into a pipeline. The image enters on the *right*.
    """

    def result(data: np.ndarray):
        for transform_fn in reversed(transforms):
            data = transform_fn(data)
        return data

    return result


def combine(*transforms):
    """
    Combine the preprocessors by stacking their output on axis 1.
    """
    return lambda data: np.stack([transform_fn(data) for transform_fn in transforms], axis=1)
