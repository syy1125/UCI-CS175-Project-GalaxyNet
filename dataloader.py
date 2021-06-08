import os
from typing import List, Tuple, Union
import numpy as np
from PIL import Image


class DataLoader:
    def __init__(self, image_dir: str, soln_path: str):
        self.image_dir = image_dir
        self.soln_path = soln_path
        self.soln = None

    def load_images(
            self,
            galaxy_ids: Union[np.ndarray, List[int]],
            rotations: Union[None, int, List[int], np.ndarray] = None,
            translations: Union[None, Tuple[int, int], List[Tuple[int, int]], np.ndarray] = None,
            hflip: Union[bool, List[bool]] = False
    ):
        """
        Batch load galaxy images, each with the given rotation

        :param galaxy_ids: The ID of the galaxies to load
        :param rotations: The rotation, in degrees counterclockwise, to apply to each image
        :param translations: The translation, in (x, y) format, to apply to each image
        :param hflip: Whether or not to horizontally flip each image
        :return: numpy array of data type byte and shape N x C x W x H, where N is the number of data points
        """

        n = len(galaxy_ids)

        # Images are 424 x 424 RGB
        output = np.zeros((n, 3, 424, 424), dtype=np.ubyte)

        if rotations is not None and isinstance(rotations, int):
            rotations = [rotations] * n
        if translations is not None and isinstance(translations, tuple):
            translations = [translations] * n
        if isinstance(hflip, bool):
            hflip = [hflip] * n

        for i in range(n):
            file_path = os.path.join(self.image_dir, str(galaxy_ids[i]) + '.jpg')
            image: Image.Image = Image.open(file_path)

            if rotations is not None:
                image = image.rotate(rotations[i])
            if hflip[i]:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if translations is not None:
                image = image.transform(image.size, Image.AFFINE, (1, 0, translations[i][0], 0, 1, translations[i][1]))

            output[i] = np.array(image, dtype=np.byte).transpose((2, 0, 1))

        return output

    def load_all_solutions(self):
        if self.soln is None:
            self.soln = np.loadtxt(self.soln_path, skiprows=1, delimiter=',')
        return self.soln
