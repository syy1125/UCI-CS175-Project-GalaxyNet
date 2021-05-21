import os
import time
from dataloader import DataLoader
from transforms import normalize_images, combine, color_index

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    loader = DataLoader(
        os.path.join(dirname, '../images_training_rev1'),
        os.path.join(dirname, '../training_solutions_rev1.csv')
    )

    start_time = time.time()
    soln = loader.load_all_solutions()
    print('Solution load time {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    image_batch = loader.load_images(soln[:1000, 0].astype(int))
    print('Batch load 1000 images time {:.2f}s'.format(time.time() - start_time))

    print('1000 images (RBG byte format) memory footprint {:.1f}MB'.format(
        (image_batch.size * image_batch.itemsize) / 1e6
    ))

    start_time = time.time()
    preprocessor = combine(normalize_images, color_index)
    image_norm_batch = preprocessor(image_batch)
    print(image_norm_batch.shape)
    print('Preprocessing took {:.2f}s'.format(time.time() - start_time))

    print('1000 images (preprocessed) memory footprint {:.1f}MB'.format(
        (image_norm_batch.size * image_norm_batch.itemsize) / 1e6
    ))
