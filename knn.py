from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Callable

import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import IncrementalPCA
from .dataloader import DataLoader
from .transforms import flatten_images


def data_pipeline(jobs: Iterable, load_data: Callable, use_data: Callable):
    jobs_iterator = iter(jobs)

    with ThreadPoolExecutor() as executor:
        try:
            job = next(jobs_iterator)
        except StopIteration:
            return

        data_load_task = executor.submit(load_data, job)

        done = False
        while not done:
            data = data_load_task.result()

            try:
                next_job = next(jobs_iterator)
            except StopIteration:
                done = True
            else:
                data_load_task = executor.submit(load_data, next_job)

            use_data(job, data)
            job = next_job


def ipca_fit(
        training_data: np.ndarray, loader: DataLoader,
        image_preprocess=flatten_images, batch_size=200, n_components=100
):
    transformer = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    def load_batch(start_index):
        end_index = np.minimum(start_index + batch_size, training_data.shape[0])
        batch_ids = training_data[start_index:end_index, 0].astype(np.int)
        return loader.load_images(batch_ids)

    def fit_batch(_, batch_data):
        batch_data = image_preprocess(batch_data)
        transformer.partial_fit(batch_data)

    data_pipeline(
        range(0, training_data.shape[0], batch_size),
        load_batch,
        fit_batch
    )

    return transformer


def ipca_transform(
        data: np.ndarray, loader: DataLoader, transformer: IncrementalPCA,
        image_preprocess=flatten_images, batch_size=200
):
    output = np.zeros((data.shape[0], transformer.n_components))

    def load_batch(start_index):
        end_index = np.minimum(start_index + batch_size, data.shape[0])
        batch_ids = data[start_index:end_index, 0].astype(np.int)
        return loader.load_images(batch_ids)

    def transform_batch(start_index, batch_data):
        batch_data = image_preprocess(batch_data)
        end_index = np.minimum(start_index + batch_size, data.shape[0])
        output[start_index:end_index] = transformer.transform(batch_data)

    data_pipeline(
        range(0, data.shape[0], batch_size),
        load_batch,
        transform_batch
    )

    return output


class KnnModel:
    def __init__(
            self,
            training_data: np.ndarray, loader: DataLoader,
            image_preprocess=flatten_images, n_components=10,
            k=5, ord=2
    ):
        self.train_soln = training_data
        self.image_preprocess = image_preprocess
        self.transformer = ipca_fit(
            training_data, loader,
            image_preprocess=self.image_preprocess, n_components=n_components
        )
        transformed_data = ipca_transform(
            training_data, loader, self.transformer,
            image_preprocess=self.image_preprocess
        )
        self.tree = KDTree(transformed_data)

        self.k = k
        self.ord = ord

    def predict(self, test_data: np.ndarray, loader: DataLoader):
        transformed_data = ipca_transform(test_data, loader, self.transformer, image_preprocess=self.image_preprocess)
        _, index_matrix = self.tree.query(transformed_data, k=self.k, p=self.ord, workers=-1)
        predictions = np.array([
            np.mean(self.train_soln[i], axis=0)[1:] for i in index_matrix
        ])
        return predictions
