import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Union
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .dataloader import DataLoader


# Based on the train function provided by pytorch sample code, but with some extra functionalities
def train(
        training_data: np.ndarray, loader: DataLoader,
        model: nn.Module, loss_fn, optimizer,
        num_epochs: int, batch_size: int = 50, lr_scheduler=None, dtype=torch.FloatTensor,
        timing_log: Union[list, None] = None, loss_record: Union[list, None] = None
):
    """
    Trains the model using the given parameters.

    :param training_data: The training data, in a similar shape to what the dataloader would load for solutions
    :param loader: The dataloader to use for loading images
    :param model: The model to train
    :param loss_fn: Loss function used in training
    :param optimizer: Optimizer used in training
    :param num_epochs: Number of epochs to train the model for
    :param batch_size: Number of images in a training batch
    :param lr_scheduler: If set, step this scheduler at the end of every epoch
    :param dtype: pytorch data type to use
    :param timing_log: Export record of execution time in the format of timing_log[epoch][batch_index] = (load_start, load_end, train_start, train_end)
    :param loss_record: Export record of training loss in the format of loss_record[epoch][batch_index] = batch_loss
    :return:
    """
    rng = np.random.default_rng()
    iter_per_epoch = training_data.shape[0] // batch_size

    def timed_load_batch():
        start_time = time.time()

        batch_indices = rng.choice(training_data.shape[0], batch_size)
        batch_data = training_data[batch_indices]
        images = loader.load_images(
            batch_data[:, 0].astype(np.int),
            rng.integers(0, 360, batch_data.shape[0])
        )
        x = Variable(torch.from_numpy(images).type(dtype))
        y = Variable(torch.from_numpy(batch_data[:, 1:]).type(dtype))

        end_time = time.time()
        return x, y, start_time, end_time

    def train_batch(x, y):
        scores = model(x)
        loss = loss_fn(scores, y)

        loss_value = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_value

    with ThreadPoolExecutor() as executor:

        for epoch in range(num_epochs):
            model.train()

            epoch_loss = []
            epoch_times = []
            data_preload_task: Future = executor.submit(timed_load_batch)

            for i in range(iter_per_epoch):
                x_var, y_var, load_start_time, load_end_time = data_preload_task.result()

                if i < iter_per_epoch - 1:
                    data_preload_task = executor.submit(timed_load_batch)

                train_start_time = time.time()
                batch_loss = train_batch(x_var, y_var)
                train_end_time = time.time()

                epoch_loss.append(batch_loss)
                epoch_times.append((load_start_time, load_end_time, train_start_time, train_end_time))

            if loss_record is not None:
                loss_record.append(epoch_loss)
            if timing_log is not None:
                timing_log.append(epoch_times)

            if lr_scheduler is not None:
                lr_scheduler.step()


def batch_predict_eval(
        data: np.ndarray, loader: DataLoader,
        model: nn.Module, batch_size: int = 50, dtype=torch.FloatTensor,
        extract_predictions: Callable[[Variable], np.ndarray] = lambda scores: scores.data.numpy()
):
    model.eval()

    n = data.shape[0]
    predictions = np.zeros((n, data.shape[1] - 1))

    def load_batch(start, end):
        return loader.load_images(data[start:end, 0].astype(np.int))

    with ThreadPoolExecutor() as executor:
        with torch.no_grad():
            data_preload_task: Future = executor.submit(load_batch, 0, np.minimum(batch_size, n))

            for i in range(0, n, batch_size):
                start_index = i
                end_index = np.minimum(i + batch_size, n)

                images = data_preload_task.result()

                if i + batch_size < n:
                    data_preload_task = executor.submit(load_batch, i + batch_size, np.minimum(i + 2 * batch_size, n))

                x_var = Variable(torch.from_numpy(images).type(dtype))

                scores = model(x_var)
                predictions[start_index:end_index] = extract_predictions(scores)

    return predictions
