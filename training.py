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
        training_data: np.ndarray, loader: DataLoader, preprocess_fn: Callable,
        model: nn.Module, loss_fn, optimizer,
        num_epochs: int, batch_size=50, lr_scheduler=None, dtype=torch.FloatTensor,
):
    """
    Trains the model using the given parameters.

    :param training_data: The training data, in a similar shape to what the dataloader would load for solutions
    :param loader: The dataloader to use for loading images
    :param preprocess_fn: The preprocessor function for images, or None if no preprocessing is to be done
    :param model: The model to train
    :param loss_fn: Loss function used in training
    :param optimizer: Optimizer used in training
    :param num_epochs: Number of epochs to train the model for
    :param batch_size: Number of images in a training batch
    :param lr_scheduler: If set, step this scheduler at the end of every epoch
    :param dtype: Data type
    :return:
    """
    rng = np.random.default_rng()
    iter_per_epoch = training_data.shape[0] // batch_size
    loss_record = []

    load_time = 0
    train_time = 0

    def load_batch():
        batch_indices = rng.choice(training_data.shape[0], batch_size)
        batch_data = training_data[batch_indices]
        images = loader.load_images(
            batch_data[:, 0].astype(np.int),
            rng.integers(0, 360, batch_data.shape[0])
        )

        if preprocess_fn is not None:
            images = preprocess_fn(images)

        return Variable(torch.from_numpy(images).type(dtype)), Variable(torch.from_numpy(batch_data[:, 1:]).type(dtype))

    def timed_load_batch():
        start_time = time.time()
        x, y = load_batch()
        end_time = time.time()
        return x, y, end_time - start_time

    def train_batch(x, y):
        scores = model(x)
        loss = loss_fn(scores, y)

        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with ThreadPoolExecutor() as executor:

        for epoch in range(num_epochs):
            print('Starting epoch {}/{}'.format(epoch + 1, num_epochs))
            model.train()

            epoch_loss = []
            data_preload_task: Union[Future, None] = None

            for i in range(iter_per_epoch):
                if data_preload_task is not None:
                    x_var, y_var, load_batch_time = data_preload_task.result()
                else:
                    x_var, y_var, load_batch_time = timed_load_batch()

                load_time += load_batch_time

                if i < iter_per_epoch - 1:
                    data_preload_task = executor.submit(timed_load_batch)

                train_start_time = time.time()

                train_batch(x_var, y_var)

                train_end_time = time.time()
                train_time += train_end_time - train_start_time

            print('Epoch {} mean loss {}'.format(epoch + 1, np.mean(epoch_loss)))
            loss_record.append(epoch_loss)

            if lr_scheduler is not None:
                lr_scheduler.step()

    print('Data loading time {}s, model training time {}s'.format(load_time, train_time))
