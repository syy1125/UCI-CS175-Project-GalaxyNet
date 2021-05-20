import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .dataloader import DataLoader
from .transforms import preprocess


# Based on the train function provided by pytorch sample code, but with some extra functionalities
def train(
        training_data: np.ndarray, loader: DataLoader,
        model: nn.Module, loss_fn, optimizer,
        num_epochs: int, batch_size=50, lr_scheduler=None, dtype=torch.FloatTensor
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
    :return:
    """
    rng = np.random.default_rng()
    loss_record = []

    for epoch in range(num_epochs):
        print('Starting epoch {}/{}'.format(epoch + 1, num_epochs))
        model.train()

        epoch_loss = []

        for i in range(training_data.shape[0] // batch_size):
            batch_indices = rng.choice(training_data.shape[0], batch_size)
            batch_data = training_data[batch_indices]

            images = loader.load_images(
                batch_data[:, 0].astype(np.int),
                rng.integers(0, 360, batch_data.shape[0])
            )

            x = preprocess(images)

            x_var = Variable(torch.from_numpy(x).type(dtype))
            y_var = Variable(torch.from_numpy(batch_data[:, 1:]).type(dtype))

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {} mean loss {}'.format(epoch + 1, np.mean(epoch_loss)))
        loss_record.append(epoch_loss)

        if lr_scheduler is not None:
            lr_scheduler.step()
