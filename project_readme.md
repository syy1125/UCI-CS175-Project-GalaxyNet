# Project files and their purposes

- `samples`: Folder containing 100 randomly selected training images for demo purposes.
- `src`: Contains the code that we have written over the course of this project.
    - `dataloader.py`: Our custom data loader, used in the early stages of the project and contains special image processing to augment our training data.
    - `knn.py`: Code related an experiment with K Nearest Neighbour classifier using incremental PCA. Currently unused because of bad performance.
    - `output_layer.py`: Our output normalization layer, utilizing the rules of the GalaxyNet questionnaire to normalize the output of the neural network.
    - `pytorch_init.py`: Helper functions for initializing the pytorch model.
    - `pytorch_preprocess.py`: Custom preprocessing layers to normalize or augment the input tensor.
    - `training.py`: Functions to help in training and validating our model.
    - `transforms.py`: Contains functions similar to `pytorch_preproces.py` but operates on numpy arrays rather than pytorch tensors.
    - `variable_stride.py`: A layer that mimics the effect of variable stride by selectively deleting rows and columns from the result of a stride-1 convolution layer. Currently unused as it failed to significantly improve our models.
- `effnet.pt`: Our demo model, selected for a balance of accuracy and speed.
- `project.ipynb`: Demo jupyter notebook.
- `project.html`: HTML printout of the demo jupyter notebook.
- `sample_solutions.csv`: Desired classifications for the 100 sample images included in the submission.