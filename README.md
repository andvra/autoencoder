# Autoencoder
This is an implementation of an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) used for dimensionality reduction. Basically, the input is encoded into a small code and decoded to recreate the original image. This is done by convoluting the input in three steps, fully connect the last layer to the code layer and then reverse the process by deconvoluting the result. The dimensions of the convolutional layers are subject to shape of a sample in the dataset, as can be seen in ``convnet.py``.
As an additional feature, there's a user interface from where you can manually modify the activations sent to latent space. This will generate new images on the fly. The server is based on Flask.
 The number of nodes in the code (= number of dimensions) are specified in the configuration file: for MNIST, good results (subjectively) are achived already using 20 nodes for the code.

## Datasets
The currently available inputs are (number of channels x width x height):
- MNIST (1 x 28 x 28)
- CIFAR10 (3 x 32 x 32)

Additional datasets can easily be added from the  ``dataloader.py``  file.

## Installation
1. Install [PyTorch](https://pytorch.org/)
2. Install additional packages by running:
```
pip install -r requirements.txt
```

## Configure
Open ``conf.json`` to configure. The keys of the JSON are:
- code_size: Size of the code (eg. 50)
- epochs: Number of epochs (eg. 200)
- dataset: Name of the dataset (mnist och cifar10)
- batch_size: Number of samples per batch. Can be increased or decrease to fit with available memory (RAM or GPU) (eg. 1024)

**Example file:**
```json
{
    "code_size": 50,
    "epochs": 200,
    "dataset": "cifar10",
    "batch_size": 1024
}
```

## Run
To run the autoencoder, setup your configuration file and run:
```
python autoencoder.py
```
States will be stored in the ``static`` folder. You'll be prompted for what state to use when running the web server:
```
python web.py
```
Open the UI and play around.
Some states are provided by default in this repo, so you can run the web UI without running the autoencoder.