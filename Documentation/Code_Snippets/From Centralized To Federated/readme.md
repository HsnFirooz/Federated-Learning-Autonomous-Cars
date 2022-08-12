## From Centralized To Federated

This PyTorch example is based on the [Deep Learning with PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial and uses the CIFAR-10 dataset (a RGB image classification task). The file `cifar.py` loads the dataset, trains a convolutional neural network (CNN) on the training set, and evaluates the trained model on the test set.

You can simply start the centralized training by running `cifar.py`:

```shell
python3 cifar.py
```

The next step is to use our existing project code in `cifar.py` and build a federated learning system based on it. The only things we need are a simple Flower server (in `server.py`) and a Flower client that connects Flower to our existing model and data (in `client.py`). The Flower client basically takes the already defined model and training code and tells Flower how to call it.

Start the server in a terminal as follows:

```shell
python3 server.py
```

Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py
```

Start client 2 in the second terminal:

```shell
python3 client.py
```
