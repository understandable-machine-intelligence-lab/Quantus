"""This module contains example LeNets for PyTorch and tensorflow."""
from importlib import util

# Import different models depending on which deep learning framework is installed.

if util.find_spec("torch"):

    import torch

    class LeNet(torch.nn.Module):
        """Network architecture from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch."""

        def __init__(self):
            super().__init__()
            self.conv_1 = torch.nn.Conv2d(1, 6, 5)
            self.pool_1 = torch.nn.MaxPool2d(2, 2)
            self.relu_1 = torch.nn.ReLU()
            self.conv_2 = torch.nn.Conv2d(6, 16, 5)
            self.pool_2 = torch.nn.MaxPool2d(2, 2)
            self.relu_2 = torch.nn.ReLU()
            self.fc_1 = torch.nn.Linear(256, 120)
            self.relu_3 = torch.nn.ReLU()
            self.fc_2 = torch.nn.Linear(120, 84)
            self.relu_4 = torch.nn.ReLU()
            self.fc_3 = torch.nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool_1(self.relu_1(self.conv_1(x)))
            x = self.pool_2(self.relu_2(self.conv_2(x)))
            x = x.view(x.shape[0], -1)
            x = self.relu_3(self.fc_1(x))
            x = self.relu_4(self.fc_2(x))
            x = self.fc_3(x)
            return x


if util.find_spec("tensorflow"):

    import tensorflow as tf

    from tensorflow.keras.models import Sequential

    class LeNetTF(Sequential):
        """Network architecture adapted from: https://www.tensorflow.org/datasets/keras_example."""

        def __init__(self):
            super().__init__(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(10),
                ]
            )
            self.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
