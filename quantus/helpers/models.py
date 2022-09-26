"""This module contains example LeNets for PyTorch and tensorflow."""
from importlib import util

# Import different models depending on which deep learning framework is installed.
from typing import Tuple

if util.find_spec("torch"):

    import torch

    class LeNet(torch.nn.Module):
        """Network architecture from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch."""

        def __init__(
            self,
        ):
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

    class LeNetAdaptivePooling(torch.nn.Module):
        """Network architecture from with adaptive pooling: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch."""

        @staticmethod
        def _eval_adaptive_size(input_size: int) -> int:
            conv1_output_size = (input_size - 5 + 1) // 2
            conv2_output_size = (conv1_output_size - 5 + 1) // 2
            return conv2_output_size

        def __init__(self, input_shape: Tuple[int, int, int]):
            super().__init__()
            n_channels = input_shape[0]
            adaptive_width = self._eval_adaptive_size(input_shape[1])
            adaptive_height = self._eval_adaptive_size(input_shape[2])
            adaptive_shape = (adaptive_width, adaptive_height)
            n_fc_input = adaptive_width * adaptive_height * 16

            self.conv_1 = torch.nn.Conv2d(n_channels, 6, 5)
            self.pool_1 = torch.nn.MaxPool2d(2, 2)
            self.relu_1 = torch.nn.ReLU()
            self.conv_2 = torch.nn.Conv2d(6, 16, 5)
            self.pool_2 = torch.nn.MaxPool2d(2, 2)
            self.relu_2 = torch.nn.ReLU()
            self.avg_pooling = torch.nn.AdaptiveAvgPool2d(adaptive_shape)
            self.fc_1 = torch.nn.Linear(n_fc_input, 120)
            self.relu_3 = torch.nn.ReLU()
            self.fc_2 = torch.nn.Linear(120, 84)
            self.relu_4 = torch.nn.ReLU()
            self.fc_3 = torch.nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool_1(self.relu_1(self.conv_1(x)))
            x = self.pool_2(self.relu_2(self.conv_2(x)))
            x = self.avg_pooling(x)
            x = x.view(x.shape[0], -1)
            x = self.relu_3(self.fc_1(x))
            x = self.relu_4(self.fc_2(x))
            x = self.fc_3(x)
            return x

    class ConvNet1D(torch.nn.Module):
        """1D-convolutional architecture inspired from LeNet."""

        def __init__(self, n_channels, n_classes):
            super().__init__()
            self.conv_1 = torch.nn.Conv1d(n_channels, 6, 5)
            self.pool_1 = torch.nn.MaxPool1d(2, 2)
            self.relu_1 = torch.nn.ReLU()
            self.conv_2 = torch.nn.Conv1d(6, 16, 5)
            self.pool_2 = torch.nn.MaxPool1d(2, 2)
            self.relu_2 = torch.nn.ReLU()

            # TODO: Use closed formula or use LazyLinear layers.
            if n_channels == 1:
                n_fc_input = 64
            elif n_channels == 3:
                n_fc_input = 352

            self.fc_1 = torch.nn.Linear(n_fc_input, 120)
            self.relu_3 = torch.nn.ReLU()
            self.fc_2 = torch.nn.Linear(120, 84)
            self.relu_4 = torch.nn.ReLU()
            self.fc_3 = torch.nn.Linear(84, n_classes)

        def forward(self, x):
            x = self.pool_1(self.relu_1(self.conv_1(x)))
            x = self.pool_2(self.relu_2(self.conv_2(x)))
            x = torch.flatten(x, start_dim=1)
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

    class ConvNet1DTF(Sequential):
        """1D-convolutional architecture."""

        def __init__(self, n_channels, seq_len, n_classes):
            super().__init__(
                [
                    tf.keras.layers.Input(shape=(seq_len, n_channels)),
                    tf.keras.layers.Conv1D(filters=6, kernel_size=5, strides=1),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.AveragePooling1D(pool_size=2, strides=2),
                    tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.AveragePooling1D(pool_size=2, strides=2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(84, activation="relu"),
                    tf.keras.layers.Dense(n_classes),
                ]
            )
            self.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
