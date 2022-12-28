"""This module contains example LeNets and other simple architectures for PyTorch and tensorflow."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from importlib import util
from typing import Tuple


# Import different models depending on which deep learning framework is installed.
if util.find_spec("torch"):

    import torch

    class LeNet(torch.nn.Module):
        """
        A torch implementation of LeNet architecture.
            Adapted from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch.
        """

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
        """
        A torch implementation of LeNet architecture, with adaptive pooling.
            Adapted from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch.
        """

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
        """
        A torch implementation of 1D-convolutional architecture inspired from LeNet.
        """

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

    class LeNet3D(torch.nn.Module):
        """
        A torch implementation of 3D-LeNet architecture.
            Adapted from: <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py>
        """

        def __init__(self):
            super(LeNet3D, self).__init__()
            self.conv_1 = torch.nn.Conv2d(3, 6, 5)
            self.pool_1 = torch.nn.MaxPool2d(2, 2)
            self.pool_2 = torch.nn.MaxPool2d(2, 2)
            self.conv_2 = torch.nn.Conv2d(6, 16, 5)
            self.fc_1 = torch.nn.Linear(16 * 5 * 5, 120)
            self.fc_2 = torch.nn.Linear(120, 84)
            self.fc_3 = torch.nn.Linear(84, 10)
            self.relu_1 = torch.nn.ReLU()
            self.relu_2 = torch.nn.ReLU()
            self.relu_3 = torch.nn.ReLU()
            self.relu_4 = torch.nn.ReLU()

        def forward(self, x):
            x = self.pool_1(self.relu_1(self.conv_1(x)))
            x = self.pool_2(self.relu_2(self.conv_2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = self.relu_3(self.fc_1(x))
            x = self.relu_4(self.fc_2(x))
            x = self.fc_3(x)
            return x


if util.find_spec("tensorflow"):
    import tensorflow as tf

    def LeNetTF() -> tf.keras.Model:
        """
        A Tensorflow implementation of LeNet5 architecture.
        """

        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=(28, 28, 1),
                ),
                tf.keras.layers.AveragePooling2D(),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(3, 3), activation="relu", name="test_conv"
                ),
                tf.keras.layers.AveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=120, activation="relu"),
                tf.keras.layers.Dense(units=84, activation="relu"),
                tf.keras.layers.Dense(units=10),
            ],
            name="LeNetTF",
        )

    def ConvNet1DTF(n_channels: int, seq_len: int, n_classes: int) -> tf.keras.Model:

        """
        A Tensorflow implementation of 1D-convolutional architecture.
        """

        return tf.keras.Sequential(
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
