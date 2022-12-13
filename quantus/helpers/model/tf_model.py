"""This model creates the ModelInterface for Tensorflow."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import Dict, Optional, Tuple
from keras.layers import Dense
from keras import activations
from keras import Model
from keras.models import clone_model
import numpy as np
import tensorflow as tf
from cachetools import cachedmethod, LRUCache
import operator

from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers import utils


class TensorFlowModel(ModelInterface):
    """Interface for tensorflow models."""

    # All kwargs supported by Keras API https://keras.io/api/models/model_training_apis/.
    _available_predict_kwargs = [
        "batch_size",
        "verbose",
        "steps",
        "callbacks",
        "max_queue_size",
        "workers",
        "use_multiprocessing",
    ]

    def __init__(
        self,
        model: Model,
        channel_first: bool = True,
        softmax: bool = False,
        model_predict_kwargs: Optional[Dict[str, ...]] = None,
    ):
        if model_predict_kwargs is None:
            model_predict_kwargs = {}
        # Disable progress bar while running inference on tf.keras.Model.
        model_predict_kwargs["verbose"] = 0

        """
        Initialisation of ModelInterface class.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A model this will be wrapped in the ModelInterface:
        channel_first: boolean, optional
             Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        """
        super().__init__(
            model=model,
            channel_first=channel_first,
            softmax=softmax,
            model_predict_kwargs=model_predict_kwargs,
        )
        # get_hidden_representations needs to rebuild and re-trace the model.
        # In the case model has softmax on top, and we need linear activation, predict also needs to re-build the model.
        # This is computationally expensive, so we save the rebuilt model in cache and reuse it for consecutive calls.
        self.cache = LRUCache(100)

    def _get_predict_kwargs(self, **kwargs: Dict[str, ...]) -> Dict[str, ...]:
        """
        Use kwargs of predict call if specified, but don't overwrite object attribute.
        Filter out those, which are supported by Keras API.
        """
        all_kwargs = {**self.model_predict_kwargs, **kwargs}
        predict_kwargs = {
            k: all_kwargs[k] for k in all_kwargs if k in self._available_predict_kwargs
        }
        return predict_kwargs

    @cachedmethod(operator.attrgetter("cache"))
    def _build_model_with_linear_top(self) -> Model:
        """
        In a case model has a softmax on top, and we want linear,
        we have to rebuild the model and replace top with linear activation.
        Cache the rebuilt model and reuse it during consecutive predict calls.
        """

        config = self.model.layers[-1].get_config()
        config["activation"] = activations.linear
        weights = self.model.layers[-1].get_weights()

        output_layer = Dense(**config)(self.model.layers[-2].output)
        new_model = Model(inputs=[self.model.input], outputs=[output_layer])
        new_model.layers[-1].set_weights(weights)
        return new_model

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict on the given input.

        Parameters
        ----------
        x: np.ndarray
            A given input that the wrapped model predicts on.
        kwargs: optional
            Keyword arguments passed to tf.keras.Model.predict.

        Returns
        --------
        np.ndarray
            predictions of the same dimension and shape as the input, values in the range [0, 1].
        """
        # Generally, one should always prefer keras predict to __call__.
        # Reference: https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call.
        predict_kwargs = self._get_predict_kwargs(**kwargs)

        output_activation = self.model.layers[-1].activation
        target_activation = activations.softmax if self.softmax else activations.linear

        if output_activation == target_activation:
            return self.model.predict(x, **predict_kwargs)

        if self.softmax and output_activation == activations.linear:
            logits = self.model.predict(x, **predict_kwargs)
            return tf.nn.softmax(logits)

        # In this case model has a softmax on top, and we want linear.
        # We have to rebuild the model and replace top with linear activation.
        predict_model = self._build_model_with_linear_top()
        return predict_model.predict(x, **predict_kwargs)

    def shape_input(
        self,
        x: np.ndarray,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batched: bool = False,
    ):
        """
        Reshape input into model-expected input.

        Parameters
        ----------
        x: np.ndarray
            A given input that is shaped.
        shape: Tuple[int...]
            The shape of the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        batched: boolean
            Indicates if the first dimension should be expanded or not, if it is just a single instance.

        Returns
        -------
        np.ndarray
            A reshaped input.
        """
        if channel_first is None:
            channel_first = utils.infer_channel_first(x)
        # Expand first dimension if this is just a single instance.
        if not batched:
            x = x.reshape(1, *shape)

        # Set channel order according to expected input of model.
        if self.channel_first:
            return utils.make_channel_first(x, channel_first)
        return utils.make_channel_last(x, channel_first)

    def get_model(self):
        """
        Get the original tf model.
        """
        return self.model

    def state_dict(self):
        """
        Get a dictionary of the model's learnable parameters.
        """
        return self.model.get_weights()

    def load_state_dict(self, original_parameters):
        """Set model's learnable parameters."""
        self.model.set_weights(original_parameters)

    def get_random_layer_generator(self, order: str = "top_down", seed: int = 42):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        For cascading randomization, set order (str) to 'top_down'. For independent randomization,
        set it to 'independent'. For bottom-up order, set it to 'bottom_up'.

        Parameters
        ----------
        order: string
            The various ways that a model's weights of a layer can be randomised.
        seed: integer
            The seed of the random layer generator.

        Returns
        -------
        layer.name, random_layer_model: string, torch.nn
            The layer name and the model.
        """
        original_parameters = self.state_dict()
        random_layer_model = clone_model(self.model)

        layers = [
            _layer
            for _layer in random_layer_model.layers
            if len(_layer.get_weights()) > 0
        ]

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                random_layer_model.set_weights(original_parameters)
            weights = layer.get_weights()
            np.random.seed(seed=seed + 1)
            layer.set_weights([np.random.permutation(w) for w in weights])
            yield layer.name, random_layer_model
