from __future__ import annotations

from abc import abstractmethod
from typing import Generator, List
import tensorflow as tf
from keras.engine.base_layer_utils import TrackableWeightHandler
import numpy as np
from quantus.nlp.helpers.model.text_classifier import TextClassifier


class TensorFlowTextClassifier(TextClassifier):
    def get_random_layer_generator(
        self,
        order: str = "top_down",
        seed: int = 42,
    ) -> Generator:
        original_weights = self.weights.copy()
        model_copy = self.clone()
        layers = list(
            model_copy.internal_model._flatten_layers(  # noqa
                include_self=False, recursive=True
            )
        )
        layers = list(filter(lambda i: len(original_weights[i]) > 0, layers))

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                model_copy.weights = original_weights
            weights = layer.get_weights()
            np.random.seed(seed=seed + 1)
            layer.set_weights([np.random.permutation(w) for w in weights])
            yield layer.name, model_copy

    @property
    def weights(self) -> List[np.ndarray]:
        # TODO get weights as tensors??? to avoid copying to CPU?
        return self.internal_model.get_weights()

    @weights.setter
    def weights(self, weights: List[np.ndarray]):
        self.internal_model.set_weights(weights)

    @property
    @abstractmethod
    def internal_model(self) -> tf.keras.Model:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> TensorFlowTextClassifier:
        raise NotImplementedError
