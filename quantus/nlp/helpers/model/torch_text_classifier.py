from __future__ import annotations

import torch.nn as nn
import torch
from abc import abstractmethod
from typing import Generator, Dict, List
import numpy as np
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import batch_list


class TorchTextClassifier(TextClassifier):
    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator:
        original_parameters = self.weights
        model_copy = self.clone()

        modules = [
            layer
            for layer in model_copy.internal_model.named_modules()
            if (hasattr(layer[1], "reset_parameters") and len(layer.state_dict()) > 0)
        ]

        if order == "top_down":
            modules = modules[::-1]

        for module in modules:
            if order == "independent":
                model_copy.weights = original_parameters
            torch.manual_seed(seed=seed + 1)
            module[1].reset_parameters()

            yield module[0], model_copy

    def to_tensor(self, x: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self.device, **kwargs)

    def predict(self, text: List[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        if len(text) <= batch_size:
            return self.predict_on_batch(text)

        batched_text = batch_list(text, batch_size)
        logits = []

        for i in batched_text:
            logits.append(self.predict_on_batch(i))

        return np.vstack(logits)

    @abstractmethod
    def predict_on_batch(self, text: List[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return self.internal_model.state_dict()

    @weights.setter
    def weights(self, weights: Dict[str, torch.Tensor]):
        self.internal_model.load_state_dict(weights)

    @property
    @abstractmethod
    def internal_model(self) -> nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> TorchTextClassifier:
        raise NotImplementedError
