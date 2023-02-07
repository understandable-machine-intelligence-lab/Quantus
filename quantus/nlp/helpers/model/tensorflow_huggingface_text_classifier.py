from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING, Callable, Dict

from transformers import (
    TFPreTrainedModel,
    PreTrainedTokenizerBase,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.tf_utils import shape_list
import tensorflow as tf
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from quantus.nlp.helpers.utils import unpack_token_ids_and_attention_mask, batch_list

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import TF_TensorLike


class TFHuggingFaceTextClassifier(TextClassifier):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: TFPreTrainedModel,
    ):
        self.model = model
        self._tokenizer = HuggingFaceTokenizer(tokenizer)
        self.word_embedding_matrix = tf.convert_to_tensor(
            model.get_input_embeddings().weight, dtype=tf.float32
        )
        self.position_embedding_matrix = tf.convert_to_tensor(
            model.get_input_embeddings().position_embeddings, dtype=tf.float32
        )
        self.tensor_rt_model = self.try_convert_to_tensor_rt()

    @staticmethod
    def from_pretrained(handle: str) -> TFHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate TensorFlow models from HuggingFace Hub."""
        return TFHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            TFAutoModelForSequenceClassification.from_pretrained(handle),
        )

    def embedding_lookup(self, input_ids: TF_TensorLike, **kwargs) -> tf.Tensor:
        word_embeds = tf.nn.embedding_lookup(self.word_embedding_matrix, input_ids)
        input_shape = shape_list(word_embeds)[:-1]
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        positional_embeds = tf.nn.embedding_lookup(
            self.position_embedding_matrix, position_ids
        )
        return word_embeds + positional_embeds

    def __call__(
        self,
        inputs_embeds: TF_TensorLike,
        attention_mask: Optional[TF_TensorLike],
        **kwargs,
    ) -> tf.Tensor:
        if self.tensor_rt_model is None:
            return self.model(
                None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                training=False,
            ).logits

        inputs_embeds = tf.convert_to_tensor(inputs_embeds, dtype=tf.float32)
        attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)

        return self.tensor_rt_model(
            {
                "input_ids": None,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "training": False,
            }
        )["logits"]

    def predict(self, text: List[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        if self.tensor_rt_model is None:
            tokens = self._tokenizer.tokenize(text)
            return self.model.predict(
                tokens, verbose=0, batch_size=batch_size, **kwargs
            ).logits

        batched_text = batch_list(text, batch_size)
        logits = []
        for i in batched_text:
            logits.append(self._predict_on_batch_tensor_rt(i))
        return np.vstack(logits)

    def _predict_on_batch_tensor_rt(self, text: List[str]) -> np.ndarray:
        input_ids, attention_mask = unpack_token_ids_and_attention_mask(
            self._tokenizer.tokenize(text)
        )
        input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        attention_mask = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        logits = self.tensor_rt_model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "training": False,
            }
        )["logits"]
        return logits.numpy()

    @property
    def weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    @weights.setter
    def weights(self, weights: List[np.ndarray]):
        self.model.set_weights(weights)
        if self.tensor_rt_model is not None:
            self.tensor_rt_model = self.try_convert_to_tensor_rt()

    @property
    def tokenizer(self) -> HuggingFaceTokenizer:
        return self._tokenizer

    def try_convert_to_tensor_rt(
        self,
    ) -> Optional[Callable[[Dict[str, tf.Tensor]], tf.Tensor]]:
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdirname:
                tf.saved_model.save(self.model, f"{tmpdirname}/saved_model")
                converter = trt.TrtGraphConverterV2(
                    input_saved_model_dir=f"{tmpdirname}/saved_model"
                )
                converter.convert()
                converter.save(f"{tmpdirname}/tensor_rt_model")

                tensor_rt_model = tf.saved_model.load(f"{tmpdirname}/tensor_rt_model")

            return tensor_rt_model

        except Exception as e:
            print(f"Failed to convert model to TensorRT: {e}")
            return None
