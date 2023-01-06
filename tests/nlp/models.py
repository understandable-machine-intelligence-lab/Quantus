import tensorflow as tf
from keras.layers import GlobalAveragePooling1D, Dense, Dropout
from keras_nlp.layers import FNetEncoder, TokenAndPositionEmbedding
from ml_collections import ConfigDict
from typing import Optional


class FNetClassifier(tf.keras.Model):
    def __init__(self, config: ConfigDict, num_classes: int):
        super(FNetClassifier, self).__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=config.fnet.vocab_size,
            sequence_length=config.fnet.max_sequence_length,
            embedding_dim=config.fnet.embed_dim,
            mask_zero=True,
        )
        self.encoder1 = FNetEncoder(config.intermediate_dim)
        self.encoder2 = FNetEncoder(config.intermediate_dim)
        self.encoder3 = FNetEncoder(config.intermediate_dim)

        self.pool = GlobalAveragePooling1D()
        self.dropout = Dropout(0.1)
        self.top = Dense(num_classes)

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        input_embeddings: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if input_embeddings is None:
            x = self.embedding(inputs)
        else:
            x = input_embeddings

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        return self.top(x)
