import logging
import os
from typing import Dict

import numpy as np
import tensorflow as tf


class Embedding(tf.keras.layers.Layer):

    _logger = logging.getLogger("Embedding")

    def __init__(self, embedding_matrix, *args, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.embedding_matrix = embedding_matrix
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return tf.nn.embedding_lookup(
            self.embedding_matrix, inputs, name="embedding_lookup"
        )

    def compute_mask(self, inputs, previous_mask=None):
        return tf.not_equal(inputs, 0)

    @property
    def size(self):
        return self.embedding_matrix.shape[1]

    @staticmethod
    def new(vocabulary, is_pretrained, embedding_size, embedding_path, **kwargs):
        if vocabulary is None:
            raise ValueError("Vocabulary cannot be None")

        var_name = "embedding_matrix"
        if is_pretrained:
            if embedding_path is None:
                raise ValueError("When using pretrained embeddings, a path to a file containing said embeddings must "
                                 "be passed.")
            if not os.path.isfile(embedding_path):
                raise ValueError(f"Path {embedding_path} either does not exist or is not a file.")
            embedding_matrix = tf.Variable(
                initial_value=Embedding._load_embedding_matrix(path=embedding_path),
                trainable=False,
                name=var_name,
                dtype=tf.float32
            )

        else:
            if embedding_size is None:
                raise ValueError("Please pass the embedding size if learning the embedding matrix.")
            embedding_matrix = tf.Variable(
                shape=(len(vocabulary), embedding_size),
                trainable=True,
                name=var_name
            )
        return Embedding(embedding_matrix=embedding_matrix, **kwargs)

    @staticmethod
    def _load_embedding_matrix(path):
        return np.load(path, allow_pickle=True)
