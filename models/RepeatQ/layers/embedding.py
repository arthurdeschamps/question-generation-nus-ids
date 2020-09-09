import logging
import os
from typing import Dict

import numpy as np
import tensorflow as tf


class Embedding(tf.keras.layers.Layer):

    _logger = logging.getLogger("Embedding")

    def __init__(self, embedding_matrix, nb_bio_tags, nb_pos_tags, *args, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.embedding_matrix = embedding_matrix
        self.supports_masking = True

        self.bio_embedding_layer = tf.keras.layers.Embedding(nb_bio_tags, 3, mask_zero=True, name="bio_embeddings")
        self.pos_embedding_layer = tf.keras.layers.Embedding(nb_pos_tags, 16, mask_zero=True, name="pos_embeddings")

    def call(self, inputs, mask=None):
        sentence = inputs["sentence"]
        features = inputs["features"]
        word_embeddings = self.embed_words(sentence)
        # Features are given in this order: (pos, bio)
        pos_embds = self.pos_embedding_layer(features[..., 0])
        bio_embds = self.bio_embedding_layer(features[..., 1])
        return tf.concat((word_embeddings, pos_embds, bio_embds), axis=-1)

    def embed_words(self, words):
        return tf.nn.embedding_lookup(
            self.embedding_matrix, words, name="embedding_lookup"
        )

    def compute_mask(self, inputs, previous_mask=None):
        return tf.not_equal(inputs["sentence"], 0)

    @property
    def size(self):
        return self.embedding_matrix.shape[1]

    @staticmethod
    def new(vocabulary, is_pretrained, embedding_size, embedding_path, nb_bio_tags, nb_pos_tags, **kwargs):
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
                initial_value=tf.initializers.GlorotUniform()(shape=(len(vocabulary), embedding_size), dtype=tf.float32),
                trainable=True,
                name=var_name
            )
        return Embedding(embedding_matrix=embedding_matrix, nb_bio_tags=nb_bio_tags, nb_pos_tags=nb_pos_tags, **kwargs)

    @staticmethod
    def _load_embedding_matrix(path):
        return np.load(path, allow_pickle=True)
