from logging import info

import tensorflow as tf

from defs import REPEAT_Q_EMBEDDINGS_FILENAME
from models.RepeatQ.layers.embedding import Embedding
from models.RepeatQ.layers.fact_encoder import FactEncoder
from models.RepeatQ.model_config import ModelConfiguration


class RepeatQ(tf.keras.models.Model):

    def __init__(self, config=None, *args, **kwargs):
        super(RepeatQ, self).__init__(*args, **kwargs)
        if config is None:
            config = {}
        self.config = ModelConfiguration.build_config(config)
        self.vocabulary_word_to_id = self._build_vocabulary()
        self.embedding_layer = self._build_embedding_layer()
        self.fact_encoder = self._build_fact_encoder(self.embedding_layer)

    def call(self, inputs, training=None, mask=None):
        facts = inputs["facts"]
        base_question = inputs["base_question"]

        # Create embeddings
        fact_embeddings = self.embedding_layer(facts)
        base_question_embedding = self.embedding_layer(base_question)
        return fact_embeddings

    def _build_fact_encoder(self, embedding_layer):
        return FactEncoder(
            embedding_layer=embedding_layer,
            encoder_hidden_size=self.config.fact_encoder_hidden_size,
            recurrent_dropout=self.config.encoder_recurrent_dropout
        )

    def _build_embedding_layer(self):
        return Embedding.new(
            vocabulary=self.vocabulary_word_to_id,
            is_pretrained=self.config.embeddings_pretrained,
            embedding_size=self.config.embedding_size,
            embedding_path=f"{self.config.data_dir}/{REPEAT_Q_EMBEDDINGS_FILENAME}"
        )

    def _build_vocabulary(self):
        token_to_id = {}
        with open(self.config.vocabulary_path, mode='r') as vocab_file:
            for i, token in enumerate(vocab_file.readlines()):
                token_to_id[token.strip()] = i
        return token_to_id
