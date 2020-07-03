from logging import info

import tensorflow as tf
import tensorflow_addons as tfa
from defs import REPEAT_Q_EMBEDDINGS_FILENAME, PAD_TOKEN
from models.RepeatQ.layers.decoder import Decoder
from models.RepeatQ.layers.embedding import Embedding
from models.RepeatQ.layers.fact_encoder import FactEncoder
from models.RepeatQ.layers.attention import Attention
from models.RepeatQ.model_config import ModelConfiguration


class RepeatQ(tf.keras.models.Model):

    def __init__(self, config=None, *args, **kwargs):
        super(RepeatQ, self).__init__(*args, **kwargs)
        if config is None:
            config = {}
        self.config = ModelConfiguration.build_config(config)
        self.vocabulary_word_to_id = self._build_vocabulary()
        self.embedding_layer = self._build_embedding_layer()
        self.fact_encoder = self._build_fact_encoder()
        self.decoder = self._build_decoder()

    def call(self, inputs, training=None, mask=None):
        facts = inputs["facts"]
        base_question = inputs["base_question"]
        if "generated_question_length" in inputs:
            generated_question_length = inputs["generated_question_length"]
        else:
            generated_question_length = self.config.max_generated_question_length

        batch_dim = tf.shape(facts)[0]

        # Create embeddings
        facts_embeddings = self.embedding_layer(facts)
        base_question_embeddings = self.embedding_layer(base_question)

        # Encode facts
        facts_hidden_states = self.fact_encoder(facts_embeddings, training=training)

        logits = self.decoder({
            "base_question_embeddings": base_question_embeddings,
            "facts_hidden_states": facts_hidden_states,
            "initial_hidden_state": tf.zeros(shape=(batch_dim, self.config.decoder_hidden_size),
                                             name="decoder_init_state"),
            "nb_steps": generated_question_length,
            "target": inputs["target"] if training else None
        })
        return logits
        return tf.reshape(
            logits_sequence.stack(), shape=(batch_dim, generated_question_length, -1), name="model_results"
        )

    def _build_decoder(self):
        question_attention = self._build_attention_layer("base_question_attention")
        facts_attention = self._build_attention_layer("facts_attention")
        return Decoder(
            embedding_layer=self.embedding_layer,
            question_attention_mechanism=question_attention,
            facts_attention_mechanism=facts_attention,
            units=self.config.decoder_hidden_size,
            recurrent_dropout=self.config.recurrent_dropout,
            vocab_size=len(self.vocabulary_word_to_id),
            readout_size=self.config.decoder_readout_size,
            bos_token=self.vocabulary_word_to_id[PAD_TOKEN],
            name="decoder"
        )

    def _build_fact_encoder(self):
        return FactEncoder(
            encoder_hidden_size=self.config.fact_encoder_hidden_size,
            recurrent_dropout=self.config.recurrent_dropout,
            name="fact_encoder"
        )

    def _build_attention_layer(self, name):
        return Attention(
            attention_style=self.config.question_attention,
            attention_depth=self.config.attention_depth,
            name=name
        )

    def _build_embedding_layer(self):
        return Embedding.new(
            vocabulary=self.vocabulary_word_to_id,
            is_pretrained=self.config.embeddings_pretrained,
            embedding_size=self.config.embedding_size,
            embedding_path=f"{self.config.data_dir}/{REPEAT_Q_EMBEDDINGS_FILENAME}",
            name="embedding_layer"
        )

    def _build_vocabulary(self):
        token_to_id = {}
        with open(self.config.vocabulary_path, mode='r') as vocab_file:
            for i, token in enumerate(vocab_file.readlines()):
                token_to_id[token.strip()] = i
        return token_to_id
