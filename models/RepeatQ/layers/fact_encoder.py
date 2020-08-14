import functools

import tensorflow as tf


class FactEncoder(tf.keras.layers.Layer):

    def __init__(self, encoder_hidden_size, recurrent_dropout, dropout_rate, *args, **kwargs):
        super(FactEncoder, self).__init__(*args, **kwargs)
        self.supports_masking = True
        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=encoder_hidden_size,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout_rate,
            return_sequences=True
        ), merge_mode="concat")

    def call(self, facts, training=None, mask=None, **kwargs):
        # Fact embeddings have shape (batch_size, nb_facts, fact_seq_length, embedding_size)
        # We collate the first 2 dimensions as the LSTM layer can only handle "one batch dimension" (and
        # nb_facts sort of implies a second batch dimension).
        batch_size, fact_count, fact_length = tf.unstack(facts.shape[:3])

        def transform_shape(s):
            return tf.concat(([batch_size * fact_count], tf.unstack(s[2:])), axis=0)
        facts = tf.reshape(facts, shape=transform_shape(tf.shape(facts)))
        mask = tf.reshape(mask, shape=transform_shape(tf.shape(mask)))
        hidden_states = self.encoder(facts, training=training, mask=mask)
        # Recover real shape by splitting again into groups of facts
        hidden_states = tf.reshape(
            hidden_states,
            shape=(batch_size, fact_count, fact_length, -1)
        )
        return hidden_states

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

