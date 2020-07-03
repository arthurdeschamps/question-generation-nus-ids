import functools

import tensorflow as tf


class FactEncoder(tf.keras.layers.Layer):

    def __init__(self, encoder_hidden_size, recurrent_dropout, *args, **kwargs):
        super(FactEncoder, self).__init__(*args, **kwargs)
        self.supports_masking = True
        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=encoder_hidden_size,
            recurrent_dropout=recurrent_dropout,
            go_backwards=True,
            return_sequences=True
        ))

    def call(self, facts, training=None, mask=None, **kwargs):
        # Fact embeddings have shape (batch_size, nb_facts, fact_seq_length, embedding_size)
        # We collate the first 2 dimensions as the LSTM layer can only handle "one batch dimension" (and
        # nb_facts sort of implies a second batch dimension).
        batch_size, fact_count = tf.unstack(tf.shape(facts)[:2])

        def transform_shape(s):
            return tf.concat(([batch_size * fact_count], tf.unstack(s[2:])), axis=0)
        facts = tf.reshape(facts, shape=transform_shape(tf.shape(facts)))
        mask = tf.reshape(mask, shape=transform_shape(tf.shape(mask)))
        hidden_states = self.encoder(facts, training=training, mask=mask)
        # Recover real shape by splitting again into groups of facts
        correct_shape = tf.concat(([batch_size, fact_count], tf.shape(hidden_states)[1:]), axis=0)
        hidden_states = tf.reshape(
            hidden_states,
            shape=correct_shape
        )
        return hidden_states

