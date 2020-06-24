import tensorflow as tf


class FactEncoder(tf.keras.layers.Layer):

    def __init__(self, embedding_layer, encoder_hidden_size, recurrent_dropout, *args, **kwargs):
        super(FactEncoder, self).__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        self.encoder = tf.keras.layers.LSTM(
            units=encoder_hidden_size,
            recurrent_dropout=recurrent_dropout,
            go_backwards=True
        )

