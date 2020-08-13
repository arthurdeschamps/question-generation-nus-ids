import tensorflow as tf


class Attention(tf.keras.layers.Layer):

    def __init__(self, attention_depth, attention_style, attention_dropout_rate, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.sequence_length = None

        self.attention_dropout = tf.keras.layers.Dropout(rate=attention_dropout_rate, name="attention_input_dropout")
        self.attention_matrix, self.attention_vector = None, None
        self.attention_style = attention_style
        if attention_style == "additive":
            self.attention_matrix = tf.keras.layers.Dense(
                units=attention_depth,
                dtype=tf.float32,
                name=f"{self.name}_additive_attention_matrix",
            )
            self.attention_vector = tf.keras.layers.Dense(
                units=1,
                dtype=tf.float32,
                name=f"{self.name}_additive_attention_vector"
            )
        else:
            raise NotImplementedError(f"Attention \"{attention_style}\" not implemented for question attention layer.")

    def build(self, input_shape):
        self.sequence_length = input_shape[1]

    def call(self, attended_vectors, decoder_hidden_state=None, apply_softmax=True, training=None, **kwargs):
        assert attended_vectors is not None
        assert decoder_hidden_state is not None
        # We need to duplicate the decoder's hidden state to be able to compute the alignment scores for each
        # attended vector
        decoder_hidden_state = tf.repeat(
            tf.expand_dims(decoder_hidden_state, axis=1),
            repeats=tf.shape(attended_vectors)[1],
            axis=1,
            name=f"decoder_hidden_state_repeated"
        )
        if self.attention_style == "additive":
            attention_input = tf.concat((decoder_hidden_state, attended_vectors), axis=-1)
            dense_input = self.attention_dropout(attention_input)
            dense_result = self.attention_matrix(dense_input)
            scores = self.attention_vector(tf.math.tanh(dense_result))
            if apply_softmax:
                return tf.math.softmax(scores, axis=-2)
            return scores
        raise NotImplementedError()
