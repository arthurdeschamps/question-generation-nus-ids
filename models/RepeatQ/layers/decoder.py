import tensorflow as tf
import tensorflow_addons as tfa


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_layer,
                 question_attention_mechanism,
                 facts_attention_mechanism,
                 units,
                 recurrent_dropout,
                 dropout_rate,
                 readout_size,
                 vocab_size,
                 bos_token,
                 **kwargs):
        """
        :param embedding_layer: Embedding layer to embed the tokens predicted by this decoder.
        :param question_attention_mechanism: Attention layer.
        :param facts_attention_mechanism: Attention layer.
        :param units: Number of units in the LSTM decoder.
        :param recurrent_dropout: Recurrent dropout of the recurrent unit.
        :param dropout_rate: Dropout rate of the RNN (for linear transformations) and the dense layers.
        :param readout_size: Size of the readout state.
        :param vocab_size: Number of words in the output vocabulary.
        :param bos_token: Beginning of sentence token (it's id).
        """
        super(Decoder, self).__init__(**kwargs)
        self.supports_masking = True
        self.embedding_layer = embedding_layer
        self.lstm_cell = tf.keras.layers.LSTMCell(units, recurrent_dropout=recurrent_dropout, dropout=dropout_rate)
        self.hidden_size = units
        self.zero_embedding_vector = None
        self.bos_token = bos_token
        self.bos_embedding = None
        self.batch_dim = None

        self.base_question_attention = question_attention_mechanism
        self.facts_attention_mechanism = facts_attention_mechanism
        self.readout_size = readout_size
        self.vocabulary_size = vocab_size
        self.embedding_layer = embedding_layer
        self.bos_token = bos_token
        self.facts_encodings = None
        self.base_question_embeddings = None
        self.batch_dim = None
        self.sequence_length = None

        # Output sub-layer weights
        self.W_r = tf.keras.layers.Dense(units=self.readout_size, name="decoder_W_r")
        self.W_r_dropout = tf.keras.layers.Dropout(rate=dropout_rate, name="W_r_input_dropout")
        self.U_r = tf.keras.layers.Dense(units=self.readout_size, name="decoder_U_r")
        self.U_r_dropout = tf.keras.layers.Dropout(rate=dropout_rate, name="U_r_input_dropout")
        self.V_r = tf.keras.layers.Dense(units=self.readout_size, name="decoder_V_r")
        self.V_r_dropout = tf.keras.layers.Dropout(rate=dropout_rate, name="V_r_input_dropout")

        self.maxout = tfa.layers.Maxout(int(self.readout_size / 2))

        self.W_y = tf.keras.layers.Dense(units=self.vocabulary_size, name="decoder_W_y")
        self.W_y_dropout = tf.keras.layers.Dropout(rate=dropout_rate, name="maxout_input_dropout")

    def build(self, input_shape):
        self.batch_dim = input_shape["base_question_embeddings"][0]
        self.sequence_length = input_shape["base_question_embeddings"][1]

    def call(self, inputs, training=None, mask=None, **kwargs):
        base_question_embeddings = inputs["base_question_embeddings"]
        base_question_mask = mask["base_question"]
        facts_encodings = inputs["facts_encodings"]
        facts_mask = mask["facts"]

        batch_dim = base_question_embeddings.shape[0]

        hidden_state, carry_state = tf.cond(
            tf.equal(tf.size(inputs["decoder_state"][0]), 0),
            lambda: tuple(self.lstm_cell.get_initial_state(batch_size=batch_dim, dtype=tf.float32)),
            lambda: inputs["decoder_state"]
        )
        previous_token_embedding = tf.cond(
            tf.equal(tf.size(inputs["previous_token_embedding"]), 0),
            lambda: tf.zeros((batch_dim, self.embedding_layer.size)),
            lambda: inputs["previous_token_embedding"]
        )

        # Compute question attention vectors
        base_question_attention_vector, base_question_attention_logits = self._compute_question_attention_vectors(
            base_question_embeddings, hidden_state, mask=base_question_mask, training=training
        )
        # Compute fact attention vectors
        fact_attention_vector = self._compute_facts_attention_vectors(
            facts_encodings, hidden_state, mask=facts_mask, training=training
        )

        # Create the decoder's next input
        decoder_input = tf.concat(
            (previous_token_embedding, fact_attention_vector, base_question_attention_vector),
            axis=1
        )
        output, (hidden_state, carry_state) = self.lstm_cell(
            decoder_input, (hidden_state, carry_state), training=training
        )

        # Compute logits
        logits = self._output_layer(hidden_state, decoder_input, base_question_attention_vector, training=training)
        return logits, (hidden_state, carry_state), (base_question_attention_vector, base_question_attention_logits)

    def _output_layer(self, hidden_state, previous_input, base_question_attention_vector, training=None):
        r_t = self.W_r(self.W_r_dropout(hidden_state)) + \
              self.U_r(self.U_r_dropout(previous_input)) + \
              self.V_r(self.V_r_dropout(base_question_attention_vector))
        maxout = self.maxout(r_t)
        logits = self.W_y(self.W_y_dropout(maxout, training=training))
        return logits

    def _compute_question_attention_vectors(self, base_question_embeddings, decoder_hidden_state, mask, training=None):
        base_question_attention_logits = self.base_question_attention(
            base_question_embeddings,
            decoder_hidden_state=decoder_hidden_state,
            apply_softmax=False,
            training=training,
            mask=mask
        )
        base_question_attention_weights = tf.math.softmax(base_question_attention_logits, axis=-2, name="q_attention")
        base_question_attention_vectors = tf.reduce_sum(
            tf.multiply(base_question_attention_weights, base_question_embeddings),
            axis=1,
            name="base_question_attention_vector"
        )
        return base_question_attention_vectors, tf.squeeze(base_question_attention_logits, axis=-1)

    def _compute_facts_attention_vectors(self, facts_encodings, decoder_hidden_state, mask, training=None):
        # Collates the batch dimension and the fact index dimension in order to compute attention for each
        # fact separately
        old_shape = tf.shape(facts_encodings)

        def flattened_shape(t):
            t_shape = t.get_shape()
            return tf.concat(([t_shape[0] * t_shape[1]], t_shape[2:]), axis=0)
        facts_encodings = tf.reshape(
            facts_encodings, shape=flattened_shape(facts_encodings), name="decoder_flattened_fact_encodings"
        )
        mask = tf.reshape(
            mask, shape=flattened_shape(mask), name="facts_mask"
        )
        # Need to duplicate the decoder's hidden state to fit the new "batch size"
        decoder_hidden_state = tf.repeat(
            decoder_hidden_state, repeats=old_shape[1], axis=0, name="decoder_hidden_state_duplicated"
        )
        facts_attention_weights = self.facts_attention_mechanism(
            facts_encodings,
            decoder_hidden_state=decoder_hidden_state,
            apply_softmax=False,
            training=training,
            mask=mask
        )
        # Transform shape into (batch_size, number of facts * fact sequence length) so we can subsequently
        # get the top m attention scores for each batch
        facts_attention_weights = tf.reshape(facts_attention_weights, shape=(old_shape[0], -1))
        # (batch size, number of facts * fact sequence length, hidden dimension)
        facts_encodings = tf.reshape(facts_encodings, shape=(old_shape[0], old_shape[1] * old_shape[2], old_shape[3]))
        # Only keep the m highest fact attention scores and compute weights through softmax, with m being the max fact
        # sequence length
        max_attention_scores, max_indices = tf.math.top_k(
            facts_attention_weights,
            k=old_shape[2],
            name="max_fact_attention_scores"
        )
        max_attention_weights = tf.expand_dims(
            tf.math.softmax(max_attention_scores), axis=-1, name="max_fact_attention_weights"
        )
        selected_facts_encodings = tf.gather(facts_encodings, max_indices, name="selected_facts_encodings",
                                             batch_dims=1)
        facts_attention_vectors = tf.reduce_sum(
            max_attention_weights * selected_facts_encodings, axis=1, name="facts_attention_vectors"
        )
        return facts_attention_vectors

