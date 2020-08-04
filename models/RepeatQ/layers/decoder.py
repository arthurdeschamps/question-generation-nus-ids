import tensorflow as tf
import tensorflow_addons as tfa


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_layer,
                 question_attention_mechanism,
                 facts_attention_mechanism,
                 units,
                 recurrent_dropout,
                 readout_size,
                 vocab_size,
                 bos_token,
                 **kwargs):
        """
        :param embedding_layer: Embedding layer to embed the tokens predicted by this decoder.
        :param question_attention_mechanism: Attention layer.
        :param facts_attention_mechanism: Attention layer.
        :param units: Number of units in the recurrent unit.
        :param recurrent_dropout: Recurrent dropout of the recurrent unit.
        :param readout_size: Size of the readout state.
        :param vocab_size: Number of words in the output vocabulary.
        :param bos_token: Beginning of sentence token (it's id).
        """
        super(Decoder, self).__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.lstm_cell = tf.keras.layers.LSTMCell(units, recurrent_dropout=recurrent_dropout)
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
        self.U_r = tf.keras.layers.Dense(units=self.readout_size, name="decoder_U_r")
        self.V_r = tf.keras.layers.Dense(units=self.readout_size, name="decoder_V_r")

        self.maxout = tfa.layers.Maxout(int(self.readout_size / 2))

        self.W_y = tf.keras.layers.Dense(units=self.vocabulary_size, name="decoder_W_y")

    def build(self, input_shape):
        self.batch_dim = input_shape["base_question_embeddings"][0]
        self.sequence_length = input_shape["base_question_embeddings"][1]

    def call(self, inputs, training=None, **kwargs):
        base_question_embeddings = inputs["base_question_embeddings"]
        facts_encodings = inputs["facts_encodings"]

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
        base_question_attention_vector = self._compute_question_attention_vectors(
            base_question_embeddings, hidden_state
        )
        # Compute fact attention vectors
        fact_attention_vector = self._compute_facts_attention_vectors(
            facts_encodings, hidden_state
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
        logits = self._output_layer(hidden_state, decoder_input, base_question_attention_vector)
        return logits, (hidden_state, carry_state)

    def _output_layer(self, hidden_state, previous_input, base_question_attention_vector):
        r_t = self.W_r(hidden_state) + self.U_r(previous_input) + self.V_r(base_question_attention_vector)
        maxout = self.maxout(r_t)
        logits = self.W_y(maxout)
        return logits

    def _compute_question_attention_vectors(self, base_question_embeddings, decoder_hidden_state):
        base_question_attention_weights = self.base_question_attention(
            base_question_embeddings,
            decoder_hidden_state=decoder_hidden_state
        )
        base_question_attention_vectors = tf.reduce_sum(
            tf.multiply(base_question_attention_weights, base_question_embeddings),
            axis=1,
            name="base_question_attention_vectors"
        )
        return base_question_attention_vectors

    def _compute_facts_attention_vectors(self, facts_encodings, decoder_hidden_state):
        # Collates the batch dimension and the fact index dimension in order to compute attention for each
        # fact separately
        old_shape = tf.shape(facts_encodings)
        facts_encodings = tf.reshape(
            facts_encodings,
            shape=tf.concat(([old_shape[0] * old_shape[1]], old_shape[2:]), axis=0),
            name="decoder_flattened_fact_encodings"
        )
        # Need to duplicate the decoder's hidden state to fit the new "batch size"
        decoder_hidden_state = tf.repeat(
            decoder_hidden_state, repeats=old_shape[1], axis=0, name="decoder_hidden_state_duplicated"
        )
        facts_attention_weights = self.facts_attention_mechanism(
            facts_encodings, decoder_hidden_state=decoder_hidden_state, apply_softmax=False
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

