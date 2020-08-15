import functools
from collections import namedtuple
import tensorflow as tf
from tensorflow import Tensor

from defs import REPEAT_Q_EMBEDDINGS_FILENAME, PAD_TOKEN, REPEAT_Q_TRAIN_CHECKPOINTS_DIR
from logging_mixin import LoggingMixin
from models.RepeatQ.layers.decoder import Decoder
from models.RepeatQ.layers.embedding import Embedding
from models.RepeatQ.layers.fact_encoder import FactEncoder
from models.RepeatQ.layers.attention import Attention
from models.RepeatQ.model_config import ModelConfiguration


class RepeatQ(LoggingMixin, tf.keras.models.Model):

    NetworkState = namedtuple("NetworkState", (
        "base_question",
        "base_question_encodings",
        "facts",
        "facts_encodings",
        "decoder_states",
        "observation",
        "is_first_step"
    ))

    def __init__(self,
                 voc_word_to_id,
                 config: ModelConfiguration,
                 *args,
                 **kwargs
                 ):
        super(RepeatQ, self).__init__(*args, **kwargs)

        self.config = config
        self.vocabulary_word_to_id = voc_word_to_id
        self.question_mark_id = voc_word_to_id["?"]

        # Layers construction
        self.embedding_layer = self._build_embedding_layer()
        self.fact_encoder = self._build_fact_encoder()
        self.decoder = self._build_decoder()

        # Dense layers for copy probability
        self.W_q_copy = tf.keras.layers.Dense(units=1, name="W_question_copy")
        self.W_q_copy_dropout = tf.keras.layers.Dropout(self.config.dropout_rate, name="W_question_copy_dropout")
        self.U_q_copy = tf.keras.layers.Dense(units=1, name="V_question_copy")
        self.U_q_copy_dropout = tf.keras.layers.Dropout(self.config.dropout_rate, name="V_question_copy_dropout")

        # Base question encoder
        self.base_question_encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=int(self.config.decoder_hidden_size / 2),
            recurrent_dropout=1e-9,
            dropout=0.5,
            return_sequences=True,
        ), merge_mode="concat", name="base_question_encoder")

        if config.restore_supervised_checkpoint:
            path = config.supervised_model_checkpoint_path
            if path is None:
                self.log.fatal("When 'restore_supervised_checkpoint' is enabled, you must provide a path to a model "
                               "checkpoint")
                exit(-1)
            path = f"{REPEAT_Q_TRAIN_CHECKPOINTS_DIR}/{path}"
            self.load_weights(path)
            self.log.info(f"Model successfully restored from '{path}'.")

    def call(self, inputs, constants=None, training=None, mask=None):

        voc_logits, (hidden_state, carry_state), (question_att_vector, question_copy_logits) = self.decoder(
            inputs={
                "base_question_encodings": inputs.base_question_encodings,
                "facts_encodings": inputs.facts_encodings,
                "previous_token_embedding": self.embedding_layer.embed_words(inputs.observation),
                "decoder_state": inputs.decoder_states
            },
            mask={
                "facts": tf.not_equal(inputs.facts, 0),
                "base_question": tf.not_equal(inputs.base_question, 0)
            },
            training=training
        )

        # Probability of copying a word from the base question
        q_copy_prob = tf.math.sigmoid(self.W_q_copy(self.W_q_copy_dropout(hidden_state)) +
                                      self.U_q_copy(self.U_q_copy_dropout(question_att_vector)))

        # Only decodes further for non-finished beams (last token wasn't a padding token or a question mark)
        mask = tf.expand_dims(tf.logical_or(
            tf.logical_and(
                tf.not_equal(inputs.observation, 0),
                tf.not_equal(inputs.observation, self.question_mark_id)
            ),
            inputs.is_first_step
        ), axis=-1)

        hidden_state = tf.where(mask, hidden_state, inputs.decoder_states[0])
        carry_state = tf.where(mask, carry_state, inputs.decoder_states[1])

        return voc_logits, question_copy_logits, q_copy_prob, (hidden_state, carry_state)

    def get_initial_state(self, base_question, base_question_features, facts, facts_features, batch_size, training=None):
        base_question_embeddings = self.embedding_layer({"sentence": base_question, "features": base_question_features})
        base_question_encodings = self.base_question_encoder(base_question_embeddings)
        facts_embeddings = self.embedding_layer({"sentence": facts, "features": facts_features})
        facts_encodings = self.fact_encoder(facts_embeddings, training=training)

        network_state = RepeatQ.NetworkState(
            base_question=base_question,
            base_question_encodings=base_question_encodings,
            facts=facts,
            facts_encodings=facts_encodings,
            decoder_states=(
                base_question_encodings[:, -1],  # Use the last hidden state of the question encoder as initial state
                tf.zeros(shape=(batch_size, self.config.decoder_hidden_size))
            ),
            observation=tf.zeros(shape=(batch_size,), dtype=tf.int32),
            is_first_step=True
        )
        return network_state

    @tf.function
    def get_actions(self, inputs, target, training, phase):
        from models.RepeatQ.trainer import RepeatQTrainer
        if training:
            batch_size = target.get_shape()[0]
        else:
            batch_size = inputs["base_question"].get_shape()[0]
        finished = tf.fill(dims=(batch_size,), value=False)

        if phase == RepeatQTrainer.supervised_phase and training:
            size = target.shape[1]
        else:
            size = self.config.max_generated_question_length

        base_question, facts = inputs["base_question"], inputs["facts"]
        base_question_features, facts_features = inputs["base_question_features"], inputs["facts_features"]
        network_state = self.get_initial_state(
            base_question=base_question,
            base_question_features=base_question_features,
            facts=facts,
            facts_features=facts_features,
            batch_size=batch_size,
            training=training
        )

        all_logits = tf.TensorArray(dtype=tf.float32, size=size, name="logits")
        actions = tf.TensorArray(dtype=tf.int32, size=size, name="agent_actions")
        ite = tf.constant(0, dtype=tf.int32)

        def _continue_loop(it, beams_finished):
            if phase == RepeatQTrainer.supervised_phase and training:
                return tf.less(it, size)
            else:
                return tf.reduce_any(tf.logical_not(beams_finished))

        while _continue_loop(ite, finished):
            voc_logits, question_word_logits, q_copy_prob, decoder_states = self(
                network_state, training=training
            )

            predicted_tokens = RepeatQ.get_output_tokens(
                voc_logits=voc_logits,
                base_question_logits=question_word_logits,
                question_copy_prob=tf.squeeze(q_copy_prob, axis=-1),
                base_question=base_question
            )

            if training:
                predicted_tokens = \
                    tf.where(
                        finished,
                        tf.zeros(shape=(batch_size,), dtype=tf.int32),
                        predicted_tokens
                    )
                predicted_tokens.set_shape(shape=(batch_size,))

            actions = actions.write(ite, predicted_tokens)
            # Final logits are the concatenation of (1 - copy prob) * vocabulary distribution and
            # copy prob * question attention
            def stable_softmax(x):
                z = x - tf.reduce_max(x, axis=-1, keepdims=True)
                numerator = tf.exp(z)
                denominator = tf.reduce_sum(numerator, axis=-1, keepdims=True)
                softmax = numerator / denominator
                return softmax
            pointer_softmax = tf.concat((
                (1.0 - q_copy_prob) * stable_softmax(voc_logits),
                 q_copy_prob * stable_softmax(question_word_logits)
            ), axis=-1, name="pointer_logits")
            all_logits = all_logits.write(ite, pointer_softmax)

            if phase == RepeatQTrainer.supervised_phase and training:
                # In supervised and training mode, we use teacher forcing
                observation = target[:, ite]
            else:
                # Otherwise, the model uses its own predicted tokens
                observation = predicted_tokens

            network_state = RepeatQ.NetworkState(
                base_question=network_state.base_question,
                base_question_encodings=network_state.base_question_encodings,
                facts=network_state.facts,
                facts_encodings=network_state.facts_encodings,
                decoder_states=decoder_states,
                observation=observation,
                is_first_step=False
            )
            ite = tf.add(ite, 1)
            finished = tf.logical_or(finished, tf.repeat(tf.equal(size, ite), repeats=(tf.shape(target)[0],)))
            if phase == RepeatQTrainer.supervised_phase and training:
                # Goes all the way to the target length
                finished = tf.logical_or(finished, tf.equal(target[:, tf.minimum(size-1, ite)], 0))
            else:
                finished = tf.logical_or(
                    finished,
                    tf.logical_or(tf.equal(predicted_tokens, 0), tf.equal(predicted_tokens, self.question_mark_id))
                )
            finished.set_shape(shape=(batch_size,))
        # Switch from time major to batch major
        actions = tf.transpose(actions.stack()[:ite])
        all_logits = tf.transpose(all_logits.stack()[:ite], perm=[1, 0, 2])
        return actions, all_logits

    @tf.function
    def beam_search(self, inputs, beam_search_size=5, training=False, return_probs=False):
        base_question, facts = inputs["base_question"], inputs["facts"]
        base_question_features, facts_features = inputs["base_question_features"], inputs["facts_features"]

        # Initialize variables
        batch_size = base_question.get_shape()[0]
        beam_length = self.config.max_generated_question_length
        collapsed_dimension = batch_size * beam_search_size

        def collapse_dims(t: Tensor):
            return tf.reshape(t, shape=(collapsed_dimension, *t.get_shape()[2:]))

        def recover_dims(t: Tensor):
            return tf.reshape(t, shape=(batch_size, beam_search_size, *t.get_shape()[1:]))

        def batchify(t: Tensor, name):
            t = tf.repeat(tf.expand_dims(t, axis=1), repeats=beam_search_size, axis=1, name=name)
            return collapse_dims(t)

        initial_network_state = self.get_initial_state(
            base_question=base_question,
            base_question_features=base_question_features,
            facts=facts,
            facts_features=facts_features,
            batch_size=base_question.get_shape()[0],
            training=training
        )
        base_question_encodings = batchify(initial_network_state.base_question_encodings, name="base_q_embds")
        base_question = batchify(initial_network_state.base_question, name="base_q")
        facts_encodings = batchify(initial_network_state.facts_encodings, "facts_encodings")
        facts = batchify(initial_network_state.facts, name="facts")
        decoder_states = (
            batchify(initial_network_state.decoder_states[0], "decoder_hidden_states"),
            batchify(initial_network_state.decoder_states[1], "decoder_carry_states")
        )
        observations = batchify(initial_network_state.observation, name="observations")
        beams = tf.zeros(shape=(batch_size, beam_search_size, beam_length), dtype=tf.int32, name="beams")
        beam_log_probs = tf.expand_dims(tf.repeat(
            tf.expand_dims(tf.concat((tf.zeros(shape=(beam_search_size-1,)), [-1.0]), axis=0), axis=0),
            repeats=batch_size,
            axis=0,
            name="beam_probs"
        ), axis=-1)
        first_step = tf.constant(True)
        best_beam = tf.zeros(shape=(batch_size, beam_length), dtype=tf.int32, name="best_beam")
        best_beam_prob = tf.fill((batch_size,), value=tf.float32.min, name="best_beam_prob")

        def beam_not_finished(_beam, _last_ind):
            return tf.logical_or(
                tf.less(_last_ind, 0),
                tf.logical_and(tf.not_equal(_beam[_last_ind], self.question_mark_id), tf.not_equal(_beam[_last_ind], 0))
            )

        def beams_not_finished(_beam_batches, _last_ind):
            res = tf.map_fn(
                lambda _beams: tf.map_fn(functools.partial(beam_not_finished, _last_ind=_last_ind), _beams, dtype=tf.bool),
                _beam_batches,
                dtype=tf.bool
            )
            return tf.expand_dims(res, axis=-1)

        for it in tf.range(beam_length):
            beam_network_state = RepeatQ.NetworkState(
                base_question=base_question,
                base_question_encodings=base_question_encodings,
                facts=facts,
                facts_encodings=facts_encodings,
                decoder_states=decoder_states,
                observation=observations,
                is_first_step=first_step
            )
            # Logits: [batch size * beam size, vocabulary size]
            voc_logits, q_copy_logits, q_copy_prob, decoder_states = self(beam_network_state, training=False)
            # [batch size, beam size, vocabulary size]
            voc_logits = recover_dims(voc_logits)
            q_copy_logits = recover_dims(q_copy_logits)
            q_copy_prob = recover_dims(q_copy_prob)
            # top_probs: [batch size, beam size, beam size]
            top_probs, top_words = RepeatQ.get_output_tokens(
                voc_logits, q_copy_logits, q_copy_prob, recover_dims(base_question), top_k=beam_search_size
            )
            decoder_states = tuple(recover_dims(s) for s in decoder_states)
            # [batch size, beam size, 1] -> [batch size, beam size, beam_size]
            beam_log_probs = tf.repeat(beam_log_probs, repeats=beam_search_size, axis=-1)
            top_probs = tf.math.add(tf.math.log(top_probs), beam_log_probs)
            # Masks finished beams probs
            beam_mask = beams_not_finished(beams, it-1)
            top_probs = tf.where(beam_mask, top_probs, tf.zeros_like(top_probs, dtype=tf.float32))
            top_words = tf.where(beam_mask, top_words, tf.zeros_like(top_words, dtype=tf.int32))
            # [batch size, beam size * beam size]
            top_probs = tf.reshape(top_probs, shape=(batch_size, beam_search_size ** 2))
            top_words = tf.reshape(top_words, shape=(batch_size, beam_search_size ** 2))
            # beam_probs: [batch size, beam size]
            beam_log_probs, top_beam_indices = tf.math.top_k(top_probs, k=beam_search_size)
            # [batch size, beam size, seq length] -> [batch size, beam size * beam size, seq length]
            beams = tf.repeat(beams, repeats=beam_search_size, axis=1)
            decoder_states = tuple(tf.repeat(s, beam_search_size, axis=1) for s in decoder_states)
            # [batch size, beam size * beam size, seq length] -> [batch size, beam size, seq length]
            beams = tf.gather(beams, indices=top_beam_indices, axis=1, batch_dims=1)
            top_words = tf.gather(top_words, indices=top_beam_indices, axis=1, batch_dims=1)
            beams = tf.concat((
                beams[:, :, :it], tf.expand_dims(top_words, axis=-1), beams[:, :, it+1:]
            ), axis=-1)
            beams.set_shape((batch_size, beam_search_size, beam_length))

            # Memorize best beam per batch so far
            beam_idx = tf.argmax(beam_log_probs, axis=-1)
            best_beam_per_batch = tf.gather(beams, tf.expand_dims(beam_idx, axis=-1), axis=1, batch_dims=1)
            beam_finished = tf.reshape(tf.logical_not(beams_not_finished(best_beam_per_batch, it)), shape=(batch_size,))
            best_beam_prob_per_batch = tf.reduce_max(beam_log_probs, axis=-1)
            # Needs to both have a higher likelihood and be a finished beam
            # We also divide by the beam length for normalization sake
            better_beam = tf.logical_and(
                tf.greater_equal(best_beam_prob_per_batch/tf.cast(it, tf.float32), best_beam_prob),
                beam_finished
            )
            best_beam = tf.where(
                tf.expand_dims(better_beam, axis=1),
                tf.squeeze(best_beam_per_batch, axis=1),
                best_beam
            )
            best_beam_prob = tf.where(better_beam, best_beam_prob_per_batch/tf.cast(it, tf.float32), best_beam_prob)

            decoder_states = tuple(tf.gather(s, top_beam_indices, axis=1, batch_dims=1) for s in decoder_states)
            decoder_states = tuple(collapse_dims(s) for s in decoder_states)
            observations = collapse_dims(top_words)
            beam_log_probs = tf.expand_dims(beam_log_probs, axis=-1)
            first_step = tf.constant(False)

            it += 1
        beam_log_probs = tf.squeeze(beam_log_probs, axis=-1)
        beam_indices = tf.argmax(beam_log_probs, axis=1)
        beam_log_probs = tf.reduce_max(beam_log_probs, axis=1)
        beams = tf.gather(beams, beam_indices, axis=1, batch_dims=1)
        cond = tf.expand_dims(tf.greater(best_beam_prob, beam_log_probs/tf.cast(beam_length, tf.float32)), axis=-1)
        best_beams = tf.where(cond, best_beam, beams)
        best_beam_probs = tf.where(tf.squeeze(cond, axis=1), best_beam_prob, beam_log_probs)
        if return_probs:
            return best_beams, best_beam_probs
        return best_beams

    @staticmethod
    def get_output_tokens(voc_logits, base_question_logits, question_copy_prob, base_question, top_k=1):
        base_question_distribution = tf.math.softmax(base_question_logits, axis=-1)
        voc_words_distribution = tf.math.softmax(voc_logits, axis=-1)
        # Prevents copying padding tokens by zero-ing out the logits where 0 tokens are found
        base_question_distribution = tf.where(
            tf.equal(base_question, 0), tf.zeros_like(base_question_distribution), base_question_distribution
        )
        if top_k == 1:
            question_indices_to_copy = tf.argmax(base_question_distribution, axis=-1)
            question_words_probs = tf.reduce_max(base_question_distribution, axis=-1)
            voc_words = tf.argmax(voc_words_distribution, axis=-1, output_type=tf.int32)
            voc_words_probs = tf.reduce_max(voc_words_distribution, axis=-1)
        else:
            question_words_probs, question_indices_to_copy = tf.math.top_k(base_question_distribution, k=top_k)
            voc_words_probs, voc_words = tf.math.top_k(voc_words_distribution, k=top_k)

        q_batch_dims = len(base_question.get_shape()) - 1
        question_words_to_copy = tf.gather(base_question, question_indices_to_copy, batch_dims=q_batch_dims)
        question_words_probs = question_copy_prob * question_words_probs
        voc_words_probs = (1.0 - question_copy_prob) * voc_words_probs
        cond = question_words_probs > voc_words_probs
        predicted_tokens = tf.where(cond, question_words_to_copy, voc_words, name="pred_tokens")
        if top_k == 1:
            return predicted_tokens
        token_probs = tf.where(cond, question_words_probs, voc_words_probs)
        return token_probs, predicted_tokens

    def _build_decoder(self):
        question_attention = self._build_attention_layer("base_question_attention")
        facts_attention = self._build_attention_layer("facts_attention")
        return Decoder(
            embedding_layer=self.embedding_layer,
            question_attention_mechanism=question_attention,
            facts_attention_mechanism=facts_attention,
            units=self.config.decoder_hidden_size,
            recurrent_dropout=self.config.recurrent_dropout,
            dropout_rate=self.config.dropout_rate,
            vocab_size=len(self.vocabulary_word_to_id),
            readout_size=self.config.decoder_readout_size,
            bos_token=self.vocabulary_word_to_id[PAD_TOKEN],
            name="decoder"
        )

    def _build_fact_encoder(self):
        return FactEncoder(
            encoder_hidden_size=self.config.fact_encoder_hidden_size,
            recurrent_dropout=self.config.recurrent_dropout,
            dropout_rate=self.config.dropout_rate,
            name="fact_encoder"
        )

    def _build_attention_layer(self, name):
        return Attention(
            attention_style=self.config.question_attention,
            attention_depth=self.config.attention_depth,
            attention_dropout_rate=self.config.attention_dropout_rate,
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
