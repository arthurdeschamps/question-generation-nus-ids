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
        "base_question_embeddings",
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

        logits, (hidden_state, carry_state) = self.decoder({
                "base_question_embeddings": inputs.base_question_embeddings,
                "facts_encodings": inputs.facts_encodings,
                "previous_token_embedding": self.embedding_layer.embed_words(inputs.observation),
                "decoder_state": inputs.decoder_states
        }, training=training)

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

        return logits, (hidden_state, carry_state)

    #@tf.function
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
            logits, decoder_states = self(network_state, training=training)
            probs = tf.math.softmax(logits, axis=-1, name="probs")
            predicted_tokens = tf.argmax(probs, axis=-1, output_type=tf.int32, name="pred_tokens")
            if training:
                predicted_tokens = \
                    tf.where(
                        finished,
                        tf.zeros(shape=(batch_size,), dtype=tf.int32),
                        predicted_tokens
                    )
                predicted_tokens.set_shape(shape=(batch_size,))

            actions = actions.write(ite, predicted_tokens)
            all_logits = all_logits.write(ite, logits)

            if phase == RepeatQTrainer.supervised_phase and training:
                # In supervised and training mode, we use teacher forcing
                observation = target[:, ite]
            else:
                # Otherwise, the model uses its own predicted tokens
                observation = predicted_tokens

            network_state = RepeatQ.NetworkState(
                base_question_embeddings=network_state.base_question_embeddings,
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
        base_question_embeddings = batchify(initial_network_state.base_question_embeddings, name="base_q_embds")
        facts_encodings = batchify(initial_network_state.facts_encodings, "facts_encodings")
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
                    base_question_embeddings=base_question_embeddings,
                    facts_encodings=facts_encodings,
                    decoder_states=decoder_states,
                    observation=observations,
                    is_first_step=first_step
            )
            # Logits: [batch size * beam size, vocabulary size]
            logits, decoder_states = self(beam_network_state, training=False)
            # [batch size, beam size, vocabulary size]
            logits = recover_dims(logits)
            distributions = tf.math.softmax(logits, axis=-1)
            decoder_states = tuple(recover_dims(s) for s in decoder_states)
            # top_probs: [batch size, beam size, beam size]
            top_probs, top_words = tf.math.top_k(distributions, k=beam_search_size)
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

    def get_initial_state(self, base_question, base_question_features, facts, facts_features, batch_size, training=None):
        base_question_embeddings = self.embedding_layer({"sentence": base_question, "features": base_question_features})
        facts_embeddings = self.embedding_layer({"sentence": facts, "features": facts_features})
        facts_encodings = self.fact_encoder(facts_embeddings, training=training)

        network_state = RepeatQ.NetworkState(
            base_question_embeddings=base_question_embeddings,
            facts_encodings=facts_encodings,
            decoder_states=(
                tf.zeros(shape=(batch_size, self.config.decoder_hidden_size)),
                tf.zeros(shape=(batch_size, self.config.decoder_hidden_size))
            ),
            observation=tf.zeros(shape=(batch_size,), dtype=tf.int32),
            is_first_step=True
        )
        return network_state

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
