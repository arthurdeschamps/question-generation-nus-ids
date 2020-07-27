import logging
from collections import namedtuple
import os
import tensorflow as tf
import numpy as np
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
                "previous_token_embedding": self.embedding_layer(inputs.observation),
                "decoder_state": inputs.decoder_states
        })

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

    @tf.function
    def get_actions(self, inputs, target, training, phase):
        from models.RepeatQ.trainer import RepeatQTrainer
        batch_size = target.get_shape()[0]
        finished = tf.fill(dims=(tf.shape(target)[0],), value=False)

        if phase == RepeatQTrainer.supervised_phase:
            size = target.shape[1]
        elif phase == RepeatQTrainer.reinforce_phase:
            size = self.config.max_generated_question_length
        else:
            raise NotImplementedError()

        base_question, facts = inputs["base_question"], inputs["facts"]
        network_state = self.get_initial_state(base_question, facts, batch_size)

        all_logits = tf.TensorArray(dtype=tf.float32, size=size, name="logits")
        actions = tf.TensorArray(dtype=tf.int32, size=size, name="agent_actions")
        ite = tf.constant(0, dtype=tf.int32)

        def _continue_loop(it, beams_finished):
            if phase == RepeatQTrainer.supervised_phase:
                return tf.less(it, size)
            else:
                return tf.reduce_any(tf.logical_not(beams_finished))

        while _continue_loop(ite, finished):
            logits, decoder_states = self(network_state, training=training)
            probs = tf.math.softmax(logits, axis=-1, name="probs")
            predicted_tokens = tf.argmax(probs, axis=-1, output_type=tf.int32, name="pred_tokens")
            if phase == RepeatQTrainer.reinforce_phase and training:
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
            if phase == RepeatQTrainer.supervised_phase:
                # Goes all the way to the target length
                finished = tf.logical_or(finished, tf.equal(target[:, tf.minimum(size-1, ite)], 0))
            elif phase == RepeatQTrainer.reinforce_phase:
                finished = tf.logical_or(
                    finished,
                    tf.logical_or(tf.equal(predicted_tokens, 0), tf.equal(predicted_tokens, self.question_mark_id))
                )
            else:
                raise NotImplementedError()
            finished.set_shape(shape=(batch_size,))
        # Switch from time major to batch major
        actions = tf.transpose(actions.stack()[:ite])
        all_logits = tf.transpose(all_logits.stack()[:ite], perm=[1, 0, 2])
        return actions, all_logits

    @tf.function
    def beam_search(self, inputs, beam_search_size=5):
        base_question = inputs["base_question"]
        facts = inputs["facts"]

        # Initialize variables
        initial_network_state = self.get_initial_state(
            base_question, facts, batch_size=base_question.get_shape()[0], training=False
        )
        base_question_embeddings = initial_network_state.base_question_embeddings
        facts_encodings = initial_network_state.facts_encodings
        decoder_states = tf.stack([tf.stack(initial_network_state.decoder_states) for _ in range(beam_search_size)])
        observations = tf.stack([initial_network_state.observation for _ in range(beam_search_size)])
        beams = tf.zeros(shape=(beam_search_size, 1), dtype=tf.int32)
        beam_probs = tf.concat((tf.zeros(shape=(beam_search_size-1,)), [1.0]), axis=0)
        first_step = tf.constant(True)
        it = tf.constant(0)

        best_beam = tf.constant([], dtype=tf.int32, name="best_beam")
        best_beam_prob = tf.constant(0.0, dtype=tf.float32, name="best_beam_prob")

        size = beam_search_size ** 2
        new_probs = tf.TensorArray(size=size, dtype=tf.float32)
        dec_states = tf.TensorArray(size=size, dtype=tf.float32)
        new_observations = tf.TensorArray(size=size, dtype=tf.int32)

        def not_finished(_beams, beam_index=None):
            def _beam_not_finished(_beam):
                return tf.logical_and(tf.not_equal(_beam[-1], self.question_mark_id), tf.not_equal(_beam[-1], 0))
            if beam_index is not None:
                return _beam_not_finished(_beams[beam_index])
            return tf.reduce_any([_beam_not_finished(_beams[i]) for i in range(beam_search_size)])

        while tf.logical_and(tf.logical_or(it == 0, not_finished(beams)), it < self.config.max_generated_question_length):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (beams, tf.TensorShape([beam_search_size, None])),
                    (best_beam, tf.TensorShape([None])),
                    (observations, tf.TensorShape([None, 1])),
                    (decoder_states, tf.TensorShape([None, 2, 1, self.config.decoder_hidden_size]))
                ]
            )
            new_beams = tf.zeros(shape=(size, 0), dtype=tf.int32)
            for i in tf.range(tf.shape(beams)[0]):
                tf.autograph.experimental.set_loop_options(shape_invariants=[(new_beams, tf.TensorShape([None, None]))])
                beam_network_state = RepeatQ.NetworkState(
                        base_question_embeddings=base_question_embeddings,
                        facts_encodings=facts_encodings,
                        decoder_states=tuple(tf.unstack(decoder_states[i])),
                        observation=observations[i],
                        is_first_step=first_step
                )
                if tf.logical_or(tf.equal(it, 0), not_finished(beams, i)):
                    logits, decoder_state = self(beam_network_state, training=False)
                    top_probs, top_words = tf.math.top_k(
                        tf.squeeze(tf.math.softmax(logits, axis=-1)),
                        k=beam_search_size
                    )
                    top_probs = top_probs * beam_probs[i]
                    if it == 0:
                        new_beam = tf.expand_dims(top_words, axis=1)
                    else:
                        new_beam = tf.concat((
                            tf.repeat(tf.expand_dims(beams[i], axis=0), repeats=beam_search_size, axis=0),
                            tf.expand_dims(top_words, axis=1)
                        ), axis=1)

                    k = i * beam_search_size
                    for beam_ind in tf.range(beam_search_size):
                        new_probs = new_probs.write(k+beam_ind, top_probs[beam_ind])
                        dec_states = dec_states.write(k+beam_ind, decoder_state)
                        new_observations = new_observations.write(k+beam_ind, [top_words[beam_ind]])
                else:
                    # For finished beams, save the most promising one if it's better than the current best
                    if beam_probs[i] > best_beam_prob:
                        best_beam_prob = beam_probs[i]
                        best_beam = beams[i]
                    # Sets probability to 0 to avoid further exploring these paths
                    for k in range(i*beam_search_size, (i+1)*beam_search_size):
                        new_probs = new_probs.write(k, 0.0)
                    new_beam = tf.zeros(shape=(beam_search_size, tf.shape(beams[0])[0] + 1), dtype=tf.int32)

                if tf.equal(tf.size(new_beams), 0):
                    new_beams = new_beam
                else:
                    new_beams = tf.concat((new_beams, new_beam), axis=0)

            probs, decoder_states, observations = new_probs.stack(), dec_states.stack(), new_observations.stack()
            probs = tf.reshape(probs, shape=(-1,))
            beam_probs, idx = tf.math.top_k(probs, k=beam_search_size)
            beams = tf.gather(new_beams, idx)
            decoder_states = tf.gather(decoder_states, idx)
            observations = tf.gather(observations, idx)
            first_step = tf.constant(False)
            it += 1
        if best_beam_prob > tf.reduce_max(beam_probs):
            return best_beam
        return beams[tf.argmax(beam_probs, axis=0)]

    def get_initial_state(self, base_question, facts, batch_size, training=True):
        base_question_embeddings = self.embedding_layer(base_question)
        facts_embeddings = self.embedding_layer(facts)
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
