from functools import reduce
from typing import List
import numpy as np
from transformers import TFBertModel as BertModel
from data_utils.embeddings import Embedder
import tensorflow as tf
from tensorflow.keras import Model


class Bert(Model):

    def __init__(self, max_sequence_length=20, beam_search_size=3):
        """
        :param max_sequence_length: Maximum length of any generated question.
        :param beam_search_size: Number of beams to keep in memory during beach search.
        """
        super(Bert, self).__init__()
        self.max_sequence_length = tf.constant(max_sequence_length, dtype=tf.int32)
        self.beam_search_size = beam_search_size
        self.embedder = Embedder()
        self.pretrained_weights_name = 'bert-base-uncased'
        self.model = BertModel.from_pretrained(self.pretrained_weights_name)
        self.padding_token = tf.constant(self.embedder.tokenizer.pad_token_id, dtype=tf.int32)
        self.mask_token = tf.constant(self.embedder.tokenizer.mask_token_id, dtype=tf.int32)
        self.sep_token = tf.constant(self.embedder.tokenizer.sep_token_id, dtype=tf.int32)

        initializer = tf.initializers.glorot_uniform()
        hidden_state_size = 768

        self.W_sqg = tf.Variable(shape=(hidden_state_size, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initial_value=initializer(shape=(hidden_state_size, self.embedder.vocab_size())))
        self.b_sqg = tf.Variable(shape=(1, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initial_value=initializer(shape=(1, self.embedder.vocab_size())))

    def call(self, tokens, training=False, mask=None):
        if training is None or training:
            raise ValueError("Only call this function for evaluation/testing")
        beams_tensors = tf.concat(list(tf.identity(tokens) for _ in range(self.beam_search_size)), axis=0)
        # initialize all likelihoods to 0 except 1 to introduce diversity to the first loop
        initial_probs = np.zeros(shape=(self.beam_search_size,))
        initial_probs[0] = 1.0
        beams_probs = tf.Variable(initial_probs, dtype=tf.float32)
        i = tf.Variable(0, dtype=tf.int32)

        def compute_next_beams(ite, current_beams, current_beam_probs):
            beams_hidden_states = self.model(current_beams)[0]
            mask_states = beams_hidden_states[:, -1]
            word_distributions = tf.math.softmax(tf.add(tf.matmul(mask_states, self.W_sqg), self.b_sqg))
            top_preds_probs, top_preds_indices = tf.math.top_k(word_distributions, k=self.beam_search_size)
            new_beams = []
            new_beams_probs = []
            for j in range(self.beam_search_size):
                new_beams.extend(tf.unstack(self.embedder.generate_next_input_tokens(current_beams,
                                                                                     top_preds_indices[j],
                                                                                     padding_token=self.padding_token)))
                new_beams_probs.extend(tf.unstack(tf.multiply(current_beam_probs, top_preds_probs[j])))
            current_beam_probs, top_beams_indices = tf.math.top_k(new_beams_probs, k=self.beam_search_size)
            current_beams = tf.gather(tf.Variable(new_beams, dtype=tf.int32), top_beams_indices)
            current_beams.set_shape([self.beam_search_size, None])
            return tf.add(ite, 1), current_beams, current_beam_probs

        nb_generated_tokens, beams_tensors, final_beam_probs = tf.while_loop(
            lambda ite, current_beams, _: tf.logical_and(
                tf.less(ite, self.max_sequence_length),
                self._no_sep_token(current_beams)
            ),
            compute_next_beams,
            loop_vars=[i, beams_tensors, beams_probs]
        )

        # Only takes the generated tokens
        best_beam = beams_tensors[tf.argmax(final_beam_probs, axis=0)]
        without_padding = tf.gather(best_beam, indices=tf.where(tf.not_equal(best_beam, self.padding_token)), axis=0)
        generated_question = without_padding[-nb_generated_tokens:]
        return generated_question

    def step(self, tokens, correct_output_tokens, step=tf.Variable(0, dtype=tf.int32)):
        """
        To use for teacher forcing training
        :param tokens:
        :param correct_output_tokens:
        :param step:
        :return:
        """

        correct_next_outputs = correct_output_tokens[:, step]
        hidden_states = self.model(tokens)[0]
        mask_states = hidden_states[:, -1, :]
        word_distributions = tf.math.softmax(tf.add(tf.matmul(mask_states, self.W_sqg), self.b_sqg))
        # Uses the correct next token (teacher forcing)
        new_input_tokens = self.embedder.generate_next_input_tokens(
            tokens,
            correct_next_outputs,
            padding_token=self.padding_token
        )
        return word_distributions, correct_next_outputs, new_input_tokens

    def _no_sep_token(self, beams):
        sep_locations = tf.equal(tf.squeeze(beams), self.sep_token)
        res = tf.reduce_sum(
            tf.reduce_sum(tf.cast(sep_locations, dtype=tf.int32), axis=0),
            axis=0
        )
        # There are 3 sep tokens initially
        return tf.less_equal(res, tf.constant(3))
