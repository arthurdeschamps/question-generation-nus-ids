import functools
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf


class Transformer(Model):

    def __init__(self,
                 embedder,
                 model,
                 max_sequence_length,
                 max_generated_question_length=40,
                 beam_search_size=3,
                 hidden_state_size=768,
                 **kwargs):
        """
        :param max_generated_question_length: Limits the length of the generated questions to avoid infinite loops or
        lengthy questions.
        :param max_sequence_length: Maximum length of any generated question.
        :param beam_search_size: Number of beams to keep in memory during beach search.
        :param hidden_state_size: Number of hidden states generated at the output layer of the specific BERT model.
        """
        super(Transformer, self).__init__()
        self.model = model
        self.max_sequence_length = max_sequence_length
        self.embedder = embedder
        self.max_generated_question_length = tf.constant(max_generated_question_length, dtype=tf.int32)
        self.beam_search_size = beam_search_size
        self.pretrained_weights_name = self.embedder.pretrained_weights_name
        self.question_mark_token = self.embedder.tokenizer.encode(" ?")[0]

        initializer = tf.initializers.glorot_uniform()

        # Simple linear layer used in conjunction with a softmax to output word distributions.
        self.W_sqg = tf.Variable(shape=(hidden_state_size, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=False,
                                 name="weight_head",
                                 initial_value=initializer(shape=(hidden_state_size, self.embedder.vocab_size())))
        self.b_sqg = tf.Variable(shape=(1, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=False,
                                 name="bias_head",
                                 initial_value=initializer(shape=(1, self.embedder.vocab_size())))
        self.beams_probs = tf.Variable(np.zeros(shape=(self.beam_search_size,)), dtype=tf.float32, trainable=False)
        self.ite = tf.Variable(0, dtype=tf.int32, trainable=False)

    def beam_search(self, tokens, produce_distribution_func):
        """
        Generates a question from the given token sequence representing a paragraph.
        :param tokens: A token sequence representing a paragraph. Shape should be (1, sequence_length) (that is,
        batches are not accepted here).
        :param training: This should always set to false as this function should only be called during
        testing/validation.
        :param mask: Not used.
        :return: The generated question.
        """
        beams_tensors = tf.concat(list(tf.identity(tokens) for _ in range(self.beam_search_size)), axis=0)
        # initialize all likelihoods to 0 except 1 to introduce diversity to the first loop
        self.beams_probs.assign(np.zeros((self.beam_search_size,)))
        self.beams_probs[0].assign(1.0)

        self.ite.assign(0)

        # Keeps computing most probably beams until either reaching the maximum question length or generating
        # a SEP token (which indicates the end of the question).
        beams_generating = tf.constant([True for _ in range(self.beam_search_size)],
                                       shape=(self.beam_search_size,),
                                       dtype=tf.bool,
                                       name="beams_generating")
        nb_generated_tokens, beams, final_beam_probs, _ = tf.while_loop(
            lambda _, __, ___, in_generation: tf.reduce_any(in_generation),
            functools.partial(self._compute_next_beams, produce_distribution=produce_distribution_func),
            loop_vars=(self.ite, beams_tensors, self.beams_probs, beams_generating),
            shape_invariants=(self.ite.shape, tf.TensorShape((self.beam_search_size, None)),
                              self.beams_probs.shape, beams_generating.shape),
            maximum_iterations=self.max_generated_question_length
        )

        best_beam = beams[tf.argmax(final_beam_probs, axis=0)]
        generated_question = best_beam[-nb_generated_tokens:]
        # Gets rid of any potential SEP token
        return tf.reshape(tf.gather(
            generated_question,
            indices=tf.where(tf.not_equal(generated_question, self.embedder.mask_token))
        ), shape=(-1,))

    def _compute_next_beams(self, ite, beams, beam_probs, generating_beams: tf.Variable, produce_distribution):
        """
        Computes beams for the next iteration (beam search algorithm).
        """

        expanding_indices = tf.where(generating_beams)
        expanding_beams = tf.reshape(tf.gather(beams, expanding_indices), shape=(tf.size(expanding_indices), -1))
        expanding_beam_probs = tf.reshape(tf.gather(beam_probs, expanding_indices), shape=(-1,))

        word_distributions = produce_distribution(expanding_beams)
        # Computes the most probable k words for each distribution.
        top_preds_probs, top_preds_indices = tf.math.top_k(word_distributions, k=self.beam_search_size)
        # Generates the next input sequences for every possible new token and calculates their likelihood.
        expanding_beams = tf.concat((
            tf.reshape(tf.repeat(expanding_beams, repeats=self.beam_search_size, axis=0),
                       shape=(tf.shape(expanding_beams)[0]*self.beam_search_size, tf.shape(expanding_beams)[1])),
            tf.reshape(top_preds_indices, shape=(-1, 1)),
        ), axis=1)
        expanding_beam_probs = tf.multiply(
            tf.reshape(tf.repeat(expanding_beam_probs, repeats=self.beam_search_size, axis=0),
                       shape=(tf.shape(expanding_beams)[0],)),
            tf.reshape(top_preds_probs, shape=(-1,))
        )

        generated_indices = tf.where(tf.logical_not(generating_beams))
        finished_beams = tf.squeeze(tf.gather(beams, generated_indices, axis=0), axis=1)
        finished_beam_probs = tf.squeeze(tf.gather(beam_probs, generated_indices, axis=0), axis=1)

        beams = tf.cond(
            tf.greater(tf.size(generated_indices), 0),
            lambda: tf.concat((expanding_beams, tf.pad(
                finished_beams,
                paddings=((0, 0), (0, tf.shape(expanding_beams)[1] - tf.shape(finished_beams)[1])),
                constant_values=self.embedder.padding_token,
                mode="CONSTANT"
            )), axis=0),
            lambda: expanding_beams
        )

        beam_probs = tf.cond(
            tf.greater(tf.size(generated_indices), 0),
            lambda: tf.concat((expanding_beam_probs, finished_beam_probs), axis=0),
            lambda: expanding_beam_probs
        )

        # Only keeps the k most probably sequences (out of k^2).
        top_beam_probs, top_beams_indices = tf.math.top_k(beam_probs, k=self.beam_search_size)
        top_beams = tf.gather(beams, top_beams_indices)
        generating_beams = tf.reshape(self._no_eos(top_beams), shape=(self.beam_search_size,))
        return tf.add(ite, 1), top_beams, top_beam_probs, generating_beams

    def _no_eos(self, beams):
        """
        :return: For each beam, if it has generated an end-of-sentence token.
        """
        sep_locations = tf.equal(tf.squeeze(beams), self.embedder.sep_token)
        sep_counts = tf.reduce_sum(tf.cast(sep_locations, dtype=tf.int32), axis=1)
        # There are 3 sep tokens initially for BERT encodings
        no_sep = tf.less_equal(sep_counts, 3)

        no_question_mark = tf.reduce_all(tf.not_equal(tf.squeeze(beams), self.question_mark_token), axis=1)
        return tf.logical_and(no_sep, no_question_mark)
