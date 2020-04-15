import numpy as np
from data_utils.embeddings import Embedder
import tensorflow as tf
from models.base_model import BaseModel


class Bert(BaseModel):
    """
    BERT implementation of this fine-tuning architecture: https://www.aclweb.org/anthology/D19-5821.pdf.
    """

    def __init__(self,
                 embedder: Embedder,
                 model,
                 max_generated_question_length=20,
                 beam_search_size=3,
                 max_sequence_length=512,
                 hidden_state_size=768):
        """
        :param max_generated_question_length: Limits the length of the generated questions to avoid infinite loops or
        lengthy questions.
        :param max_sequence_length: Maximum length of any generated question.
        :param beam_search_size: Number of beams to keep in memory during beach search.
        :param hidden_state_size: Number of hidden states generated at the output layer of the specific BERT model.
        """
        super(Bert, self).__init__()
        self.max_generated_question_length = tf.constant(max_generated_question_length, dtype=tf.int32)
        self.beam_search_size = beam_search_size
        self.embedder = embedder
        self.pretrained_weights_name = embedder.pretrained_weights_name
        self.model = model
        self.max_sequence_length = max_sequence_length

        initializer = tf.initializers.glorot_uniform()
        hidden_state_size = hidden_state_size

        # Simple linear layer used in conjunction with a softmax to output word distributions.
        self.W_sqg = tf.Variable(shape=(hidden_state_size, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initial_value=initializer(shape=(hidden_state_size, self.embedder.vocab_size())))
        self.b_sqg = tf.Variable(shape=(1, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initial_value=initializer(shape=(1, self.embedder.vocab_size())))
        self.beams_probs = tf.Variable(np.zeros(shape=(self.beam_search_size,)), dtype=tf.float32, trainable=False)
        self.ite = tf.Variable(0, dtype=tf.int32, trainable=False)

    def call(self, tokens, training=False, mask=None):
        """
        Generates a question from the given token sequence representing a paragraph.
        :param tokens: A token sequence representing a paragraph. Shape should be (1, sequence_length) (that is,
        batches are not accepted here).
        :param training: This should always set to false as this function should only be called during
        testing/validation.
        :param mask: Not used.
        :return: The generated question.
        """
        if training is None or training:
            raise ValueError("Only call this function for evaluation/testing")
        beams_tensors = tf.concat(list(tf.identity(tokens) for _ in range(self.beam_search_size)), axis=0)
        # initialize all likelihoods to 0 except 1 to introduce diversity to the first loop
        self.beams_probs.assign(np.zeros((self.beam_search_size,)))
        self.beams_probs[0].assign(1.0)

        self.ite.assign(0)

        # Keeps computing most probably beams until either reaching the maximum question length or generating
        # a SEP token (which indicates the end of the question).
        nb_generated_tokens, beams_tensors, final_beam_probs = tf.while_loop(
            lambda ite, current_beams, _: tf.logical_and(
                tf.less(ite, self.max_generated_question_length),
                self._no_sep_token(current_beams)
            ),
            self._compute_next_beams,
            loop_vars=[self.ite, beams_tensors, self.beams_probs],
            shape_invariants=[self.ite.shape, tf.TensorShape((3, None)), self.beams_probs.shape]
        )

        best_beam = beams_tensors[tf.argmax(final_beam_probs, axis=0)]
        # Get rid of the special tokens
        without_special_tokens = tf.reshape(tf.gather(
            best_beam,
            indices=tf.where(
                tf.logical_and(
                    tf.not_equal(best_beam, self.embedder.padding_token),
                    tf.not_equal(best_beam, self.embedder.mask_token)
                )
            ),
            axis=0
        ), shape=(-1,))
        generated_question = without_special_tokens[-nb_generated_tokens:]
        # Gets rid of any potential SEP token
        return tf.reshape(tf.gather(
            generated_question,
            indices=tf.where(tf.not_equal(generated_question, self.embedder.mask_token))
        ), shape=(-1,))

    def step(self, tokens, correct_output_tokens, step=tf.Variable(0, dtype=tf.int32, name='sequence_pointer')):
        """
        To use for teacher forcing training.
        :param tokens: A token sequence representing a context. Shape should be (batch_size, token_sequences_length).
        :param correct_output_tokens: The correct question. Shape expected to be (batch_size, 1).
        :param step: Pointer to the token being currently predicted.
        :return: Distributions for the next words, the correct next outputs and new input tokens to use for the next
        iteration.
        """
        correct_next_outputs = correct_output_tokens[:, step]
        attention_mask = tf.cast(tf.not_equal(tokens, self.embedder.padding_token), dtype=tf.int32,
                                 name='attention_mask')
        hidden_states = self.model(tokens, attention_mask=attention_mask)[0]
        mask_indices = tf.reshape(tf.reduce_sum(attention_mask, axis=1) - 1, shape=(hidden_states.shape[0], 1))
        mask_states = tf.gather_nd(hidden_states, mask_indices, batch_dims=1)
        word_logits = tf.add(tf.matmul(mask_states, self.W_sqg), self.b_sqg, name='word_logits')
        # Uses the correct next token (teacher forcing)
        new_input_tokens = self.embedder.generate_next_input_tokens(
            tokens,
            correct_next_outputs,
            padding_token=self.embedder.padding_token
        )
        return word_logits, correct_next_outputs, new_input_tokens

    def _compute_next_beams(self, ite, current_beams, current_beam_probs):
        """
        Computes beams for the next iteration (beam search algorithm).
        """
        beams_hidden_states = self.model(current_beams)[0]
        mask_states = beams_hidden_states[:, -1]
        word_distributions = tf.math.softmax(tf.add(tf.matmul(mask_states, self.W_sqg), self.b_sqg))
        # Computes the most probable k words for each distribution.
        top_preds_probs, top_preds_indices = tf.math.top_k(word_distributions, k=self.beam_search_size)
        new_beams = []
        new_beams_probs = []
        # Generates the next input sequences for every possible new token and calculates their likelihood.
        for j in range(self.beam_search_size):
            new_beams.extend(
                tf.unstack(self.embedder.generate_next_input_tokens(current_beams,
                                                                    top_preds_indices[j],
                                                                    padding_token=self.embedder.padding_token)))
            new_beams_probs.extend(tf.unstack(tf.multiply(current_beam_probs[j], top_preds_probs[j])))
        # Only keeps the k most probably sequences (out of k^2).
        current_beam_probs, top_beams_indices = tf.math.top_k(new_beams_probs, k=self.beam_search_size)
        current_beams = tf.gather(new_beams, top_beams_indices)
        current_beams.set_shape([self.beam_search_size, None])
        return tf.add(ite, 1), current_beams, current_beam_probs

    def _no_sep_token(self, beams):
        """
        :return: if a separation token has been generated by our model.
        """
        sep_locations = tf.equal(tf.squeeze(beams), self.embedder.sep_token)
        res = tf.reduce_sum(
            tf.reduce_sum(tf.cast(sep_locations, dtype=tf.int32), axis=0),
            axis=0
        )
        # There are 3 sep tokens initially
        return tf.less_equal(res, tf.constant(3))
