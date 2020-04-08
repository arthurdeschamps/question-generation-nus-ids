from functools import reduce
from typing import List
import numpy as np
from transformers import TFBertModel as BertModel
from data_utils.embeddings import Embedder
import tensorflow as tf
from tensorflow.keras import Model


class Bert(Model):

    def __init__(self, max_sequence_length=15, beam_search_size=3):
        super(Bert, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.beam_search_size = beam_search_size
        self.embedder = Embedder()
        self.pretrained_weights_name = 'bert-base-uncased'
        self.model = BertModel.from_pretrained(self.pretrained_weights_name)

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

    def call(self, tokens, training=None, mask=None):
        beams_tensors = tf.Variable(list(tf.identity(tokens) for _ in range(self.beam_search_size)), dtype=tf.int32)
        # initialize all likelihoods to 0 except 1 to introduce diversity to the first loop
        beams_probs = list(0.0 for _ in range(self.beam_search_size))
        beams_probs[0] = 1.0

        i = 0
        while (i < self.max_sequence_length) and self._no_sep_token(beams_tensors):
            beams_hidden_states = (self.model(tf.expand_dims(beams_tensors[j], axis=0))[0] for j in
                                   range(beams_tensors.shape[0]))
            mask_states = (hidden_states[:, -1] for hidden_states in beams_hidden_states)
            word_distributions = (tf.math.softmax(tf.add(tf.matmul(mask_state, self.W_sqg), self.b_sqg))
                                  for mask_state in mask_states)
            top_preds_per_beam = list(
                tf.math.top_k(word_distribution, k=self.beam_search_size)
                for word_distribution in word_distributions
            )
            new_beams = []
            new_beams_probs = []
            for j in range(len(top_preds_per_beam)):
                for k in range(self.beam_search_size):
                    new_beams.append(self.embedder.generate_next_tokens(beams_tensors[j],
                                                                        top_preds_per_beam[j][1][0][k]))
                    new_beams_probs.append(beams_probs[j] * top_preds_per_beam[j][0][0][k])
            beam_probs, top_beams_indices = tf.math.top_k(new_beams_probs, k=self.beam_search_size)
            beams_tensors = tf.gather(tf.Variable(new_beams, dtype=tf.int32), top_beams_indices)
            beams_tensors.set_shape([self.beam_search_size, None])
            i += 1
        # Only takes the generated tokens
        generated_question_tokens = beams_tensors[tf.argmax(beams_probs, axis=0)][-i:]
        return generated_question_tokens

    def _no_sep_token(self, beams):
        return reduce(lambda b1, b2: b1 and b2,
                      (self.embedder.tokenizer.sep_token_id != beams[i, -1] for i in range(beams.shape[0])))
