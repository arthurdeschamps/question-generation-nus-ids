from collections import namedtuple
from typing import List
import nltk.translate.bleu_score as bleu
from tensorflow import Tensor
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
import numpy as np
from tf_agents.trajectories import time_step as ts
import tensorflow as tf


class RepeatQEnvironment(PyEnvironment):
    class State:

        def __init__(self, base_question, predicted_tokens, sequence_index):
            super(RepeatQEnvironment.State, self).__init__()
            self.base_question = base_question
            self.predicted_tokens = predicted_tokens
            self.sequence_index = sequence_index

    def __init__(self, reverse_vocabulary, max_sequence_length, eos_token, pad_token):
        super(RepeatQEnvironment, self).__init__()

        self.voc = reverse_vocabulary
        self.voc_size = len(reverse_vocabulary)
        self.max_sequence_length = max_sequence_length
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.weighted_reward_functions = self._build_reward_functions()

        # Choose a token from the vocabulary
        self._action_spec = (
            BoundedTensorSpec(
                shape=(1,), dtype=tf.int32, minimum=0, maximum=self.voc_size,
                name="predict_token_spec"
            ),
            BoundedTensorSpec(
                shape=(None,), dtype=tf.int32, minimum=0, maximum=self.voc_size
            )
        )
        # Observation is the last generated token
        self._obs_spec = BoundedTensorSpec(
            shape=(), dtype=tf.int32, minimum=0, maximum=self.voc_size,
            name="produced_tokens_spec"
        )
        self._state = self._initial_state()
        self._episode_ended = False

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return 1

    def get_state(self):
        return self._state

    def set_state(self, state):
        if state.predicted_tokens is None:
            state.predicted_tokens = self._state.predicted_tokens
        self._state = state

    def compute_reward(self, predicted_tokens, base_question):
        predicted_tokens = self._make_sequence(predicted_tokens.numpy())
        base_question = self._make_sequence(base_question.numpy())

        if len(predicted_tokens) == 0:
            return 0.0
        reward = sum(weight * reward_fn(predicted_tokens, base_question)
                     for weight, reward_fn in self.weighted_reward_functions)
        tf.print("Reward: ", reward)
        return reward

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def get_info(self):
        pass

    def _step(self, action):
        predicted_token = action[0][0]
        if self._episode_ended:
            return self._reset()
        current_state = self.get_state()
        # If end of sentence token or pad token was produced or if max length has been reached, terminate the sequences
        if (predicted_token in (self.eos_token, self.pad_token)) or \
                (current_state.sequence_index + 1 == self.max_sequence_length):
            self._episode_ended = True

        if predicted_token != self.pad_token:
            current_state.predicted_tokens[current_state.sequence_index] = predicted_token

        current_state.sequence_index += 1
        if self._episode_ended:
            reward = self.compute_reward(current_state, action[1])
            return ts.termination(np.array([predicted_token], dtype=np.int32), [reward])
        else:
            # Penalize repeating oneself
            if tf.equal(
                current_state.predicted_tokens[current_state.sequence_index-2],
                current_state.predicted_tokens[current_state.sequence_index-1]
            ):
                reward = -0.1
            else:
                reward = 0.0
            return ts.transition(
                observation=np.array([predicted_token], dtype=np.int32),
                reward=[reward],  # Length penalty
                discount=[1.0]
            )

    def _reset(self):
        self.set_state(self._initial_state())
        self._episode_ended = False
        return ts.restart(np.array([], dtype=np.int32), batch_size=1)

    def _build_reward_functions(self):
        def _bleu(n):
            def _bleu_n(predicted_tokens, base_question):
                weights = [1.0/n for _ in range(n)]
                score = bleu.sentence_bleu([base_question], predicted_tokens, weights=weights)
                #tf.print("BLEU-", n, " ", score)
                return score
            return _bleu_n

        def _length_penalty(predicted_tokens, base_question):
            penalty = -float(len(predicted_tokens))/len(base_question)
            return penalty

        def _repetitiveness_penalty(predicted_tokens, _):
            penalty = 0.0
            for i in range(1, len(predicted_tokens)):
                if predicted_tokens[i] == predicted_tokens[i-1]:
                    penalty += 0.01
            return -penalty
        return {
            #(0.25, _bleu(1)),
            (0.33, _bleu(2)),
            (0.33, _bleu(3)),
            (0.33, _bleu(4)),
        }

    def _make_sequence(self, tokens: Tensor):
        no_pad = [str(t) for t in tokens if t != self.pad_token]
        return no_pad

    def _initial_state(self):
        return RepeatQEnvironment.State(
            base_question=None, predicted_tokens=[self.pad_token for _ in range(self.max_sequence_length)],
            sequence_index=0
        )
