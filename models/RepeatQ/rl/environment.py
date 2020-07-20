from collections import namedtuple
from typing import List
import nltk.translate.bleu_score as bleu
from nltk.translate.meteor_score import meteor_score
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

    def __init__(self, vocabulary, reversed_vocabulary, max_sequence_length, eos_token, pad_token):
        super(RepeatQEnvironment, self).__init__()

        self.word_to_token_voc = vocabulary
        self.token_to_word_voc = reversed_vocabulary
        self.voc_size = len(vocabulary)
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

    def compute_reward(self, predicted_tokens, base_question, target_question):
        predicted_tokens = self.make_sequence(predicted_tokens.numpy())
        base_question = self.make_sequence(base_question.numpy())
        target_question = self.make_sequence(target_question.numpy())

        if len(predicted_tokens) == 0:
            return 1e-9

        def compute_reward(candidate):
            return sum(weight * reward_fn(candidate, target_question) for weight, reward_fn in
                       self.weighted_reward_functions)

        baseline_reward = compute_reward(base_question)
        reward = compute_reward(predicted_tokens)
        return reward
        return reward - baseline_reward

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
            reward = self.compute_reward(current_state, action[1], action[2])
            return ts.termination(np.array([predicted_token], dtype=np.int32), [reward])
        else:
            return ts.transition(
                observation=np.array([predicted_token], dtype=np.int32),
                reward=[0],
                discount=[1.0]
            )

    def _reset(self):
        self.set_state(self._initial_state())
        self._episode_ended = False
        return ts.restart(np.array([], dtype=np.int32), batch_size=1)

    def _build_reward_functions(self):
        def _bleu(n):
            def _bleu_n(candidate, target_question):
                weights = [1.0/n for _ in range(n)]
                score = bleu.sentence_bleu([target_question], candidate, weights=weights)
                return score
            return _bleu_n

        def _meteor(candidate, target_question):
            predicted_sentence = " ".join(self.token_to_word_voc[int(t)] for t in candidate)
            target = " ".join(self.token_to_word_voc[int(t)] for t in target_question)
            score = meteor_score([target], predicted_sentence)
            return score

        def _repetitiveness_penalty(candidate, _):
            unique_words = set(candidate)
            penalty = -(len(candidate) - len(unique_words))
            return penalty

        def _contains_question_mark(candidate, _):
            return 1.0 if candidate[-1] == str(self.word_to_token_voc["?"]) else 0.0

        return {
            # (5.0, _bleu(1)),
            # (10.0, _bleu(2)),
            # (50.0, _bleu(3)),
            # (100.0, _bleu(4)),
            (1.0, _contains_question_mark),
            #(0.5, _repetitiveness_penalty),
            (10.0, _meteor)
        }

    def make_sequence(self, tokens: Tensor):
        no_pad = [str(t) for t in tokens if t != self.pad_token]
        try:
            return no_pad[:no_pad.index(str(self.eos_token))+1]
        except ValueError:
            pass
        return no_pad

    def _initial_state(self):
        return RepeatQEnvironment.State(
            base_question=None, predicted_tokens=[self.pad_token for _ in range(self.max_sequence_length)],
            sequence_index=0
        )
