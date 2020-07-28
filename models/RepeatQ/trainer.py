import os
import shutil
from logging import info
from pathlib import Path

import nltk
import tensorflow_addons as tfa
import tensorflow as tf
from tqdm import tqdm
from defs import PAD_TOKEN, TRAINED_MODELS_DIR
from models.RepeatQ.model import RepeatQ
from models.RepeatQ.model_config import ModelConfiguration
from models.RepeatQ.rl.environment import RepeatQEnvironment


class RepeatQTrainer:
    supervised_phase = "supervised"
    reinforce_phase = "unsupervised"

    def __init__(self,
                 model_config: ModelConfiguration,
                 model,
                 training_data,
                 dev_data,
                 vocabulary,
                 optimizer=None):
        super(RepeatQTrainer, self).__init__()
        self.training_data = training_data
        self.dev_data = dev_data
        self.vocabulary = vocabulary
        self.reverse_voc = {v: k for k, v in vocabulary.items()}
        self.config = model_config
        if optimizer is None:
            opti = tf.keras.optimizers.Adam
            if self.config.learning_rate is None:
                self.optimizer = opti()
            else:
                self.optimizer = opti(learning_rate=self.config.learning_rate)
        else:
            self.optimizer = optimizer
        self.model = model

    def train(self):
        env = self._build_environment()
        model_save_dir = RepeatQTrainer.prepare_model_save_dir()

        for epoch in range(self.config.epochs):
            phase = self.phase(epoch)

            tf.print(f"Starting Epoch {epoch + 1}.")
            for features, label in tqdm(self.training_data):
                self.train_step(features, label, env, phase=phase)

            dev_score = self.dev_step(phase, env)
            tf.print("Score on dev set: ", dev_score)
            if self.config.saving_model:
                checkpoint_filename = f"{model_save_dir}/{phase}_epoch_{epoch+1}_bleu_{'%.2f' % dev_score}"
                self.model.save_weights(filepath=checkpoint_filename)

    @tf.function
    def train_step(self, features, labels, environment, phase):
        if phase == RepeatQTrainer.reinforce_phase:
            loss, tape = self._reinforce_step(features, labels, environment)
        else:
            loss, tape = self._supervised_step(features, labels)
        tf.print("Loss: ", loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def dev_step(self, phase, env):
        tf.print("Performing dev step...")
        predicted_questions = tf.TensorArray(size=self.config.dev_step_size, dtype=tf.int32, name="dev_predictions")
        labels = tf.TensorArray(size=self.config.dev_step_size, dtype=tf.int32, name="dev_labels")
        for i, (features, label) in tqdm(enumerate(self.dev_data.take(self.config.dev_step_size))):
            actions, _ = self.model.get_actions(features, target=label, training=False, phase=phase)
            paddings = (
                (0, 0), (0, tf.math.maximum(0, self.config.max_generated_question_length - tf.shape(actions)[1]))
            )
            actions = tf.pad(
                actions,
                paddings=paddings,
                mode="CONSTANT"
            )
            predicted_questions = predicted_questions.write(i, actions)
            labels = labels.write(i, label)
        predicted_questions = predicted_questions.stack()
        labels = labels.stack()
        s = tf.shape(predicted_questions)
        predicted_questions = tf.reshape(predicted_questions, shape=(s[0]*s[1], s[2]))
        s = tf.shape(labels)
        labels = tf.reshape(labels, shape=(s[0]*s[1], s[2]))
        bleu_score = tf.py_function(
            lambda refs, hyps: 100*nltk.translate.bleu_score.corpus_bleu(
                [[env.make_sequence(ref)] for ref in refs.numpy()],
                [env.make_sequence(hyp) for hyp in hyps.numpy()]
            ),
            inp=[labels, predicted_questions],
            Tout=tf.float32
        )
        tf.print("BLEU-4: ", bleu_score)
        return bleu_score

    def phase(self, current_epoch):
        if self.config.restore_supervised_checkpoint:
            return self.reinforce_phase
        if current_epoch < self.config.supervised_epochs:
            return self.supervised_phase
        return self.reinforce_phase

    def _supervised_step(self, features, target):
        with tf.GradientTape() as tape:
            actions, logits = self.model.get_actions(
                features, target=target, training=True, phase=self.supervised_phase
            )
            tf.py_function(self._debug_output, inp=(actions[0], features["base_question"][0], target[0]), Tout=tf.int32)
            mask = tf.cast(tf.not_equal(target, 0), dtype=tf.float32, name="seq_loss_mask")
            loss = tfa.seq2seq.sequence_loss(
                logits=logits, targets=target, weights=mask, sum_over_batch=True, sum_over_timesteps=True,
                average_across_timesteps=False, average_across_batch=False
            )
        return loss, tape

    def _reinforce_step(self, features, targets, environment):
        nb_episodes = targets.get_shape()[0]
        # First collects episodes using non-differentiable beam search
        tf.print("Collecting ", nb_episodes, " episodes...")
        beams = self.model.beam_search(
            inputs=features, beam_search_size=self.config.training_beam_search_size
        )
        tf.print("Episodes collected.")

        # Uses predicted sequence as ground truth in "teacher forcing" phase
        with tf.GradientTape() as tape:
            actions, logits = self.model.get_actions(features, beams, training=True, phase=self.reinforce_phase)
            tf.py_function(self._debug_output, inp=(actions[0], features["base_question"][0], targets[0]), Tout=tf.int32)
            loss = self._policy_gradient_loss(
                actions=actions, targets=targets, features=features, environment=environment, logits=logits
            )
        return loss, tape

    def _policy_gradient_loss(self, actions, targets, features, environment, logits=None, action_probs=None):
        if logits is None and action_probs is None:
            ValueError("Both logits and action_probs are None. Please provide either.")

        def compute_reward(predictions, base_question, target_question):
            return environment.compute_reward(predictions, base_question, target_question)

        gradients = tf.TensorArray(dtype=tf.float32, size=self.config.batch_size, name="grads")
        if action_probs is None:
            probs = tf.math.softmax(logits, axis=-1)
            action_probs = tf.gather(probs, actions, batch_dims=2, axis=2)
        log_probs = tf.math.log(action_probs, name="log_probs")
        mask = tf.cast(tf.not_equal(actions, 0), dtype=tf.float32, name="reinforce_mask")
        average_reward = 0.0
        for i in tf.range(self.config.batch_size):
            episode_actions = actions[i]
            episode_reward = tf.py_function(
                compute_reward,
                inp=(episode_actions, features["base_question"][i], targets[i]),
                Tout=tf.float32
            )
            # episode_rewards = tf.concat(
            #     (tf.zeros(shape=(tf.size(episode_actions) - 1,)), [episode_reward]), axis=0
            # )
            average_reward += episode_reward / self.config.batch_size
            average_reward.set_shape(())
            # policy_gradient = RepeatQTrainer._get_policy_gradient(episode_rewards, log_probs[i], action_probs[i], mask[i])
            policy_gradient = - episode_reward * tf.reduce_sum(mask[i] * log_probs[i])
            gradients = gradients.write(i, policy_gradient)
        tf.print("Avg Reward: ", average_reward)
        gradients = gradients.stack()
        policy_gradient_mean = tf.reduce_mean(gradients)
        if tf.math.is_nan(policy_gradient_mean):
            return tf.float32.max
        return policy_gradient_mean

    def _build_environment(self):
        environment = RepeatQEnvironment(
            eos_token=self.vocabulary["?"],  # TODO re-generate dataset to have the EOS token in vocabulary
            pad_token=self.vocabulary[PAD_TOKEN],
            max_sequence_length=self.config.max_generated_question_length,
            vocabulary=self.vocabulary,
            reversed_vocabulary=self.reverse_voc
        )
        return environment

    def _debug_output(self, predictions, base_question, target):
        tf.print("Predicted: ", " ".join([self.reverse_voc[int(t)] for t in predictions if int(t) != 0]))
        tf.print("Rewritten : ", " ".join([self.reverse_voc[int(t)] for t in target if int(t) != 0]))
        tf.print("Base question: ", " ".join([self.reverse_voc[int(t)] for t in base_question if int(t) != 0]))
        return 0

    @staticmethod
    def _get_policy_gradient(rewards, log_probs, episode_probs, mask):
        discount_factor = 0.9
        discounted_rewards = tf.TensorArray(size=len(rewards), dtype=tf.float32)
        for t in range(len(rewards)):
            Gt = 0.0
            pw = 0.0
            for r in rewards[t:]:
                Gt = Gt + discount_factor ** pw * r
                pw = pw + 1
            discounted_rewards = discounted_rewards.write(t, Gt)
        discounted_rewards = discounted_rewards.stack()
        std = tf.math.reduce_std(discounted_rewards, axis=0) + 1e-9
        mean = tf.reduce_mean(discounted_rewards, axis=0)
        discounted_rewards = (discounted_rewards - mean) / std  # normalize discounted rewards
        if rewards[-1] > 0:
            factors = log_probs
        else:
            factors = tf.math.log(1.0 - episode_probs + 1e-8)
        policy_gradient = - mask * factors * discounted_rewards
        policy_gradient = tf.reduce_sum(mask * policy_gradient, axis=0)
        return policy_gradient

    @staticmethod
    def ids_to_words(ids, ids_to_words_voc):
        return " ".join(ids_to_words_voc[id_] for id_ in ids)

    @staticmethod
    def prepare_model_save_dir():
        model_dir = f"{TRAINED_MODELS_DIR}/repeat_q/"
        if not os.path.exists(model_dir):
            info(f"Creating model saving directory: {model_dir}.")
            Path(model_dir).mkdir(parents=False)
        return model_dir
