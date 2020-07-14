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
                 optimizer=tf.keras.optimizers.Adam()):
        super(RepeatQTrainer, self).__init__()
        self.training_data = training_data
        self.dev_data = dev_data
        self.vocabulary = vocabulary
        self.reverse_voc = {v: k for k, v in vocabulary.items()}
        self.optimizer = optimizer
        self.config = model_config
        self.model = model

    def train(self, nb_epochs=15):
        env = self._build_environment()

        model_save_dir = RepeatQTrainer.prepare_model_save_dir()

        for epoch in range(nb_epochs):
            phase = self.phase(epoch)
            tf.print(f"Starting Epoch {epoch + 1}.")
            for features, label in tqdm(self.training_data):
                features["target"] = features["base_question"]
                features["facts"] = features["facts"][:, :3]
                self.train_step(features, label, env, phase=phase)

            dev_score = self.dev_step(phase, env)
            checkpoint_filename = f"{model_save_dir}/{phase}_epoch_{epoch+1}_bleu_{'%.2f' % dev_score}"
            self.model.save_weights(filepath=checkpoint_filename)

    @tf.function
    def train_step(self, features, labels, environment, phase):
        batch_size = features["facts"].shape[0]

        def debug_output(predictions, base_question):
            tf.print("Predicted: ", " ".join([self.reverse_voc[int(t)] for t in predictions if int(t) != 0]))
            tf.print("Base question: ", " ".join([self.reverse_voc[int(t)] for t in base_question if int(t) != 0]))
            return 0

        with tf.GradientTape() as tape:
            target = labels
            actions, logits = self.model.get_actions([
                features["base_question"], features["facts"]
            ], target=target, training=True, phase=phase)
            tf.py_function(debug_output, inp=(actions[0], features["base_question"][0]), Tout=tf.int32)

            if phase == RepeatQTrainer.reinforce_phase:
                mean_gradient = self._policy_gradient(actions, logits, features, environment, batch_size)
                tf.print("Mean gradient: ", mean_gradient)
            elif phase == RepeatQTrainer.supervised_phase:
                mask = tf.cast(tf.not_equal(target, 0), dtype=tf.float32, name="seq_loss_mask")
                mean_gradient = tfa.seq2seq.sequence_loss(
                    logits=logits, targets=target, weights=mask, sum_over_batch=True, sum_over_timesteps=True,
                    average_across_timesteps=False, average_across_batch=False
                )
        gradients = tape.gradient(mean_gradient, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return mean_gradient

    def dev_step(self, phase, env):
        print("Performing dev step...")
        predicted_questions = tf.TensorArray(size=self.config.dev_step_size, dtype=tf.int32, name="dev_predictions")
        labels = tf.TensorArray(size=self.config.dev_step_size, dtype=tf.int32, name="dev_labels")
        for i, (features, label) in tqdm(enumerate(self.dev_data.take(self.config.dev_step_size))):
            actions, logits = self.model.get_actions(
                [features["base_question"], features["facts"]], target=label, training=False, phase=phase
            )
            predicted_questions = predicted_questions.write(i, actions)
            labels = labels.write(i, label)
        predicted_questions = predicted_questions.stack()
        labels = labels.stack()
        s = tf.shape(predicted_questions)
        predicted_questions = tf.reshape(predicted_questions, shape=(s[0]*s[1], s[2]))
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

    def _policy_gradient(self, actions, logits, features, environment, batch_size):

        def compute_reward(predictions, base_question):
            return environment.compute_reward(predictions, base_question)

        gradients = tf.TensorArray(dtype=tf.float32, size=self.config.batch_size, name="grads")
        log_probs = tf.reduce_max(tf.math.softmax(logits, axis=-1), axis=-1)
        log_probs = tf.math.log(log_probs, name="log_probs")
        for i in tf.range(batch_size):
            episode_actions = actions[i]
            episode_log_probs = log_probs[i]
            episode_reward = tf.py_function(compute_reward, inp=(episode_actions, features["base_question"][i]),
                                            Tout=tf.float32)
            episode_rewards = tf.concat(
                (tf.zeros(shape=(tf.size(episode_actions) - 1,)), [episode_reward]), axis=0
            )
            policy_gradient = RepeatQTrainer._get_policy_gradient(episode_rewards, episode_log_probs)
            gradients = gradients.write(i, policy_gradient)
        policy_gradient_mean = tf.reduce_mean(gradients.stack())
        tf.print("Policy gradient", policy_gradient_mean, "\n")
        return policy_gradient_mean

    def _build_environment(self):
        environment = RepeatQEnvironment(
            eos_token=self.vocabulary["?"],  # TODO re-generate dataset to have the EOS token in vocabulary
            pad_token=self.vocabulary[PAD_TOKEN],
            max_sequence_length=self.config.max_generated_question_length,
            reverse_vocabulary=self.reverse_voc,
        )
        return environment

    @staticmethod
    def _get_policy_gradient(rewards, log_probs):
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
        policy_gradient = -log_probs * discounted_rewards
        policy_gradient = tf.reduce_sum(policy_gradient, axis=0)
        return policy_gradient

    @staticmethod
    def ids_to_words(ids, ids_to_words_voc):
        return " ".join(ids_to_words_voc[id_] for id_ in ids)

    @staticmethod
    def prepare_model_save_dir():
        model_dir = f"{TRAINED_MODELS_DIR}/repeat_q/"
        shutil.rmtree(model_dir)
        info(f"Creating model saving directory: {model_dir}.")
        Path(model_dir).mkdir(parents=False)
        return model_dir
