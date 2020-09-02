import os
import shutil
from logging import info
from pathlib import Path

import nltk
import tensorflow as tf
from tqdm import tqdm
from defs import PAD_TOKEN, TRAINED_MODELS_DIR
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

        def train(nb_epochs, data, dev_data, ds_type):
            tf.print("About to start training for ", nb_epochs, " epochs with ", ds_type, " dataset.")
            for epoch in range(nb_epochs):
                phase = self.phase(epoch)

                for features, label in tqdm(data):
                    self.train_step(features, label, env, phase=phase, epoch=epoch, ds_type=ds_type)

                dev_score = self.dev_step(phase, env, dev_data)
                tf.print("Score on dev set:", dev_score)
                if self.config.saving_model:
                    checkpoint_filename = f"{model_save_dir}/{phase}_{ds_type}_epoch_{epoch + 1}_bleu_{'%.2f' % dev_score}"
                    self.model.save_weights(filepath=checkpoint_filename)

        nb_epochs_config = {
            "synthetic": self.config.synth_supervised_epochs, "organic": self.config.org_supervised_epochs
        }

        if self.config.mixed_data:
            nb_epochs = sum(nb_epochs_config.values())
            train(nb_epochs=nb_epochs, data=self.training_data, dev_data=self.dev_data, ds_type="mixed")
        else:
            for ds_type in self.training_data.keys():
                nb_epochs = nb_epochs_config[ds_type]
                data = self.training_data[ds_type]
                dev_data = self.dev_data[ds_type]
                train(nb_epochs=nb_epochs, data=data, dev_data=dev_data, ds_type=ds_type)

    @tf.function
    def train_step(self, features, labels, environment, phase, epoch, ds_type):
        if phase == RepeatQTrainer.reinforce_phase:
            loss, tape = self._reinforce_step(features, labels, environment)
        else:
            loss, tape = self._supervised_step(features, labels)
        tf.print("Dataset:", ds_type, "/ Epoch:", epoch + 1, "/ Loss:", loss, "\n")
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def dev_step(self, phase, env, dev_data):
        tf.print("Performing dev step...")
        if self.config.dev_step_size is not None and self.config.dev_step_size > 0:
            dev_step_size = self.config.dev_step_size
            dev_data = dev_data.take(dev_step_size)
        else:
            dev_step_size = 1

        predicted_questions = tf.TensorArray(
            size=dev_step_size, dtype=tf.int32, name="dev_predictions", dynamic_size=True
        )
        labels = tf.TensorArray(
            size=dev_step_size, dtype=tf.int32, name="dev_labels", dynamic_size=True
        )
        for i, (features, label) in tqdm(enumerate(dev_data)):
            actions, _ = self.model.get_actions(features, target=label, training=False, phase=phase)
            paddings = (
                (0, 0), (0, tf.math.maximum(0, self.config.max_generated_question_length - tf.shape(actions)[1]))
            )
            actions_padded = tf.pad(
                actions,
                paddings=paddings,
                mode="CONSTANT"
            )
            predicted_questions = predicted_questions.write(i, actions_padded)
            labels = labels.write(i, label)
        predicted_questions = predicted_questions.stack()[:i+1]
        labels = labels.stack()[:i+1]
        s = tf.shape(predicted_questions)
        predicted_questions = tf.reshape(predicted_questions, shape=(s[0]*s[1], s[2]))
        s = tf.shape(labels)
        labels = tf.reshape(labels, shape=(s[0]*s[1], s[2]))

        def compute_bleu(refs, hyps):
            refs = [[env.make_sequence(ref)] for ref in refs.numpy()]
            hyps = [env.make_sequence(hyp) for hyp in hyps.numpy()]
            for i in range(len(hyps)):
                tf.print("Ref:", env._tokens_to_sentence(refs[i][0]))
                tf.print("Hyp:", env._tokens_to_sentence(hyps[i]), "\n")
            return 100*nltk.translate.bleu_score.corpus_bleu(refs, hyps)
        bleu_score = tf.py_function(compute_bleu, inp=[labels, predicted_questions], Tout=tf.float32)
        tf.print("BLEU-4:", bleu_score)
        return bleu_score

    def phase(self, current_epoch):
        if current_epoch < self.config.org_supervised_epochs + self.config.synth_supervised_epochs:
            return self.supervised_phase
        return self.reinforce_phase

    def _supervised_step(self, features, target, loss_fc=tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)):
        with tf.GradientTape() as tape:
            actions, pointer_softmax = self.model.get_actions(
                features, target=target, training=True, phase=self.supervised_phase
            )
            tf.py_function(self._debug_output, inp=(
                actions[0],
                features["base_question"][0],
                target[0],
                features["facts"][0]
            ), Tout=tf.int32)
            # Sets loss weights to 0 for padding tokens
            is_not_padding = tf.not_equal(target, 0)
            weights = tf.cast(is_not_padding, dtype=tf.float32, name="seq_loss_mask")
            # Sets loss weights to 0.25 for base question tokens (to hopefully encourage diversification)
            # weights = tf.where(
            #     tf.logical_and(features["from_base_question"], is_not_padding),
            #     0.25 * tf.ones_like(weights),
            #     weights
            # )
            # We need to slightly modify the targets so that the words that come from the base question are offset
            # by voc_size, as the logits are the concatenation of the vocabulary logits with the logits for the
            # base question (if words are being copied from there)
            copied = tf.not_equal(features["target_copy_indicator"], -1, name="copied_tokens")
            modified_targets = tf.where(copied, len(self.vocabulary) + features["target_copy_indicator"], target)
            num_classes = pointer_softmax.get_shape()[-1]
            flattened_targets = tf.reshape(modified_targets, (-1,))
            flattened_probs = tf.reshape(pointer_softmax, (-1, num_classes))
            flattened_weights = tf.reshape(weights, (-1,))
            losses = loss_fc(flattened_targets, flattened_probs)
            loss = tf.reduce_sum(flattened_weights * losses, axis=-1) / tf.reduce_sum(flattened_weights)

        return loss, tape

    def _reinforce_step(self, features, targets, environment):
        beams, beams_probs = self.model.beam_search(
            inputs=features, beam_search_size=self.config.training_beam_search_size, training=True, return_probs=True
        )
        # First collects episodes using non-differentiable beam search
        with tf.GradientTape() as tape:
            actions, logits = self.model.get_actions(features, beams, training=True, phase=RepeatQTrainer.supervised_phase)
            tf.py_function(
                self._debug_output,
                inp=(beams[0], features["base_question"][0], targets[0], beams[0]),
                Tout=tf.int32
            )
            loss = self._policy_gradient_loss(
                actions=actions, targets=targets, features=features, environment=environment, logits=logits
            )
        return loss, tape

    def _policy_gradient_loss(self, actions, targets, features, environment, logits=None, action_probs=None):
        if logits is None and action_probs is None:
            ValueError("Both logits and action_probs are None. Please provide either.")

        def compute_reward(predictions, base_question, target_question):
            return environment.compute_reward(predictions, base_question, target_question)

        rewards = tf.TensorArray(dtype=tf.float32, size=self.config.batch_size, name="rewards")
        if action_probs is None:
            probs = tf.math.softmax(logits, axis=-1)
            action_probs = tf.gather(probs, actions, batch_dims=2, axis=2)
        log_probs = tf.math.log(action_probs, name="log_probs")
        mask = tf.cast(tf.not_equal(actions, 0), dtype=tf.float32, name="reinforce_mask")

        for i in tf.range(self.config.batch_size):
            episode_actions = actions[i]
            episode_reward = tf.py_function(
                compute_reward,
                inp=(episode_actions, features["base_question"][i], targets[i]),
                Tout=tf.float32
            )
            # policy_gradient = RepeatQTrainer._get_policy_gradient(episode_rewards, log_probs[i], action_probs[i], mask[i])
            rewards = rewards.write(i, episode_reward)
        log_likelihood = tf.reduce_sum(mask * log_probs, axis=-1)
        rewards = rewards.stack()
        std = tf.math.reduce_std(rewards, axis=0) + 1e-9
        mean = tf.reduce_mean(rewards, axis=0)
        normalized_rewards = (rewards - mean) / std  # normalize discounted rewards
        policy_gradients = normalized_rewards * log_likelihood
        tf.print("Avg Reward:", mean)
        policy_gradient_mean = tf.reduce_mean(policy_gradients)
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

    def _debug_output(self, predictions, base_question, target, facts):
        def _debug_output(tokens, name):
            tf.print(name, ":", " ".join([self.reverse_voc[int(t)] for t in tokens if int(t) != 0]))
        _debug_output(predictions, "Predicted")
        _debug_output(target, "Rewritten")
        _debug_output(base_question, "Base question")
        for i, fact in enumerate(facts):
            _debug_output(fact, f"Fact {i}")
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
