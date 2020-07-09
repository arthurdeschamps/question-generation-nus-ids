from logging import info
import tensorflow_addons as tfa
import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory, time_step
from tf_agents.utils import common
from tqdm import tqdm
from defs import PAD_TOKEN
from models.RepeatQ.model import RepeatQ
from models.RepeatQ.model_config import ModelConfiguration
from models.RepeatQ.rl.environment import RepeatQEnvironment


class RepeatQTrainer:

    def __init__(self,
                 model_config: ModelConfiguration,
                 training_data,
                 vocabulary,
                 optimizer=tf.keras.optimizers.Adam()):
        super(RepeatQTrainer, self).__init__()
        self.training_data = training_data
        self.vocabulary = vocabulary
        self.reverse_voc = {v: k for k, v in vocabulary.items()}
        self.optimizer = optimizer
        self.config = model_config
        self.model = None

    def _setup_reinforce(self):
        # utils.validate_py_environment(environment, episodes=5)
        environment = self._build_environment()
        environment = tf_py_environment.TFPyEnvironment(environment)
        actor = self._build_actor(environment, list(self.training_data.take(1))[0][0])
        agent = reinforce_agent.ReinforceAgent(
            time_step_spec=environment.time_step_spec(),
            action_spec=environment.action_spec(),
            actor_network=actor,
            optimizer=self.optimizer,
            name="repeat_q_reinforce_agent"
        )
        agent.initialize()
        agent.train = common.function(agent.train)
        agent.train_step_counter.assign(0)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=self.config.batch_size,
            max_length=32)

        return environment, agent, actor, replay_buffer

    def collect_episode(self, environment, policy, num_episodes, replay_buffer):
        episode_counter = 0
        environment.reset()
        policy._automatic_state_reset = False
        n_state = None
        while episode_counter < num_episodes:
            time_step = environment.current_time_step()
            if time_step.is_first():
                features, label = list(self.training_data.take(1))[0]
                environment.pyenv.set_state(RepeatQEnvironment.State(
                    base_question=features["base_question"],
                    sequence_index=0,
                    predicted_tokens=None
                ))
                # Creates initial 0-state
                n_state = policy.get_initial_state(1)
                # Fill the base question and facts
                n_state = RepeatQ.NetworkState(
                    base_question=features["base_question"],
                    facts=features["facts"],
                    facts_encodings=n_state.facts_encodings,
                    previous_token_embedding=n_state.previous_token_embedding,
                    base_question_embeddings=n_state.base_question_embeddings,
                    decoder_states=n_state.decoder_states
                )
            policy_step = policy.action(time_step, n_state)
            n_state = policy_step.state
            next_time_step = environment.step(policy_step.action)
            traj = trajectory.from_transition(time_step, policy_step, next_time_step)

            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj)

            if traj.is_boundary():
                episode_counter += 1

    @staticmethod
    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def train(self, nb_epochs=25):
        self.model = RepeatQ(self.vocabulary, self.config)
        env = self._build_environment()
        for epoch in range(nb_epochs):
            phase = "supervised" if epoch <= 25 else "unsupervised"
            tf.print(f"Starting Epoch {epoch + 1}.")
            for features, label in tqdm(self.training_data):
                features["target"] = features["base_question"]
                features["facts"] = features["facts"][:, :3]
                self.episode(features, label, env, phase=phase)

    @tf.function
    def episode(self, features, labels, environment, phase="supervised"):
        batch_size = features["facts"].shape[0]

        def debug_output(predictions):
            print("Predicted: ", " ".join([self.reverse_voc[int(t)] for t in predictions if int(t) != 0]))
            return 0

        with tf.GradientTape() as tape:
            target = features["target"]
            actions, logits = self.model.get_actions([
                features["base_question"], features["facts"]
            ], target=target, training=True, phase=phase)
            tf.py_function(debug_output, inp=(actions[0],), Tout=tf.int32)

            if phase == "unsupervised":
                mean_gradient = self._policy_gradient(actions, logits, features, environment, batch_size)
            elif phase == "supervised":
                mask = tf.cast(tf.not_equal(target, 0), dtype=tf.float32, name="seq_loss_mask")
                mean_gradient = tfa.seq2seq.sequence_loss(
                    logits=logits, targets=target, weights=mask, sum_over_batch=True, sum_over_timesteps=True,
                    average_across_timesteps=False, average_across_batch=False
                )
        tf.print("Mean gradient: ", mean_gradient)
        gradients = tape.gradient(mean_gradient, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return mean_gradient

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
            policy_gradient = self._get_policy_gradient(episode_rewards, episode_log_probs)
            gradients = gradients.write(i, policy_gradient)
        policy_gradient_mean = tf.reduce_mean(gradients.stack())
        tf.print("Policy gradient", policy_gradient_mean, "\n")
        return policy_gradient_mean

    def _get_policy_gradient(self, rewards, log_probs):
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

    def _build_environment(self):
        environment = RepeatQEnvironment(
            eos_token=self.vocabulary["?"],  # TODO re-generate dataset to have the EOS token in vocabulary
            pad_token=self.vocabulary[PAD_TOKEN],
            max_sequence_length=self.config.max_generated_question_length,
            reverse_vocabulary=self.reverse_voc
        )
        return environment

    def _build_actor(self, environment, ds_example):
        facts = ds_example["facts"]
        (batch_size, facts_per_example, fact_length) = facts.get_shape()
        base_question_length = ds_example["base_question"].get_shape()[1]
        model = RepeatQ(
            action_spec=environment.action_spec(),
            observation_spec=environment.observation_spec(),
            config=self.config,
            voc_word_to_id=self.vocabulary,
            fact_length=fact_length,
            facts_per_example=facts_per_example,
            base_question_length=base_question_length
        )
        # Builds the model's shapes
        random_input = tensor_spec.sample_spec_nest(model.input_tensor_spec, outer_dims=[1])
        random_state = tensor_spec.sample_spec_nest(model.state_spec, outer_dims=[1])
        step_type = tf.zeros([time_step.StepType.FIRST], dtype=tf.int32)
        model.__call__(
            random_input, step_type=step_type, network_state=random_state,
        )
        return model
