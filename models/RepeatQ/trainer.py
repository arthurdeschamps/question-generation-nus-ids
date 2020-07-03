from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from tf_agents.agents import ReinforceAgent
from tqdm import tqdm

from defs import REPEAT_Q_MODEL_DIR


class RepeatQTrainer:

    def __init__(self, model, training_data, reverse_voc, optimizer=tf.keras.optimizers.Adam()):
        super(RepeatQTrainer, self).__init__()
        self.model = model
        self.training_data = training_data
        self.reverse_voc = reverse_voc
        self.optimizer = optimizer

    def train_reinforce(self):
        time_step_spec =
        agent = ReinforceAgent(
            time_step_spec=time_step_spec, action_spec=action_sepc, actor_network=self.model, optimizer=self.optimizer
        )

    def train(self, nb_epochs=15):
        for epoch in range(nb_epochs):
            print(f"Starting Epoch {epoch + 1}.")
            for features, label in tqdm(self.training_data):
                features["generated_question_length"] = tf.reduce_sum(tf.cast(tf.not_equal(label[0], 0), dtype=tf.int32))#label.shape[1]
                features["target"] = label
                loss, logits = self.train_step(features, label)
                it = self.optimizer.iterations.numpy()
                if it > 0 and it % 10 == 0:
                    print("Step: {}, Loss: {}".format(it, loss.numpy()))
                    preds = RepeatQTrainer.ids_to_words(tf.argmax(logits[0], axis=-1).numpy(), self.reverse_voc)
                    target = RepeatQTrainer.ids_to_words(label[0].numpy(), self.reverse_voc)
                    base = RepeatQTrainer.ids_to_words(features["base_question"][0].numpy(), self.reverse_voc)
                    print(preds)
                    print(target)
                    print(base)
                    print()

    # @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            logits = self.model(features, training=True)
            targets = labels[:, :tf.shape(logits)[1]]
            weights = tf.cast(tf.not_equal(targets, 0), dtype=tf.float32, name="loss_mask")
            loss = tfa.seq2seq.sequence_loss(
                logits=logits, targets=targets, weights=weights, sum_over_batch=True, sum_over_timesteps=True,
                average_across_batch=False, average_across_timesteps=False
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, logits

    @staticmethod
    def ids_to_words(ids, ids_to_words_voc):
        return " ".join(ids_to_words_voc[id_] for id_ in ids)

    def visualize_graph(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = REPEAT_Q_MODEL_DIR + ('/logs/func/%s' % stamp)
        writer = tf.summary.create_file_writer(logdir)
        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        tf.summary.trace_on(graph=True, profiler=True)
        for features, label in self.training_data[0]:
            features["generated_question_length"] = label.shape[1]
            self.train_step(features, label)
        # Call only one tf.function when tracing.
        with writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)
