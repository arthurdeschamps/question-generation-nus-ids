import os
import shutil

import tensorflow as tf
from defs import GRADIENT_DIR
from training.utils.rouge_score import rouge_n


class Trainer:
    """
    Use to train a model.
    """

    def __init__(self, model, model_name,
                 train_loss_object=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer=tf.optimizers.Adam(),
                 train_loss=tf.keras.metrics.Mean(name='train_loss'),
                 train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
                 test_loss=tf.keras.metrics.Mean(name='test_loss'),
                 test_accuracy=rouge_n):
        super(Trainer, self).__init__()
        self.model = model
        self.embedder = model.embedder
        self.train_loss_object = train_loss_object
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

        model_gradient_dir = f"{GRADIENT_DIR}/{model_name}"
        if os.path.isdir(model_gradient_dir):
            shutil.rmtree(model_gradient_dir)
        self.train_summary_writer = tf.summary.create_file_writer(f"{model_gradient_dir}/train")
        self.test_summary_writer = tf.summary.create_file_writer(f"{model_gradient_dir}/test")

    @tf.function
    def train_step(self, paragraph_tokens_batches, target_question_tokens_batches, step, global_step: tf.Variable):

        def compute_loss(i, answer_tokens, question_tokens):
            # Only computes the loss for non-padding tokens (that is, tokens that are actually part of the question)
            with tf.GradientTape() as tape:
                word_logits, correct_outputs, new_input_tokens = self.model.step(answer_tokens, question_tokens, i)
                padding_free_indices = tf.where(tf.not_equal(correct_outputs,
                                                             tf.fill(
                                                                 correct_outputs.shape,
                                                                 value=tf.constant(self.embedder.tokenizer.pad_token_id,
                                                                                   tf.int32)
                                                             )))
                padding_free_logits = tf.gather(word_logits, padding_free_indices)
                padding_free_predictions = tf.math.argmax(tf.math.softmax(padding_free_logits, axis=-1), axis=-1)
                padding_free_outputs = tf.gather(correct_outputs, padding_free_indices)

                # Computes the perplexity
                loss = self.train_loss_object(padding_free_outputs, padding_free_logits)

            def compute_and_apply_gradients():
                global_step.assign(global_step + 1)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self.train_loss(loss)
                self.train_accuracy(padding_free_outputs, padding_free_predictions)
                tf.print(tf.map_fn(lambda preds: tf.py_function(
                    self.embedder.tokenizer.convert_ids_to_tokens,
                    inp=[preds],
                    Tout=tf.string
                ), padding_free_predictions, dtype=tf.string))

            # Only backpropagates for non padding outputs
            tf.cond(
                tf.greater(tf.size(padding_free_outputs), tf.constant(0)),
                compute_and_apply_gradients,
                lambda: None
            )
            return tf.add(i, 1), new_input_tokens, question_tokens

        initial_size = paragraph_tokens_batches.shape[1]

        def cond(i, *_):
            # Predicts a token for each token in the target, limiting the paragraph + generated question length to the
            # max allowed length for this model
            return tf.logical_and(
                tf.less(i, target_question_tokens_batches.shape[1]),
                tf.less(tf.add(initial_size, i), self.model.max_sequence_length)
            )

        tf.while_loop(
            cond,
            compute_loss,
            loop_vars=[step, paragraph_tokens_batches, target_question_tokens_batches],
            shape_invariants=[step.shape,
                              tf.TensorShape([paragraph_tokens_batches.shape[0], None]),
                              target_question_tokens_batches.shape]
        )
        with self.train_summary_writer.as_default():
            tf.print("Sentence loss: ", self.train_loss.result())
            tf.summary.scalar('loss', self.train_loss.result(), step=global_step)
            self.train_loss.reset_states()

    @tf.function
    def test_step(self, paragraph_tokens, target_question_tokens, global_step, log_metrics):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(paragraph_tokens, training=False)
        t_question_no_padding = tf.reshape(tf.gather(target_question_tokens, tf.where(
            tf.not_equal(target_question_tokens, self.embedder.padding_token))), shape=(-1,))

        def compute_accuracy(target_question, generated_question):
            target_question = self.embedder.vocab_lookup(target_question.numpy())
            generated_question = self.embedder.vocab_lookup(generated_question.numpy()).replace('?', '')
            tf.print(generated_question)
            acc = self.test_accuracy([generated_question], [target_question])
            if log_metrics:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('dev_accuracy', acc, step=global_step)
            return acc

        self.test_loss(tf.py_function(
            compute_accuracy,
            inp=[t_question_no_padding, predictions],
            Tout=tf.float32
        ))
