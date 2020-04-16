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
                 test_accuracy=rouge_n,
                 print_predictions=False):
        super(Trainer, self).__init__()
        self.model = model
        self.embedder = model.embedder
        self.train_loss_object = train_loss_object
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.print_predictions = print_predictions

        model_gradient_dir = f"{GRADIENT_DIR}/{model_name}"
        if os.path.isdir(model_gradient_dir):
            shutil.rmtree(model_gradient_dir)
        self.train_summary_writer = tf.summary.create_file_writer(f"{model_gradient_dir}/train")
        self.test_summary_writer = tf.summary.create_file_writer(f"{model_gradient_dir}/test")

    @tf.function
    def train_step(self,
                   total_loss,
                   nb_losses,
                   paragraph_tokens_batches,
                   target_question_tokens_batches,
                   step,
                   global_step: tf.Variable):
        with tf.GradientTape() as tape:

            def compute_loss(i, sequence_loss, nb_generated_tokens, answer_tokens, question_tokens):
                # Only computes the loss for non-padding tokens (that is, tokens that are actually part of the question)
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

                # Accuracy
                self.train_accuracy(padding_free_outputs, padding_free_predictions)

                has_non_padding_outputs = tf.greater(tf.size(padding_free_outputs), tf.constant(0))
                if self.print_predictions:
                    tf.cond(has_non_padding_outputs, lambda: tf.print(tf.map_fn(lambda preds: tf.py_function(
                        self.embedder.tokenizer.convert_ids_to_tokens,
                        inp=[preds],
                        Tout=tf.string
                    ), padding_free_predictions, dtype=tf.string), "\n"), lambda: True)

                # Only records loss for non padding outputs
                sequence_loss = tf.add(sequence_loss, tf.cond(has_non_padding_outputs, lambda: loss, lambda: 0.0))
                nb_generated_tokens = tf.add(nb_generated_tokens, tf.cond(has_non_padding_outputs,
                                                                          lambda: 1.0,
                                                                          lambda: 0.0))
                return tf.add(i, 1), sequence_loss, nb_generated_tokens, new_input_tokens, question_tokens

            initial_size = paragraph_tokens_batches.shape[1]

            def cond(i, *_):
                # Predicts a token for each token in the target, limiting the paragraph + generated question length to
                # the max allowed length for this model
                return tf.logical_and(
                    tf.less(i, target_question_tokens_batches.shape[1]),
                    tf.less(tf.add(initial_size, i), self.model.max_sequence_length)
                )

            nb_iterations, total_loss, nb_losses, *_ = tf.while_loop(
                cond,
                compute_loss,
                loop_vars=[step, total_loss, nb_losses, paragraph_tokens_batches, target_question_tokens_batches],
                shape_invariants=[
                    step.shape,
                    total_loss.shape,
                    nb_losses.shape,
                    tf.TensorShape([paragraph_tokens_batches.shape[0], None]),
                    target_question_tokens_batches.shape]
            )

            global_step.assign(global_step + 1)
            sentence_loss = total_loss / nb_losses
            gradients = tape.gradient(sentence_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            with self.train_summary_writer.as_default():
                tf.print("Sentence loss: ", sentence_loss)
                tf.summary.scalar('loss', sentence_loss, step=global_step)

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
