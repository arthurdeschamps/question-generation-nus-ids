import os
import shutil

import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu

from defs import GRADIENT_DIR
from evaluating.rouge_score import rouge_n


class Trainer:
    """
    Use to train a model.
    """

    def __init__(self, model, model_name,
                 train_loss_object=tf.losses.SparseCategoricalCrossentropy(
                     from_logits=True,
                     reduction=tf.losses.Reduction.NONE
                 ),
                 optimizer=tf.optimizers.Adam(),
                 train_loss=tf.keras.metrics.Mean(name='train_loss'),
                 train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
                 test_loss=tf.keras.metrics.Mean(name='test_loss'),
                 test_accuracy=sentence_bleu,
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
                   paragraph_tokens_batches,
                   target_question_tokens_batches,
                   global_step: tf.Variable):
        pass

    @tf.function
    def test_step(self, paragraph_tokens, target_question_tokens, global_step, log_metrics):
        pass
