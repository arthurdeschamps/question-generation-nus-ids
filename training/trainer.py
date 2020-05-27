import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu


class Trainer:
    """
    Use to train a model.
    """

    def __init__(self, model,
                 train_loss_object=tf.losses.SparseCategoricalCrossentropy(
                     from_logits=True,
                     reduction=tf.losses.Reduction.NONE
                 ),
                 optimizer=tf.optimizers.Adam(),
                 train_loss=tf.keras.metrics.Mean(name='train_loss'),
                 train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
                 print_predictions=False):
        super(Trainer, self).__init__()
        self.model = model
        self.embedder = model.embedder
        self.train_loss_object = train_loss_object
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.print_predictions = print_predictions

    @tf.function
    def train_step(self,
                   paragraph_tokens_batches,
                   target_question_tokens_batches):
        pass

    @tf.function
    def test_step(self, paragraph_tokens, target_question_tokens, log_metrics):
        pass
