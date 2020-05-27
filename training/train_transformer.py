import datetime
import logging
import os
import shutil
import tensorflow as tf
from defs import GRADIENT_DIR
from transformers import BertTokenizer, TFBertModel, GPT2Tokenizer, TFGPT2LMHeadModel
from training.gpt_trainer import GPTTrainer
from training.utils.embeddings import Embedder
from data_processing.parse import read_bert_config
from models.transformer import Transformer
from training.trainer import Trainer
from training.utils.model_manager import ModelManager
from training.utils.hlsqg_dataset import HlsqgDataset
from defs import PRETRAINED_MODELS_DIR
from nltk.translate.bleu_score import corpus_bleu

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate for Adam.')
flags.DEFINE_boolean('train', True, 'If training must be performed or only evaluating.')
flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.  Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('debug', False, 'If to run in debug mode or not.')
flags.DEFINE_string('pretrained_model_name', 'bert_mini_uncased', 'Name of the pre-trained BERT model.')
flags.DEFINE_boolean('load_model', False, 'If the model should be trained or loaded from memory for evaluation')
flags.DEFINE_string('loaded_model_name', None, 'Name of the model to load.')
flags.DEFINE_boolean('save_model', True, 'If the model is to be saved or not (after training).')
flags.DEFINE_string('saved_model_name', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'Name of the model to save.')
flags.DEFINE_integer('limit_train_data', -1, 'Number of rows to take from the training set.')
flags.DEFINE_integer('limit_dev_data', -1, 'Number of rows to take from the dev set.')
flags.DEFINE_boolean('log_test_metrics', True, 'Whether to log metrics for tensorboard at test time.')
flags.DEFINE_boolean('print_predictions', True, 'Whether to print predictions at training time for each token.')
flags.DEFINE_boolean('print_dev_predictions', False,
                     'Whether to print predictions and BLEU score during the dev steps.')

EPOCHS = FLAGS.nb_epochs
debug = FLAGS.debug
training = FLAGS.train

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.print("Num GPUs Available: ", len(gpus))
if len(gpus) < 1:
    exit(-1)

for flag, flag_val in FLAGS.flag_values_dict().items():
    tf.print(f"{flag}: {flag_val}")

if FLAGS.pretrained_model_name == "bert_base_uncased":
    pretrained_model_name = "bert-base-uncased"
    vocab_name = pretrained_model_name
    base_model = TFBertModel.from_pretrained(pretrained_model_name)
    config = base_model.config
    tokenizer = BertTokenizer.from_pretrained(vocab_name)
    trainer = Trainer
elif FLAGS.pretrained_model_name == "bert_mini_uncased":
    pretrained_model_name = "uncased_L-4_H-256_A-4"
    vocab_name = f"{PRETRAINED_MODELS_DIR}/{pretrained_model_name}/"
    config = read_bert_config(pretrained_model_name)
    base_model = TFBertModel(config)
    base_model = ModelManager.load_pretrained_model(base_model, f"{pretrained_model_name}/bert_model.ckpt.index")
    tokenizer = BertTokenizer.from_pretrained(vocab_name)
    trainer = Trainer
elif FLAGS.pretrained_model_name == "gpt2":
    pretrained_model_name = "gpt2"
    vocab_name = "gpt2"
    base_model = TFGPT2LMHeadModel.from_pretrained(pretrained_model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(vocab_name)
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]"
    })
    base_model.config.pad_token_id = tokenizer.pad_token_id
    config = base_model.config
    trainer = GPTTrainer
else:
    raise NotImplementedError()

model_gradient_dir = f"{GRADIENT_DIR}/{FLAGS.saved_model_name}"
if os.path.isdir(model_gradient_dir):
    shutil.rmtree(model_gradient_dir)
train_summary_writer = tf.summary.create_file_writer(f"{model_gradient_dir}/train")
test_summary_writer = tf.summary.create_file_writer(f"{model_gradient_dir}/test")

embedder = Embedder(pretrained_model_name, tokenizer)
model = Transformer(embedder=embedder, model=base_model, hidden_state_size=config.hidden_size, max_sequence_length=150)

if FLAGS.load_model:
    if FLAGS.loaded_model_name is None:
        raise ValueError("loaded_model_name was None while load_model was requested")
    model = ModelManager.load_model(model, FLAGS.loaded_model_name)

ds = HlsqgDataset(max_sequence_length=model.max_sequence_length,
                  max_generated_question_length=model.max_generated_question_length,
                  embedder=embedder,
                  batch_size=FLAGS.batch_size,
                  nb_epochs=EPOCHS,
                  limit_dev_data=FLAGS.limit_dev_data,
                  limit_train_data=FLAGS.limit_train_data)
if training:
    train_ds = ds.get_train_set()
    dev_ds = ds.get_dev_set()
else:
    train_ds = None
    dev_ds = ds.get_dev_set()

if not debug:
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

trainer = trainer(
    model, print_predictions=FLAGS.print_predictions,
    optimizer=tf.optimizers.Adam(lr=FLAGS.learning_rate)
)
# This is how many times backprob has been performed
total_loss = tf.Variable(0.0, dtype=tf.float32, name='sequence_loss', trainable=False)
nb_losses = tf.Variable(0.0, dtype=tf.float32, name='nb_computed_losses', trainable=False)

i = 0
dev_score = tf.constant(0.0, dtype=tf.float32, name="dev_accuracy")
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    trainer.train_accuracy.reset_states()

    def dev_step(dev_size=500):
        targets = []
        predictions = []
        for test_features, test_labels in dev_ds.take(dev_size):
            target_tokens, prediction_tokens = \
                trainer.test_step(test_features, tf.squeeze(test_labels), FLAGS.log_test_metrics)
            targets.append([tokenizer.decode(target_tokens).replace("?", " ?").split()])
            predictions.append(tokenizer.decode(prediction_tokens).replace("?", " ?").split())
            if FLAGS.print_predictions:
                print(f"Target: {targets[-1]}")
                print(f"Predictions: {predictions[-1]}\n")

        acc = corpus_bleu(targets, predictions) * 100
        return acc


    if training:
        tf.print("Epoch ", epoch + 1, " started.")

        for features, labels in train_ds:
            prev_time = tf.timestamp()
            trainer.train_loss(trainer.train_step(features, labels))
            if FLAGS.print_predictions:
                tf.print("Step", i, " completed in ", (tf.timestamp() - prev_time), " seconds")
            i += 1
            # Records the mean loss every 2000 sentences
            if i * FLAGS.batch_size % 2000 == 0:
                mean_loss = trainer.train_loss.result()
                tf.print("Trained on ", i * FLAGS.batch_size, " examples. Mean loss: ", mean_loss)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', mean_loss, step=i*FLAGS.batch_size)
                trainer.train_loss.reset_states()
            # Runs a dev step every 20000 sentences
            if i * FLAGS.batch_size % 20000 == 0:
                mean_accuracy = dev_step()
                tf.print("Mean accuracy: ", mean_accuracy)
                if mean_accuracy > dev_score:
                    dev_score = mean_accuracy
                    if FLAGS.save_model:
                        tf.print("Saving model...")
                        ModelManager.save_model(model, FLAGS.saved_model_name)

                if FLAGS.log_test_metrics:
                    with test_summary_writer.as_default():
                        tf.summary.scalar('dev_accuracy', mean_accuracy, step=i*FLAGS.batch_size)

        template = 'Epoch {}, Train Loss: {}, Mean Accuracy: {}'
        tf.print(template.format(
            epoch + 1, trainer.train_loss.result(), trainer.train_accuracy.result() * 100
        ))
    else:
        tf.print("Accuracy :", dev_step(-1))
tf.print("Training finished.")
