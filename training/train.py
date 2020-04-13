import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig
from data_utils.embeddings import Embedder
from data_utils.parse import read_bert_config
from models.bert import Bert
from training.trainer import Trainer
from training.utils.model_manager import ModelManager
from training.utils.squad_dataset import SquadDataset
from defs import PRETRAINED_MODELS_DIR

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'If training must be performed or only evaluating.')
flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.  Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('debug', False, 'If to run in debug mode or not.')
flags.DEFINE_string('pretrained_model_name', 'bert_mini_uncased', 'Name of the pre-trained BERT model.')
flags.DEFINE_boolean('load_model', False, 'If the model should be trained or loaded from memory for evaluation')
flags.DEFINE_string('loaded_model_name', None, 'Name of the model to load.')
flags.DEFINE_boolean('save_model', True, 'If the model is to be saved or not (after training).')
flags.DEFINE_string('saved_model_name', None, 'Name of the model to save.')
flags.DEFINE_integer('limit_train_data', -1, 'Number of rows to take from the training set.')
flags.DEFINE_integer('limit_dev_data', -1, 'Number of rows to take from the dev set.')

EPOCHS = FLAGS.nb_epochs
debug = FLAGS.debug
training = FLAGS.train

for flag, flag_val in FLAGS.flag_values_dict().items():
    print(f"{flag}: {flag_val}")

if FLAGS.pretrained_model_name == "bert_base_uncased":
    pretrained_model_name = "bert-base-uncased"
    vocab_name = pretrained_model_name
    base_model = TFBertModel.from_pretrained(pretrained_model_name)
    bert_config = base_model.config
elif FLAGS.pretrained_model_name == "bert_mini_uncased":
    pretrained_model_name = "uncased_L-4_H-256_A-4"
    vocab_name = f"{PRETRAINED_MODELS_DIR}/{pretrained_model_name}/"
    bert_config = read_bert_config(pretrained_model_name)
    base_model = TFBertModel(bert_config)
    base_model = ModelManager.load_pretrained_model(base_model, f"{pretrained_model_name}/bert_model.ckpt.index")
else:
    raise NotImplementedError()

tokenizer = BertTokenizer.from_pretrained(vocab_name)
embedder = Embedder(pretrained_model_name, tokenizer)
model = Bert(embedder, model=base_model,  hidden_state_size=bert_config.hidden_size)

if FLAGS.load_model:
    if FLAGS.loaded_model_name is None:
        raise ValueError("loaded_model_name was None while load_model was requested")
    model = ModelManager.load_model(model, FLAGS.loaded_model_name)

ds = SquadDataset(max_sequence_length=model.max_sequence_length,
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


trainer = Trainer(model)
step = tf.Variable(0, dtype=tf.int32, name='step')

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    trainer.train_loss.reset_states()
    # train_accuracy.reset_states()
    trainer.test_loss.reset_states()
    # test_accuracy.reset_states()

    if training:
        for features, labels in train_ds:
            prev_time = tf.timestamp()
            step.assign(0)
            trainer.train_step(features, labels, step)
            tf.print("Step completed in ", (tf.timestamp() - prev_time), " seconds")

    for test_features, test_labels in dev_ds:
        trainer.test_step(test_features, tf.squeeze(test_labels))

    #  logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    #  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    template = 'Epoch {}, Train Loss: {}, Mean Accuracy: {}, Test loss: {}'
    print(template.format(epoch + 1,
                          trainer.train_loss.result(),
                          trainer.train_accuracy.result() * 100,
                          trainer.test_loss.result() * 100,
                          ))

    if FLAGS.save_model and training:
        print("Saving model...")
        ModelManager.save_model(model, FLAGS.saved_model_name)


