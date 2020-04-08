from typing import List
import tensorflow as tf
from data_utils.embeddings import Embedder
from data_utils.parse import read_square_dataset
from defs import SQUAD_DEV
from models.bert import Bert
from utils.rouge_score import rouge_n

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.  Must divide evenly into the dataset sizes.')


EPOCHS = FLAGS.nb_epochs

data = read_square_dataset(SQUAD_DEV)
x_train, y_train = Embedder().generate_bert_hlsqg_dataset(data)
x_train = tf.ragged.constant(x_train, dtype=tf.int32)
y_train = tf.ragged.constant(y_train, dtype=tf.int32)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(FLAGS.batch_size).repeat(EPOCHS)
# dataset = dataset.batch(32).repeat()

model = Bert()
loss_object = rouge_n
optimizer = tf.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


#@tf.function
def train_step(paragraph_tokens_batches, target_question_tokens_batches):
    with tf.GradientTape() as tape:
        predictions = tf.stack(list(model(paragraph_tokens, training=True)
                                    for paragraph_tokens in paragraph_tokens_batches), axis=0)
        loss = loss_object(target_question_tokens_batches, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    # train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    # test_accuracy(labels, predictions)


for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    # train_accuracy.reset_states()
    test_loss.reset_states()
    # test_accuracy.reset_states()

    for features, labels in train_ds:
        train_step(features, labels)

    test_ds = train_ds
    for test_features, test_labels in test_ds:
        test_step(test_features, test_labels)

    #  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    template = 'Epoch {}, Loss: {}, Test Loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          #train_accuracy.result() * 100,
                          test_loss.result(),
                          #test_accuracy.result() * 100
                          ))
