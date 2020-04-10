from datetime import datetime

import tensorflow as tf
from data_utils.embeddings import Embedder
from data_utils.parse import read_square_dataset
from data_utils.pre_processing import pad_data
from defs import SQUAD_DEV, SQUAD_TRAIN
from models.bert import Bert

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.  Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('debug', False, 'If to run in debug mode or not.')

EPOCHS = FLAGS.nb_epochs
debug = FLAGS.debug

if debug:
    data = read_square_dataset(SQUAD_DEV, limit=1)
else:
    data = read_square_dataset(SQUAD_TRAIN)
embedder = Embedder()
x_train, y_train = embedder.generate_bert_hlsqg_dataset(data)
padding_value = embedder.tokenizer.pad_token_id
x_train = pad_data(x_train, padding_value)
y_train = pad_data(y_train, padding_value)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(buffer_size=256, reshuffle_each_iteration=True).repeat(EPOCHS).batch(FLAGS.batch_size)

model = Bert()
loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')


# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
def print_outputs(word_distributions, correct_tokens, nb_epochs):
    if tf.executing_eagerly():
        for pred_distribution, correct_token in zip(word_distributions, correct_tokens):
            correct_word = embedder.vocab_lookup([correct_token])
            predicted_word = embedder.vocab_lookup([tf.argmax(tf.squeeze(pred_distribution), axis=0)])
            print(f"Pred/Correct: {predicted_word} - {correct_word}")
    else:
        for pred_distribution, correct_token in zip(tf.unstack(word_distributions, num=nb_epochs),
                                                    tf.unstack(correct_tokens, num=nb_epochs)):
            correct_word = tf.py_function(
                embedder.vocab_lookup,
                inp=[tf.expand_dims(correct_token, axis=0)],
                Tout=tf.string
            )
            predicted_word = tf.py_function(
                embedder.vocab_lookup,
                inp=[tf.expand_dims(tf.argmax(tf.squeeze(pred_distribution), axis=0), axis=0)],
                Tout=tf.string
            )
            tf.print("Pred/Correct: ", predicted_word, " - ", correct_word)


@tf.function
def train_step(paragraph_tokens_batches, target_question_tokens_batches, step):

    def compute_loss(i, answer_tokens, question_tokens):
        # Only computes the loss for non-padding tokens (that is, tokens that are actually part of the question)
        with tf.GradientTape() as tape:
            word_distributions, correct_outputs, new_input_tokens = model.step(answer_tokens, question_tokens, i)
            non_padding_indices = tf.where(tf.not_equal(correct_outputs,
                                                        tf.fill(
                                                            correct_outputs.shape,
                                                            value=tf.constant(embedder.tokenizer.pad_token_id,
                                                                              tf.int32)
                                                        )))
            non_padding_distributions = tf.gather(word_distributions, non_padding_indices)
            non_padding_predictions = tf.math.softmax(non_padding_distributions, axis=1)
            non_padding_outputs = tf.gather(correct_outputs, non_padding_indices)
            loss = loss_object(non_padding_outputs, non_padding_distributions)

        def compute_and_apply_gradients():
            tf.print("Computing gradient - Token number ", i)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(non_padding_outputs, non_padding_predictions)

        # Only backpropagates for non padding outputs
        tf.cond(
            tf.greater(non_padding_outputs.shape[1], tf.constant(0)),
            compute_and_apply_gradients,
            lambda: None
        )
        return tf.add(i, 1), new_input_tokens, question_tokens

    def cond(i, *_):
        return tf.less(i, target_question_tokens_batches.shape[1])

    tf.while_loop(
        cond,
        compute_loss,
        loop_vars=[step, paragraph_tokens_batches, target_question_tokens_batches],
        shape_invariants=[step.shape,
                          tf.TensorShape([paragraph_tokens_batches.shape[0], None]),
                          target_question_tokens_batches.shape]
    )


#@tf.function
def test_step(paragraph_tokens, target_question_tokens):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(paragraph_tokens, training=False)
    generated_question = tf.py_function(
        embedder.vocab_lookup,
        inp=[tf.squeeze(predictions)],
        Tout=tf.string
    )
    tf.print(generated_question)
    t_loss = loss_object(target_question_tokens, predictions)

    test_loss(t_loss)
    #
    # test_accuracy(labels, predictions)


step = tf.Variable(0, dtype=tf.int32, name='step')

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    # train_accuracy.reset_states()
    test_loss.reset_states()
    # test_accuracy.reset_states()

    # for features, labels in train_ds:
    #     prev_time = tf.timestamp()
    #     step.assign(0)
    #     train_step(features, labels, step)
    #     tf.print("Step completed in ", (tf.timestamp() - prev_time), " seconds")

    test_ds = train_ds
    for test_features, test_labels in test_ds:
        test_step(test_features, test_labels)

    #  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    template = 'Epoch {}, Loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          #test_loss.result(),
                          # test_accuracy.result() * 100
                          ))
