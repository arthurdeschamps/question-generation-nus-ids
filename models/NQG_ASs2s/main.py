import argparse
import pickle as pkl
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import params
import model as model
import mytools
# Enable logging for tf.estimator
tf.get_logger().setLevel("INFO")
FLAGS = None


def remove_eos(sentence, eos='<EOS>', pad='<PAD>'):
    if eos in sentence:
        return sentence[:sentence.index(eos)] + ['\n']
    elif pad in sentence:
        return sentence[:sentence.index(pad)] + ['\n']
    else:
        return sentence + ['\n']


def write_result(predict_results, dic_path):
    print('Load dic file...')
    with open(dic_path, mode='rb') as dic:
        dic_file = pkl.load(dic)
    reversed_dic = dict((y, x) for x, y in dic_file.items())

    print('Writing into file...')
    with open(FLAGS.pred_dir, 'w') as f:
        while True:
            try:
                output = next(predict_results)
                output = output['question'].tolist()
                if -1 in output:  # beam search
                    output = output[:output.index(-1)]
                indices = [reversed_dic[index] for index in output]
                sentence = remove_eos(indices)
                sentence = ' '.join(sentence)
                f.write(sentence)
                f.write(sentence)

            except StopIteration:
                break


def get_train_dataset(batch_size):
    # Load training data
    train_sentence = np.load(FLAGS.train_sentence)  # train_data
    train_question = np.load(FLAGS.train_question).astype(np.int)  # train_label
    train_answer = np.load(FLAGS.train_answer)
    train_sentence_length = np.load(FLAGS.train_sentence_length)
    train_question_length = np.load(FLAGS.train_question_length)
    train_answer_length = np.load(FLAGS.train_answer_length)

    return tf.data.Dataset.from_tensor_slices((
        {"s": train_sentence,
         'q': train_question,
         'a': train_answer,
         'len_s': train_sentence_length,
         'len_q': train_question_length, 'len_a': train_answer_length
         }, train_question)) \
        .shuffle(buffer_size=len(train_sentence)) \
        .repeat(FLAGS.num_epochs) \
        .batch(batch_size=batch_size, drop_remainder=True)


def get_validation_dataset(batch_size):
    # Load evaluation data
    eval_sentence = np.load(FLAGS.eval_sentence)
    eval_question = np.load(FLAGS.eval_question).astype(np.int)
    eval_answer = np.load(FLAGS.eval_answer)
    eval_sentence_length = np.load(FLAGS.eval_sentence_length)
    eval_question_length = np.load(FLAGS.eval_question_length)
    eval_answer_length = np.load(FLAGS.eval_answer_length)

    # Evaluation input function for estimator
    return tf.data.Dataset.from_tensor_slices((
        {
            "s": eval_sentence, 'q': eval_question, 'a': eval_answer,
            'len_s': eval_sentence_length, 'len_q': eval_question_length, 'len_a': eval_answer_length
        }, eval_question)) \
        .shuffle(len(eval_sentence)) \
        .batch(batch_size=batch_size, drop_remainder=True)


def get_test_dataset(batch_size):
    # Load test data
    test_sentence = np.load(FLAGS.test_sentence)
    test_answer = np.load(FLAGS.test_answer)
    test_sentence_length = np.load(FLAGS.test_sentence_length)
    test_answer_length = np.load(FLAGS.test_answer_length)

    # prediction input function for estimator
    return tf.data.Dataset.from_tensor_slices({"s": test_sentence, 'a': test_answer,
                                               'len_s': test_sentence_length, 'len_a': test_answer_length}).batch(
        batch_size=batch_size)


def get_params():
    # Load parameters
    model_params = getattr(params, FLAGS.params)().values()

    # Add embedding path to model_params
    model_params['embedding'] = FLAGS.embedding
    return model_params


def loss_function(batch_size, dtype, maxlen_q):
    def _with_question_length(len_q):
        @tf.function
        def _loss(targets, candidate_logits):
            # Loss
            targets = tf.cast(targets, tf.int32, name="targets")
            labels = tf.concat([targets[:, 1:], tf.zeros([batch_size, 1], dtype=tf.int32)], axis=1, name='label_q')
            weight_q = tf.sequence_mask(tf.squeeze(len_q), maxlen_q, dtype)

            return tfa.seq2seq.sequence_loss(
                candidate_logits,
                labels,
                weight_q,  # [batch, length]
                average_across_timesteps=True,
                average_across_batch=True,
                sum_over_batch=False,
                sum_over_timesteps=False,
                softmax_loss_function=None  # default : sparse_softmax_cross_entropy
            )
        return _loss

    return _with_question_length


def main():
    model_params = get_params()
    b_size = model_params['batch_size']

    config = {
        "show_loss_steps": 100,
        "dev_steps": 500
    }

    if FLAGS.mode == "train":
        train_dataset = get_train_dataset(b_size)
        valid_dataset = get_validation_dataset(b_size)

        # Optimizer
        learning_rate = model_params['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # eval_metric for estimator

        ce_loss = loss_function(batch_size=b_size, dtype=model_params['dtype'], maxlen_q=model_params['maxlen_q_train'])
        nn = model.ASs2s(model_params)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=FLAGS.model_dir + "weights.{epoch:02d}-{bleu:.2f}.hdf5",
            monitor='bleu',
            verbose=1,
            save_freq=300,
            save_best_only=True
        )

        @tf.function
        def train_step(features, labels):
            with tf.GradientTape() as tape:
                candidate_logits = nn(features, training=True)
                loss = ce_loss(feature['len_q'])(labels, candidate_logits)

            grads = tape.gradient(loss, nn.trainable_weights)
            optimizer.apply_gradients(zip(grads, nn.trainable_weights))
            return loss

        def dev_step():
            references = []
            candidates = []

            @tf.function
            def _make_pred(feature):
                return nn(feature, training=False)

            for features, labels in valid_dataset.take(10):
                predictions = _make_pred(features)
                candidates.extend(predictions)
                references.extend(labels.numpy())

            bleu_score = mytools.nltk_blue_score(references, candidates)
            tf.print(bleu_score)

        t1 = datetime.now()
        for step, (feature, label) in enumerate(train_dataset):

            if step % config['dev_steps'] == 0:
                dev_step()

            loss = train_step(feature, label)
            if step % config['show_loss_steps'] == 0:
                t2 = datetime.now()
                tf.print("Step ", step, " - Loss ", loss, " - Elapsed time: ", (t2 - t1).total_seconds(), " seconds")
                t1 = t2

    else:
        nn = tf.keras.models.load_model(FLAGS.model_dir)
        test_data = get_test_dataset(b_size)

        # prediction
        predict_results = nn.predict(test_data)
        # write result(question) into file
        write_result(predict_results, FLAGS.dictionary)
        # print_result(predict_results)


def run(opt):
    global FLAGS
    FLAGS = opt
    main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    base_path = 'data/processed/mpqg_substitute_a_vocab_include_a/'
    parser.add_argument('--mode', type=str, default='train', help='train, eval')
    parser.add_argument('--train_sentence', type=str, default=base_path + 'train_sentence.npy',
                        help='path to the training sentence.')
    parser.add_argument('--train_question', type=str, default=base_path + 'train_question.npy',
                        help='path to the training question.')
    parser.add_argument('--train_answer', type=str, default=base_path + 'train_answer.npy',
                        help='path to the training answer')
    parser.add_argument('--train_sentence_length', type=str, default=base_path + 'train_length_sentence.npy')
    parser.add_argument('--train_question_length', type=str, default=base_path + 'train_length_question.npy')
    parser.add_argument('--train_answer_length', type=str, default=base_path + 'train_length_answer.npy')
    parser.add_argument('--eval_sentence', type=str, default=base_path + 'dev_sentence.npy',
                        help='path to the evaluation sentence. ')
    parser.add_argument('--eval_question', type=str, default=base_path + 'dev_question.npy',
                        help='path to the evaluation question.')
    parser.add_argument('--eval_answer', type=str, default=base_path + 'dev_answer.npy',
                        help='path to the evaluation answer')
    parser.add_argument('--eval_sentence_length', type=str, default=base_path + 'dev_length_sentence.npy')
    parser.add_argument('--eval_question_length', type=str, default=base_path + 'dev_length_question.npy')
    parser.add_argument('--eval_answer_length', type=str, default=base_path + 'dev_length_answer.npy')
    parser.add_argument('--test_sentence', type=str, default=base_path + 'test_sentence.npy',
                        help='path to the test sentence.')
    parser.add_argument('--test_answer', type=str, default=base_path + 'test_answer.npy',
                        help='path to the test answer')
    parser.add_argument('--test_sentence_length', type=str, default=base_path + 'test_length_sentence.npy')
    parser.add_argument('--test_answer_length', type=str, default=base_path + 'test_length_answer.npy')
    parser.add_argument('--embedding', type=str, default=base_path + 'glove840b_vocab300.npy')
    parser.add_argument('--dictionary', type=str, default=base_path + 'vocab.dic', help='path to the dictionary')
    parser.add_argument('--model_dir', type=str, help='path to save the model')
    parser.add_argument('--params', type=str, help='parameter setting')
    parser.add_argument('--pred_dir', type=str, default='result/predictions.txt', help='path to save the predictions')
    parser.add_argument('--num_epochs', type=int, default=8, help='training epoch size')
    FLAGS = parser.parse_args()
    run(FLAGS)
