import argparse
import json
import pickle as pkl
import time
from datetime import datetime
from logging import info

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import params
from model import ASs2s
import mytools
# Enable logging for tf.estimator
tf.get_logger().setLevel("INFO")
FLAGS = None


def remove_eos(sentence, eos='<EOS>', pad='<PAD>'):
    if eos in sentence:
        return sentence[:sentence.index(eos)]
    elif pad in sentence:
        return sentence[:sentence.index(pad)]
    else:
        return sentence


def write_result(predictions, ner_mappings, reversed_dic):
    print('Writing into file...')
    with open(FLAGS.pred_dir, 'w') as f:
        for output, ner_mapping in zip(predictions, ner_mappings):
            try:
                output = output.numpy()
                if -1 in output:  # beam search
                    output = output[:output.index(-1)]
                indices = [reversed_dic[index] for index in output]
                sentence = remove_eos(indices)
                sentence = ' '.join(sentence)
                sentence = mytools.replace_unknown_tokens(sentence, ner_mapping)
                sentence = mytools.remove_adjacent_duplicate_grams(sentence, n=4)
                f.write(sentence + "\n")

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


def get_test_dataset(b_size):
    # Load test data
    test_sentence = np.load(FLAGS.test_sentence)
    test_answer = np.load(FLAGS.test_answer)
    test_sentence_length = np.load(FLAGS.test_sentence_length)
    test_answer_length = np.load(FLAGS.test_answer_length)

    # prediction input function for estimator
    return tf.data.Dataset.from_tensor_slices(
        {"s": test_sentence, 'a': test_answer, 'len_s': test_sentence_length, 'len_a': test_answer_length}
    ).batch(batch_size=b_size)


def get_ner_mappings(mode):
    if mode == "train":
        filepath = FLAGS.train_ners
    elif mode == "dev":
        filepath = FLAGS.dev_ners
    elif mode == "test":
        filepath = FLAGS.test_ners
    else:
        raise ValueError(f"mode '{mode}' not recognized. Choices are: 'train', 'dev' or 'test'.")
    with open(filepath) as ner_file:
        ners = json.load(ner_file)
    return ners


def get_params(training):
    # Load parameters
    model_params = getattr(params, FLAGS.params)().values()

    if training:
        model_params['beam_width'] = 0
    model_params["voca_size"] = FLAGS.voca_size
    model_params["batch_size"] = FLAGS.batch_size

    # Add embedding path to model_params
    model_params['embedding'] = FLAGS.embedding
    return model_params


def checkpoint(model: tf.keras.Model, model_acc: float, epoch):
    filename = "weights.epoch_{:d}-bleu_{:.2f}.ckpt".format(epoch, model_acc)
    filepath = f"{FLAGS.model_dir}/{filename}"
    tf.print(f"Checkpoint filename: {filename}")
    model.save_weights(filepath)


def loss_function(batch_size, dtype, maxlen_q):
    def _with_question_length(len_q):
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
    model_params = get_params(training=FLAGS.mode == "train")
    model = ASs2s(model_params)

    info('Loading dic file...')
    with open(FLAGS.dictionary, mode='rb') as dic:
        dic_file = pkl.load(dic)
    reversed_dic = dict((y, x) for x, y in dic_file.items())

    b_size = model_params['batch_size']

    config = {
        "show_loss_steps": 100,
        "dev_steps": 500,
    }

    best_acc_score = 0.0

    if FLAGS.mode == "train":

        train_dataset = get_train_dataset(b_size)
        valid_dataset = get_validation_dataset(b_size)

        # Optimizer
        learning_rate = model_params['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # eval_metric for estimator

        ce_loss = loss_function(batch_size=b_size, dtype=model_params['dtype'], maxlen_q=model_params['maxlen_q_train'])

        @tf.function
        def train_step(features, labels):
            with tf.GradientTape() as tape:
                candidate_logits = model(features, training=True)
                loss = ce_loss(feature['len_q'])(labels, candidate_logits)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss

        t1 = datetime.now()
        step = 0
        for epoch in range(1, FLAGS.num_epochs+1):

            def dev_step():
                references = []
                candidates = []

                @tf.function
                def _make_pred(feature):
                    return model(feature, training=False)

                for features, labels in valid_dataset:
                    predictions = _make_pred(features)
                    candidates.extend(predictions)
                    references.extend(labels.numpy())

                bleu_score = 100 * mytools.nltk_blue_score(references, candidates)
                return bleu_score

            for feature, label in train_dataset:
                step += 1
                if step % config['dev_steps'] == 0:
                    acc = dev_step()
                    tf.print("BLEU-4: ", acc)
                    if acc > best_acc_score:
                        tf.print("Saving model...")
                        checkpoint(model, acc, epoch)
                        best_acc_score = acc

                loss = train_step(feature, label)
                if step % config['show_loss_steps'] == 0:
                    t2 = datetime.now()
                    tf.print("Epoch ", epoch,
                             " - Step ", step,
                             " - Loss ", loss,
                             " - Elapsed time: ", (t2 - t1).total_seconds(), " seconds")
                    t1 = t2

    else:
        model.load_weights(FLAGS.model_dir)
        test_data = get_test_dataset(b_size)
        test_ners = get_ner_mappings(mode="test")

        # prediction
        @tf.function
        def predict_results(features):
            return model(features, training=False)
        predictions = []
        for x in test_data:
            predictions.extend(predict_results(x))
        assert len(test_ners) == len(predictions)
        # write result(question) into file
        write_result(predictions, test_ners, reversed_dic)
        # print_result(predict_results)


def run(opt):
    global FLAGS
    FLAGS = opt
    # These 2 lines are at the moment required because of a bug in Tensorflow
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    main()
    print("Done.")


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
