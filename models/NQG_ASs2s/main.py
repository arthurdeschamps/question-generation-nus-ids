import argparse
import pickle as pkl
import numpy as np
import tensorflow as tf

import params
import model as model

FLAGS = None


def remove_eos(sentence, eos='<EOS>', pad='<PAD>'):
    if eos in sentence:
        return sentence[:sentence.index(eos)] + '\n'
    elif pad in sentence:
        return sentence[:sentence.index(pad)] + '\n'
    else:
        return sentence + '\n'


def write_result(predict_results, dic_path):
    print('Load dic file...')
    with open(dic_path) as dic:
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
                sentence = ' '.join(indices)
                sentence = remove_eos(sentence)
                f.write(sentence.encode('utf-8'))

            except StopIteration:
                break


def main():
    # Enable logging for tf.estimator
    tf.get_logger().setLevel("INFO")

    # Config
    config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        keep_checkpoint_max=3,
        save_checkpoints_steps=100,

    )

    # Load parameters
    model_params = getattr(params, FLAGS.params)().values()

    # Add embedding path to model_params
    model_params['embedding'] = FLAGS.embedding

    # Define estimator
    nn = tf.estimator.Estimator(
        model_fn=model.q_generation,
        config=config,
        params=model_params
    )

    # Load training data
    train_sentence = np.load(FLAGS.train_sentence)  # train_data
    train_question = np.load(FLAGS.train_question)  # train_label
    train_answer = np.load(FLAGS.train_answer)
    train_sentence_length = np.load(FLAGS.train_sentence_length)
    train_question_length = np.load(FLAGS.train_question_length)
    train_answer_length = np.load(FLAGS.train_answer_length)

    def train_input_fn():
        return tf.data.Dataset.from_tensor_slices((
            {"s": train_sentence,
             'q': train_question,
             'a': train_answer,
             'len_s': train_sentence_length,
             'len_q': train_question_length, 'len_a': train_answer_length
             }, train_sentence)) \
            .shuffle(buffer_size=len(train_sentence))\
            .repeat(FLAGS.num_epochs) \
            .batch(batch_size=model_params['batch_size'], drop_remainder=True)

    # Load evaluation data
    eval_sentence = np.load(FLAGS.eval_sentence)
    eval_question = np.load(FLAGS.eval_question)
    eval_answer = np.load(FLAGS.eval_answer)
    eval_sentence_length = np.load(FLAGS.eval_sentence_length)
    eval_question_length = np.load(FLAGS.eval_question_length)
    eval_answer_length = np.load(FLAGS.eval_answer_length)

    # Evaluation input function for estimator
    def eval_input_fn():
        return tf.data.Dataset.from_tensor_slices((
            {
                "s": eval_sentence, 'q': eval_question, 'a': eval_answer,
                'len_s': eval_sentence_length, 'len_q': eval_question_length, 'len_a': eval_answer_length
            }, None))\
            .batch(batch_size=model_params['batch_size'], drop_remainder=True)

    # train and evaluate
    if FLAGS.mode == 'train':
        # define experiment
        train_spec = tf.estimator.TrainSpec(train_input_fn)
        eval_spec = tf.estimator.EvalSpec(eval_input_fn)
        tf.estimator.train_and_evaluate(nn, train_spec, eval_spec)
    elif FLAGS.mode == "eval":
        nn.evaluate(eval_input_fn)

    else:  # 'pred'
        # Load test data
        test_sentence = np.load(FLAGS.test_sentence)
        test_answer = np.load(FLAGS.test_answer)
        test_sentence_length = np.load(FLAGS.test_sentence_length)
        test_answer_length = np.load(FLAGS.test_answer_length)

        # prediction input function for estimator
        def pred_input_fn():
            return tf.data.Dataset.from_tensor_slices((
                {"s": test_sentence, 'a': test_answer,
                   'len_s': test_sentence_length, 'len_a': test_answer_length},
                None
            )).batch(batch_size=model_params['batch_size'])

        # prediction
        predict_results = nn.predict(input_fn=pred_input_fn)
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
