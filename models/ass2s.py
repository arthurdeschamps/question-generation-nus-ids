import argparse
import sys
from datetime import datetime
from defs import ASS2S_DIR
sys.path.append(f"{ASS2S_DIR}/")
from main import run


def translate(ds_name):
    raise NotImplementedError()


def preprocess(ds_name):
    raise NotImplementedError()


def train(ds_name):
    model_dir = f"{ASS2S_DIR}/store_model/{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    data_dir = f"{ASS2S_DIR}/data/processed/mpqg_substitute_a_vocab_include_a"
    opt = argparse.Namespace(
        mode="train",
        train_sentence=f'{data_dir}/train_sentence.npy',
        train_question=f'{data_dir}/train_question.npy',
        train_answer=f'{data_dir}/train_answer.npy',
        train_sentence_length=f'{data_dir}/train_length_sentence.npy',
        train_question_length=f'{data_dir}/train_length_question.npy',
        train_answer_length=f'{data_dir}/train_length_answer.npy',
        eval_sentence=f'{data_dir}/dev_sentence.npy',
        eval_question=f'{data_dir}/dev_question.npy',
        eval_answer=f'{data_dir}/dev_answer.npy',
        eval_sentence_length=f'{data_dir}/dev_length_sentence.npy',
        eval_question_length=f'{data_dir}/dev_length_question.npy',
        eval_answer_length=f'{data_dir}/dev_length_answer.npy',
        test_sentence=f'{data_dir}/test_sentence.npy',
        test_answer=f'{data_dir}/test_answer.npy',
        test_sentence_length=f'{data_dir}/test_length_sentence.npy',
        test_answer_length=f'{data_dir}/test_length_answer.npy',
        embedding=f'{data_dir}/glove840b_vocab300.npy',
        dictionary=f'{data_dir}/vocab.dic',
        model_dir=model_dir,
        params="basic_params",
        pred_dir='result/predictions.txt',
        num_epochs=15
    )
    run(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="translate", type=str, help='What to do (e.g. "translate")',
                        choices=("translate", "preprocess", "train"))
    parser.add_argument("-ds", help="Which dataset to use for the specified action (e.g \"squad\").", type=str,
                        choices=("squad",), default="squad")

    args = parser.parse_args()
    if args.action == "translate":
        translate(args.ds)
    elif args.action == "preprocess":
        preprocess(args.ds)
    elif args.action == "train":
        train(args.ds)
