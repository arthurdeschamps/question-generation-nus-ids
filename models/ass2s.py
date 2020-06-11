import argparse
import os
import pathlib
import sys
from datetime import datetime

from data_processing.data_generator import generate_ass2s_mpqg_features
from defs import ASS2S_DIR, ASS2S_PROCESSED_SQUAD_DIR, ASS2S_PROCESSED_MPQG_DATA, ASS2S_SQUAD_PREDS_OUTPUT_PATH, \
    ASS2S_PRED_DIR

sys.path.append(f"{ASS2S_DIR}/")
from main import run

ass2s_data_dir = f"{ASS2S_DIR}/data/processed/mpqg_substitute_a_vocab_include_a"


def opt(model_dir, data_dir, mode, pred_path = None):
    assert mode in ("train", "eval", "infer")
    return argparse.Namespace(
            mode=mode,
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
            pred_dir=pred_path,
            num_epochs=15
        )


def eval(ds_name, model_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model_dir = f"{ASS2S_DIR}/store_model/{model_dir}"
    run(opt(model_dir, ass2s_data_dir, "eval"))


def translate(ds_name, model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    pathlib.Path(ASS2S_PRED_DIR).mkdir(parents=True, exist_ok=True)
    if ds_name == "squad":
        model_dir = f"{ASS2S_DIR}/store_model/{model_name}"
        run(opt(model_dir, ass2s_data_dir, "infer", pred_path=ASS2S_SQUAD_PREDS_OUTPUT_PATH))


def preprocess(ds_name):
    if ds_name == "squad":
        generate_ass2s_mpqg_features(ds_name)
    else:
        raise NotImplementedError()


def train(ds_name, model_name=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    if model_name is None:
        model_name = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
    model_dir = f"{ASS2S_DIR}/store_model/{model_name}"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    run(opt(model_dir, ass2s_data_dir, "train"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="translate", type=str, help='What to do (e.g. "translate")',
                        choices=("translate", "preprocess", "train", "eval"))
    parser.add_argument("-ds", help="Which dataset to use for the specified action (e.g \"squad\").", type=str,
                        choices=("squad",), default="squad")
    parser.add_argument("-model", help="Name of the directory containing the model's checkpoints to restore from for"
                                       "evaluation", type=str, required=False)

    args = parser.parse_args()
    if args.action == "translate":
        translate(args.ds, args.model)
    elif args.action == "eval":
        eval(args.ds, args.model)
    elif args.action == "preprocess":
        preprocess(args.ds)
    elif args.action == "train":
        train(args.ds)
