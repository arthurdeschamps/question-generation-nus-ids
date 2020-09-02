import argparse
import os
import pathlib
import sys
from datetime import datetime

from data_processing.data_generator import generate_ass2s_mpqg_features
from defs import ASS2S_DIR, \
    ASS2S_PRED_DIR, ASS2S_PROCESSED_DIR, ASS2S_PROCESSED_SQUAD_DIR

sys.path.append(f"{ASS2S_DIR}/")
from main import run
from process_mpqg_data import run as process_mpqg_data
from process_embedding import run as process_embedding


def opt(model_dir, data_dir, mode, voc_size=None, pred_path = None):
    if voc_size is None:
        voc_size = 30000
    assert mode in ("train", "eval", "infer")
    return argparse.Namespace(
            mode=mode,
            train_sentence=f'{data_dir}/train_sentence.npy',
            train_question=f'{data_dir}/train_question.npy',
            train_answer=f'{data_dir}/train_answer.npy',
            train_sentence_length=f'{data_dir}/train_length_sentence.npy',
            train_question_length=f'{data_dir}/train_length_question.npy',
            train_answer_length=f'{data_dir}/train_length_answer.npy',
            train_ners=f"{data_dir}/filtered_txt/train_sentence_ner_map.json",
            eval_sentence=f'{data_dir}/dev_sentence.npy',
            eval_question=f'{data_dir}/dev_question.npy',
            eval_answer=f'{data_dir}/dev_answer.npy',
            eval_sentence_length=f'{data_dir}/dev_length_sentence.npy',
            eval_question_length=f'{data_dir}/dev_length_question.npy',
            eval_answer_length=f'{data_dir}/dev_length_answer.npy',
            eval_ners=f"{data_dir}/filtered_txt/dev_sentence_ner_map.json",
            test_sentence=f'{data_dir}/test_sentence.npy',
            test_answer=f'{data_dir}/test_answer.npy',
            test_sentence_length=f'{data_dir}/test_length_sentence.npy',
            test_answer_length=f'{data_dir}/test_length_answer.npy',
            test_question=f'{data_dir}/test_question.npy',
            test_ners=f"{data_dir}/filtered_txt/test_sentence_ner_map.json",
            embedding=f'{data_dir}/glove840b_vocab300.npy',
            dictionary=f'{data_dir}/vocab.dic',
            model_dir=model_dir,
            params="basic_params",
            pred_dir=pred_path,
            num_epochs=20,
            voca_size=voc_size
        )


def translate(ds_name, model_name, ass2s_data_dir, voc_size=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    pathlib.Path(ASS2S_PRED_DIR).mkdir(parents=True, exist_ok=True)
    model_dir = f"{ASS2S_DIR}/store_model/{model_name}"
    params = opt(
        model_dir, ass2s_data_dir, "infer", pred_path=f"{ASS2S_PRED_DIR}/{ds_name}_preds.txt", voc_size=voc_size
    )
    run(params)


def preprocess(ds_name):
    #generate_ass2s_mpqg_features(ds_name)
    for data_dir in (f"{ASS2S_PROCESSED_DIR}/{ds_name}", f"{ASS2S_PROCESSED_DIR}/{ds_name}_mturk"):
        process_mpqg_data(data_dir)
        process_embedding(data_dir)


def train(ds_name, voc_size, model_name=None):
    if model_name is None:
        model_name = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
    model_dir = f"{ASS2S_DIR}/store_model/{model_name}"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    run(opt(model_dir, ds_name, "train", voc_size=voc_size))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="translate", type=str, help='What to do (e.g. "translate")',
                        choices=("translate", "preprocess", "train", "eval"))
    parser.add_argument("-ds", help="Which dataset to use for the specified action (e.g \"squad\").", type=str,
                        choices=("squad", "squad_repeat_q", "squad_repeat_q_mapped_triples",
                                 "squad_repeat_q_mapped_triples_mturk", "squad_repeat_q_mturk",), default="squad")
    parser.add_argument("-model", help="Name of the directory containing the model's checkpoints to restore from for"
                                       "evaluation or name of the directory to save the model being trained to",
                        type=str, required=False)
    parser.add_argument("-voc_size", help="Size of the vocabulary to use or used for training.", type=int,
                        required=False, default=None)

    args = parser.parse_args()
    if "squad" == args.ds:
        data_dir = ASS2S_PROCESSED_SQUAD_DIR
    else:
        data_dir = f"{ASS2S_PROCESSED_DIR}/{args.ds}/mpqg_substitute_a_vocab_include_a"
    if args.action == "translate":
        translate(args.ds, args.model, data_dir, voc_size=args.voc_size)
    elif args.action == "preprocess":
        preprocess(args.ds)
    elif args.action == "train":
        train(data_dir, model_name=args.model, voc_size=args.voc_size)
