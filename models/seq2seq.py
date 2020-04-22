import os
import subprocess

from data_utils.nqg_dataset import NQGDataset
from defs import PROCESSED_DATA_DIR, NQG_PREDS_OUTPUT_PATH, PRETRAINED_MODELS_DIR, ROOT_DIR, NQG_DIR, TRAINED_MODELS_DIR
from data_utils.pre_processing import NQGDataPreprocessor
import numpy as np


class NQG:
    data_home = f"{PROCESSED_DATA_DIR}/nqg"

    def __init__(self, model_path: str = None):
        """
        :param model_path Path to a .pt model file. None if run in training mode.
        """
        super(NQG, self).__init__()
        self.model_path = model_path

    @staticmethod
    def train():
        subprocess.run([
            "bash",
            f"{ROOT_DIR}/training/run_squad_qg.sh",
            f"{NQG.data_home}",
            f"{NQG_DIR}/code/NQG/seq2seq_pt",
            f"{TRAINED_MODELS_DIR}/nqg"
        ])

    def generate_questions(self):
        subprocess.run([
            "python3",
            f"{NQG_DIR}/code/NQG/seq2seq_pt/translate.py",
            "-model",
            f"{ROOT_DIR}/{self.model_path}",
            "-src",
            f"{self.data_home}/dev/data.txt.source.txt",
            "-bio",
            f"{self.data_home}/dev/data.txt.bio",
            "-feats",
            f"{self.data_home}/dev/data.txt.pos",
            f"{self.data_home}/dev/data.txt.ner",
            f"{self.data_home}/dev/data.txt.case",
            "-output",
            NQG_PREDS_OUTPUT_PATH,
            "-tgt",
            f"{self.data_home}/dev/data.txt.target.txt",
            "-verbose"
        ])

    @staticmethod
    def generate_vocabulary_files(vocab_size='45000'):
        collect_vocab = f"{NQG_DIR}/code/NQG/seq2seq_pt/CollectVocab.py"
        python = "python3"
        subprocess.run([
            python,
            collect_vocab,
            f"{NQG.data_home}/train/data.txt.source.txt",
            f"{NQG.data_home}/train/data.txt.target.txt",
            f"{NQG.data_home}/train/vocab.txt",
        ])
        subprocess.run([
            python,
            collect_vocab,
            f"{NQG.data_home}/train/data.txt.bio",
            f"{NQG.data_home}/train/bio.vocab.txt",
        ])
        subprocess.run([
            python,
            collect_vocab,
            f"{NQG.data_home}/train/data.txt.pos",
            f"{NQG.data_home}/train/data.txt.ner",
            f"{NQG.data_home}/train/data.txt.case",
            f"{NQG.data_home}/train/feat.vocab.txt",
        ])
        output = subprocess.run([
            "head",
            "-n",
            f"{vocab_size}",
            f"{NQG.data_home}/train/vocab.txt"
        ], capture_output=True).stdout
        with open(f"{NQG.data_home}/train/vocab.txt.pruned", mode='wb+') as f:
            f.write(output)

    @staticmethod
    def generate_features(mode: str):
        if mode not in ("train", "dev"):
            raise ValueError(f"mode should be one of 'train' or 'dev")

        ds = NQGDataset(dataset_type=f"squad_{mode}")
        contexts, answers, questions = ds.get_dataset()
        data_preprocessor = NQGDataPreprocessor(contexts)
        answer_starts = np.array(list(answer.start_index for answer in answers))
        answer_lengths = np.array(list(answer.nb_words for answer in answers))
        ner = data_preprocessor.create_ner_sequences()
        bio = data_preprocessor.create_bio_sequences(answer_starts, answer_lengths)
        case = data_preprocessor.create_case_sequences()
        pos = data_preprocessor.create_pos_sequences()
        passages = data_preprocessor.uncased_sequences()
        data_dir = f"{NQG.data_home}/{mode}"
        os.makedirs(data_dir, exist_ok=True)
        for data_name, data in (("source.txt", passages), ("target.txt", questions), ("bio", bio),
                                ("case", case), ("ner", ner), ("pos", pos)):
            fname = f"{data_dir}/data.txt.{data_name}"
            np.savetxt(fname, data, fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        help='One of "predict", "train", "predict_original" or "generate_data"',
                        default="train")
    parser.add_argument('--model_path', type=str, nargs=1,
                        help='Path to the model', required=False,
                        default=f"{PRETRAINED_MODELS_DIR}/nqg/data/redistribute/QG/models/NQG_plus/base_20_epochs/" +
                                f"model_dev_metric_0.133_e7.pt")

    args = parser.parse_args()

    if args.action == 'generate_data':
        NQG.generate_features('dev')
        NQG.generate_features('train')
        NQG.generate_vocabulary_files()
    elif args.action == 'train':
        NQG.train()
    elif args.action == 'predict_original':
        subprocess.run([
            "python3",
            f"{NQG_DIR}/code/NQG/seq2seq_pt/translate.py",
            "-model",
            f"{ROOT_DIR}/{args.model_path}",
            "-src",
            "./pre_trained/nqg/data/redistribute/QG/test/dev.txt.shuffle.test.source.txt",
            "-bio",
            "./pre_trained/nqg/data/redistribute/QG/test/dev.txt.shuffle.test.bio",
            "-feats",
            "./pre_trained/nqg/data/redistribute/QG/test/dev.txt.shuffle.test.pos",
            "./pre_trained/nqg/data/redistribute/QG/test/dev.txt.shuffle.test.ner",
            "./pre_trained/nqg/data/redistribute/QG/test/dev.txt.shuffle.test.case",
            "-output",
            NQG_PREDS_OUTPUT_PATH,
            "-tgt",
            "./pre_trained/nqg/data/redistribute/QG/test/dev.txt.shuffle.test.target.txt",
            "-verbose"
        ])
    elif args.action == 'predict':
        model = NQG(args.model_path[0])
        model.generate_questions()
    else:
        raise NotImplementedError(f"Action '{args.action}' not implemented.")
