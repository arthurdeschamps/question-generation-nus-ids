import subprocess
from functools import reduce
from defs import NQG_PREDS_OUTPUT_PATH, ROOT_DIR, NQG_MODEL_DIR, TRAINED_MODELS_DIR, NQG_DATA_HOME
from data_processing.data_generator import generate_vocabulary_files

ds_types = ("squad", "squad_GA", "squad_+NER", "medquad")
ds_types_str = reduce(lambda t1, t2: t1 + "\n" + t2, ds_types)


class NQG:

    def __init__(self, model_path: str = None):
        """
        :param model_path Path to a .pt model file. None if run in training mode.
        """
        super(NQG, self).__init__()
        self.model_path = model_path

    @staticmethod
    def train(ds_name):
        subprocess.run([
            "bash",
            f"{ROOT_DIR}/training/run_squad_qg.sh",
            f"{NQG_DATA_HOME}/{ds_name}",
            f"{NQG_MODEL_DIR}/code/NQG/seq2seq_pt",
            f"{TRAINED_MODELS_DIR}/nqg/{ds_name}",
            ROOT_DIR,
        ])

    def generate_questions(self):
        subprocess.run([
            "python3",
            f"{NQG_MODEL_DIR}/code/NQG/seq2seq_pt/translate.py",
            "-model",
            f"{ROOT_DIR}/{self.model_path}",
            "-src",
            f"{NQG_DATA_HOME}/dev/data.txt.source.txt",
            "-bio",
            f"{NQG_DATA_HOME}/dev/data.txt.bio",
            "-feats",
            f"{NQG_DATA_HOME}/dev/data.txt.pos",
            f"{NQG_DATA_HOME}/dev/data.txt.ner",
            f"{NQG_DATA_HOME}/dev/data.txt.case",
            "-output",
            NQG_PREDS_OUTPUT_PATH,
            "-tgt",
            f"{NQG_DATA_HOME}/dev/data.txt.target.txt",
            "-verbose"
        ])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab_size', help='Size of the vocabulary to use for training', default=20000)
    parser.add_argument('-dataset_name', type=str, required=True,
                        help=f"Name of the dataset to train on: \n{ds_types_str}")

    args = parser.parse_args()
    assert args.dataset_name in ds_types
    generate_vocabulary_files(dataset_path=f"{NQG_DATA_HOME}/{args.dataset_name}", vocab_size=str(args.vocab_size))
    NQG.train(args.dataset_name)
