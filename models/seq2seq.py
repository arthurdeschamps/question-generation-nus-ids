import subprocess
from defs import ROOT_DIR, NQG_MODEL_DIR, TRAINED_MODELS_DIR, NQG_DATA_HOME
from data_processing.data_generator import generate_vocabulary_files


class NQG:

    def __init__(self, model_path: str = None):
        """
        :param model_path Path to a .pt model file. None if run in training mode.
        """
        super(NQG, self).__init__()
        self.model_path = model_path

    @staticmethod
    def train(dataset_name, dataset_dir, bio_data_dir):
        subprocess.run([
            "bash",
            f"{ROOT_DIR}/training/run_squad_qg.sh",
            dataset_dir,
            bio_data_dir,
            f"{NQG_MODEL_DIR}/code/NQG/seq2seq_pt",
            f"{TRAINED_MODELS_DIR}/nqg/{dataset_name}",
            ROOT_DIR,
        ])


if __name__ == '__main__':
    ds_types = ("squad", "squad_GA", "squad_+NER", "medquad", "squad_NA")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab_size', help='Size of the vocabulary to use for training', default=20000)
    parser.add_argument('-dataset_name', type=str, required=True, choices=ds_types,
                        help=f"Name of the dataset to train on.")

    args = parser.parse_args()
    assert args.dataset_name in ds_types

    ds_name = args.dataset_name
    data_home = f"{NQG_DATA_HOME}/"
    if ds_name in ("squad", "medquad", "squad_+NER"):
        data_home += ds_name
        bio_dir = data_home
    else:
        data_home += "squad"
        bio_dir = f"{NQG_DATA_HOME}/{ds_name}"

    generate_vocabulary_files(
        dataset_path=data_home,
        bio_path=bio_dir,
        vocab_size=str(args.vocab_size)
    )
    NQG.train(ds_name, data_home, bio_dir)
