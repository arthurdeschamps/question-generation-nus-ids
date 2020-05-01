import subprocess

from defs import NQG_PREDS_OUTPUT_PATH, PRETRAINED_MODELS_DIR, ROOT_DIR, NQG_MODEL_DIR, TRAINED_MODELS_DIR, NQG_DATA_HOME


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
            ROOT_DIR
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
    parser.add_argument('action',
                        help='One of "predict", "train", "predict_original" or "generate_data"',
                        default="train")
    parser.add_argument('--model_path', type=str, nargs=1,
                        help='Path to the model', required=False,
                        default=f"{PRETRAINED_MODELS_DIR}/nqg/data/redistribute/QG/models/NQG_plus/base_20_epochs/" +
                                f"model_dev_metric_0.133_e7.pt")
    parser.add_argument('--vocab_size', help='Size of the vocabulary to use for training', default=20000)
    parser.add_argument('--dataset_name', type=str, required=False, help="Name of the dataset to train on (e.g. squad,"
                                                                         "medquad)")

    args = parser.parse_args()

    if args.action == 'train':
        from data_processing.data_generator import generate_vocabulary_files
        assert args.dataset_name is not None
        generate_vocabulary_files(dataset_path=f"{NQG_DATA_HOME}/{args.dataset_name}", vocab_size=str(args.vocab_size))
        NQG.train(args.dataset_name)
    elif args.action == 'predict_original':
        subprocess.run([
            "python3",
            f"{NQG_MODEL_DIR}/code/NQG/seq2seq_pt/translate.py",
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
