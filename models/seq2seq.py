import logging
import sys
from defs import NQG_MODEL_DIR, TRAINED_MODELS_DIR, NQG_DATA_HOME
from data_processing.data_generator import generate_vocabulary_files
import argparse
sys.path.append(f"{NQG_MODEL_DIR}/code/NQG/seq2seq_pt/")
from train_nqg import main as train_nqg


class NQG:

    def __init__(self, model_path: str = None):
        """
        :param model_path Path to a .pt model file. None if run in training mode.
        """
        super(NQG, self).__init__()
        self.model_path = model_path

    @staticmethod
    def train(dataset_name, dataset_dir, bio_data_dir):
        DATAHOME = dataset_dir
        BIO_DIR = bio_data_dir
        save_path = f"{TRAINED_MODELS_DIR}/nqg/{dataset_name}"
        opt = argparse.Namespace(
            save_path=save_path,
            log_home=save_path,
            online_process_data=True,
            train_src=f"{DATAHOME}/train/data.txt.source.txt",
            src_vocab=f"{DATAHOME}/train/vocab.txt.pruned",
            train_bio=f"{BIO_DIR}/train/data.txt.bio",
            bio_vocab=f"{BIO_DIR}/train/bio.vocab.txt",
            train_feats=[f"{DATAHOME}/train/data.txt.pos", f"{DATAHOME}/train/data.txt.ner",
                         f"{DATAHOME}/train/data.txt.case"],
            feat_vocab=f"{DATAHOME}/train/feat.vocab.txt",
            train_tgt=f"{DATAHOME}/train/data.txt.target.txt",
            tgt_vocab=f"{DATAHOME}/train/vocab.txt.pruned",
            layers=1,
            enc_rnn_size=512,
            brnn=True,
            word_vec_size=300,
            dropout=0.5,
            batch_size=64,
            beam_size=5,
            epochs=20, optim="adam", learning_rate=0.001,
            curriculum=0, extra_shuffle=True,
            start_eval_batch=1000, eval_per_batch=500, halve_lr_bad_count=3,
            seed=12345, cuda_seed=12345,
            log_interval=100,
            dev_input_src=f"{DATAHOME}/dev/data.txt.source.txt",
            dev_bio=f"{BIO_DIR}/dev/data.txt.bio",
            dev_feats=[f"{DATAHOME}/dev/data.txt.pos", f"{DATAHOME}/dev/data.txt.ner", f"{DATAHOME}/dev/data.txt.case"],
            dev_ref=f"{DATAHOME}/dev/data.txt.target.txt",
            max_sent_length=100,
            gpus=[1],
            lower_input=False,
            process_shuffle=False,
            input_feed=1,
            maxout_pool_size=2,
            att_vec_size=512,
            dec_rnn_size=512,
            pre_word_vecs_enc=None,
            pre_word_vecs_dec=None,
            max_grad_norm=5,
            max_weight_value=15,
            max_generator_batches=32,
            learning_rate_decay=0.5,
            start_decay_at=8,
            start_epoch=1,
            param_init=0.1
        )
        train_nqg(opt)


if __name__ == '__main__':
    logging.root.setLevel(logging.NOTSET)

    ds_types = ("squad", "squad_GA", "squad_+NER", "medquad", "squad_NA", "squad_repeat_q")

    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab_size', help='Size of the vocabulary to use for training', default=20000)
    parser.add_argument('-dataset_name', type=str, required=True, choices=ds_types,
                        help=f"Name of the dataset to train on.")

    args = parser.parse_args()
    assert args.dataset_name in ds_types

    ds_name = args.dataset_name
    data_home = f"{NQG_DATA_HOME}/"
    if ds_name in ("squad", "medquad", "squad_+NER", "squad_repeat_q"):
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
