import os
import subprocess
from defs import NQG_MODEL_DIR, NQG_DATA_HOME
from data_processing.nqg_dataset import NQGDataset
from data_processing.pre_processing import NQGDataPreprocessor
import numpy as np


def generate_vocabulary_files(dataset_path, vocab_size):

    collect_vocab = f"{NQG_MODEL_DIR}/code/NQG/seq2seq_pt/CollectVocab.py"
    python = "python3"
    subprocess.run([
        python,
        collect_vocab,
        f"{dataset_path}/train/data.txt.source.txt",
        f"{dataset_path}/train/data.txt.target.txt",
        f"{dataset_path}/train/vocab.txt",
    ])
    subprocess.run([
        python,
        collect_vocab,
        f"{dataset_path}/train/data.txt.bio",
        f"{dataset_path}/train/bio.vocab.txt",
    ])
    subprocess.run([
        python,
        collect_vocab,
        f"{dataset_path}/train/data.txt.pos",
        f"{dataset_path}/train/data.txt.ner",
        f"{dataset_path}/train/data.txt.case",
        f"{dataset_path}/train/feat.vocab.txt",
    ])
    output = subprocess.run([
        "head",
        "-n",
        f"{vocab_size}",
        f"{dataset_path}/train/vocab.txt"
    ], capture_output=True).stdout
    with open(f"{dataset_path}/train/vocab.txt.pruned", mode='wb+') as f:
        f.write(output)


def generate_nqg_features(mode: str, dataset_name: str):
    if mode not in ("train", "dev"):
        raise ValueError(f"mode should be one of 'train' or 'dev")

    ds = NQGDataset(dataset_name=dataset_name, mode=mode)
    if mode == 'dev':
        # Need to split into dev/test
        segments = ['dev', 'test']
        c_dev, a_dev, q_dev, c_test, a_test, q_test = ds.get_split(0.5)
        data = [(c_dev, a_dev, q_dev), (c_test, a_test, q_test)]
    else:
        segments = ['train']
        data = [ds.get_dataset()]

    for segment_type, segment_data in zip(segments, data):
        data_preprocessor = NQGDataPreprocessor(segment_data[0])
        answer_starts = np.array(list(answer.start_index for answer in segment_data[1]))
        answer_lengths = np.array(list(answer.nb_words for answer in segment_data[1]))
        ner = data_preprocessor.create_ner_sequences()
        bio = data_preprocessor.create_bio_sequences(answer_starts, answer_lengths)
        case = data_preprocessor.create_case_sequences()
        pos = data_preprocessor.create_pos_sequences()
        passages = data_preprocessor.uncased_sequences()
        data_dir = f"{NQG_DATA_HOME}/{dataset_name}/{segment_type}"
        os.makedirs(data_dir, exist_ok=True)

        for data_name, content in (("source.txt", passages), ("target.txt", segment_data[2]), ("bio", bio),
                                ("case", case), ("ner", ner), ("pos", pos)):
            if content is not None:
                fname = f"{data_dir}/data.txt.{data_name}"
                np.savetxt(fname, content, fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    args = parser.parse_args()

    if args.dataset_name == 'nqg_squad':
        generate_nqg_features('dev', 'squad')
        generate_nqg_features('train', 'squad')
    elif args.dataset_name == 'nqg_medquad':
        generate_nqg_features('dev', 'medquad')
        generate_nqg_features('train', 'medquad')
