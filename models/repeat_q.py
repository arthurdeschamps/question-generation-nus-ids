import argparse
import json
import logging
import os
from logging import info
from typing import Dict
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from data_processing.repeat_q_dataset import RepeatQDataset
from defs import UNKNOWN_TOKEN, REPEAT_Q_SQUAD_DATA_DIR, REPEAT_Q_RAW_DATASETS, GLOVE_PATH, PAD_TOKEN, \
    REPEAT_Q_EMBEDDINGS_FILENAME, REPEAT_Q_VOCABULARY_FILENAME, REPEAT_Q_DATA_DIR
from models.RepeatQ.model import RepeatQ
from models.RepeatQ.trainer import RepeatQTrainer


def make_tf_dataset(base_questions, facts_list, targets, batch_size=1):
    def _gen():
        for base_question, facts, target in zip(base_questions, facts_list, targets):
            yield ({
                       "facts": facts,
                       "base_question": base_question
                   }, target)

    return tf.data.Dataset \
        .from_generator(_gen, output_types=({
                                                "facts": tf.int32, "base_question": tf.int32
                                            }, tf.int32)) \
        .shuffle(buffer_size=len(base_questions), reshuffle_each_iteration=True) \
        .batch(batch_size=batch_size, drop_remainder=True)


def get_data(data_dir, vocabulary, data_limit, batch_size):
    info("Preparing dataset...")
    datasets = {}
    for mode in ("train", "dev", "test"):
        base_questions, facts, targets = RepeatQDataset(
            f"{data_dir}/{mode}.data.json",
            vocabulary,
            data_limit=data_limit
        ).get_dataset()
        datasets[mode] = make_tf_dataset(base_questions, facts, targets, batch_size=batch_size)
    info("Done.")
    return datasets


def train(data_dir, data_limit, batch_size):
    model = RepeatQ()
    reverse_voc = {v: k for k, v in model.vocabulary_word_to_id.items()}
    # Gets the default vocabulary from NQG from now
    data = get_data(data_dir, model.vocabulary_word_to_id, data_limit, batch_size)
    training_data, dev_data, test_data = data["train"], data["dev"], data["test"]
    trainer = RepeatQTrainer(model, training_data, reverse_voc)
    trainer.train()


def translate():
    raise NotImplementedError()


def create_embedding_matrix(pretrained_path, vocab, pad_token, unk_token):
    embeddings = [None for _ in range(len(vocab))]
    info("Generating embedding matrix...")
    with open(pretrained_path, mode='r', encoding='utf-8') as embedding_file:
        for line in tqdm(embedding_file.readlines()):
            token, *token_embedding = line.split(" ")
            if token in vocab:
                token_embedding = [float(val) for val in token_embedding]
                embeddings[vocab[token]] = np.array(token_embedding)
                vocab.pop(token)
    # Takes care of special tokens. Unknown and padding tokens are not present in some cases (GloVE)
    if pad_token in vocab:
        # Takes a random example to use the correct shape to create the zero embedding
        i = 0
        while embeddings[i] is None:
            i += 1
        embeddings[vocab[pad_token]] = np.zeros_like(embeddings[i])
        vocab.pop(pad_token)
    # Mean vector assigned to unknown token and words not found in the given pre-embeddings
    mean_embedding = np.mean(np.array([emb for emb in embeddings if emb is not None]), axis=0)
    if unk_token in vocab:
        embeddings[vocab[unk_token]] = mean_embedding
        vocab.pop(unk_token)
    for not_found_word in vocab:
        embeddings[vocab[not_found_word]] = mean_embedding
    info("Embedding matrix generation completed.")
    return np.array(embeddings, dtype=np.float)


def generate_vocabulary(ds, save_dir, voc_size, unk_token, pad_token) -> Dict[str, int]:
    # Compute word occurrence frequencies
    info("Generating vocabulary...")
    frequencies = {}
    for dp in tqdm(ds):
        assert all(k in dp for k in ("facts", "base_question", "target"))
        words = " ".join(dp["facts"]).split() + dp["base_question"].split(" ")
        if dp["target"] != "":
            words += dp["target"].split(" ")
        for word in words:
            frequencies[word] = frequencies.get(word, 0) + 1
    # Keeps the PAD and UNKNOWN tokens as well as the voc_size - 2 most frequent words
    vocabulary = [pad_token, unk_token] + ["" for _ in range(voc_size - 2)]
    frequencies = sorted(frequencies, key=frequencies.get, reverse=True)
    for i, word in enumerate(frequencies[:voc_size - 2]):
        vocabulary[i + 2] = word
    vocabulary = [w for w in vocabulary if len(w) > 0]
    # Saves the vocabulary
    with open(f"{save_dir}/{REPEAT_Q_VOCABULARY_FILENAME}", mode='w') as f:
        for word in vocabulary:
            f.write(word + "\n")
    info("Vocabulary saved.")
    return {vocabulary[index]: index for index in range(len(vocabulary))}


def preprocess(dataset_path, save_dir, ds_name, voc_size, pretrained_embeddings_path=None):
    save_dir = f"{save_dir}/{ds_name}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not dataset_path.endswith(".json"):
        raise ValueError("Dataset file must be a JSON file.")
    with open(dataset_path, mode='r') as f:
        content = f.readlines()[0]
        content = content.replace(', {"base_question": "in a 2009 national readership survey , what newspaper has '
                                  'the highest number of abc1 25 - 44 readers ?", "target": "", "facts": ', ']')
        ds = json.loads(content)

    pad_token = PAD_TOKEN
    unk_token = UNKNOWN_TOKEN

    vocab = generate_vocabulary(ds, save_dir, voc_size, unk_token=unk_token, pad_token=pad_token)
    # Keeps the embeddings for the words in the vocabulary and saves them to a file for later use
    if pretrained_embeddings_path is not None:
        embeddings = create_embedding_matrix(pretrained_embeddings_path, vocab, pad_token, unk_token)
        np.save(f"{save_dir}/{REPEAT_Q_EMBEDDINGS_FILENAME}", embeddings)
    # Creates the data split  (80/10/10)
    d1 = int(0.8 * len(ds))
    d2 = int(d1 + 0.1 * len(ds))
    ds_train = ds[:d1]
    ds_dev = ds[d1:d2]
    ds_test = ds[d2:]
    for prefix, data in zip(("train", "dev", "test"), (ds_train, ds_dev, ds_test)):
        with open(f"{save_dir}/{prefix}.data.json", mode='w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    logging.getLogger(__name__).setLevel(logging.NOTSET)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="train", type=str, choices=("translate", "preprocess", "train"))
    parser.add_argument("-data_dir", help="Used if action is train or translate. Directory path where all the data "
                                          "files are located.", type=str, required=False,
                        default=REPEAT_Q_SQUAD_DATA_DIR)
    parser.add_argument("-save_dir", help="Used if action is preprocess. Base directory where the processed files will "
                                          "be saved.", type=str, required=False, default=REPEAT_Q_DATA_DIR)
    parser.add_argument("-ds_name", help="Used if action is preprocess. A simple string indicating the name of the "
                                         "dataset to create. This will be appended to the 'save_dir' argument, if "
                                         "provided, or to the default data directory.", required=False, default=None,
                        type=str)
    parser.add_argument("-data_limit", help="Number of examples to use for training. Default to -1, which means "
                                            "taking the whole dataset.", type=int, default=-1, required=False)
    parser.add_argument("-json_path", help="Used if action is preprocess. Path to a JSON file with the schema presented"
                                           "in the README file. This file should contain all data, including train, "
                                           "dev and test data. Please omit target values when these are not known. For"
                                           "an example, see "
                                           "data_preprocessing.data_generator.generate_repeat_q_squad_raw()",
                        type=str, required=False, default=f"{REPEAT_Q_RAW_DATASETS}/squad.json")
    parser.add_argument("-pretrained_embeddings_path", type=str,
                        help="Path to a pre-trained set of embeddings. If passed, it"
                             "will be used to generate an embedding file to use"
                             "for training, based on the created vocabulary.",
                        required=False, default=GLOVE_PATH)
    parser.add_argument("-voc_size", type=int, help="Number of words n to keep in the final vocabulary. The vocabulary "
                                                    "file will contain the n most frequent words", required=False,
                        default=30000)
    parser.add_argument("-batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.action == "train":
        assert args.data_dir is not None
        train(args.data_dir, args.data_limit, args.batch_size)
        info("Training completed.")
    elif args.action == "preprocess":
        assert all(arg is not None for arg in (args.save_dir, args.json_path, args.ds_name))
        preprocess(args.json_path, args.save_dir, args.ds_name, args.voc_size, args.pretrained_embeddings_path)
        info("Preprocessing completed successfully.")
    elif args.action == "translate":
        translate()
        info("Translation completed.")
