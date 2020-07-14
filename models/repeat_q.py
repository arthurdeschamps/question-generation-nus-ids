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
    REPEAT_Q_EMBEDDINGS_FILENAME, REPEAT_Q_VOCABULARY_FILENAME, REPEAT_Q_DATA_DIR, EOS_TOKEN, TRAINED_MODELS_DIR, \
    REPEAT_Q_TRAIN_CHECKPOINTS_DIR
from models.RepeatQ.model import RepeatQ
from models.RepeatQ.model_config import ModelConfiguration
from models.RepeatQ.rl.environment import RepeatQEnvironment
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


def build_vocabulary(vocabulary_path):
    token_to_id = {}
    with open(vocabulary_path, mode='r') as vocab_file:
        for i, token in enumerate(vocab_file.readlines()):
            token_to_id[token.strip()] = i
    return token_to_id


def train(data_dir, data_limit, batch_size):
    default_config = ModelConfiguration.new().with_batch_size(batch_size).with_supervised_epochs(15)
    vocabulary = build_vocabulary(default_config.vocabulary_path)
    # Gets the default vocabulary from NQG from now
    data = get_data(data_dir, vocabulary, data_limit, default_config.batch_size)
    training_data, dev_data, test_data = data["train"], data["dev"], data["test"]
    model = RepeatQ(vocabulary, default_config)
    trainer = RepeatQTrainer(default_config, model, training_data, dev_data, vocabulary)
    trainer.train()


def translate(model_dir, data_dir):
    config = ModelConfiguration.new().with_batch_size(1)
    vocabulary = build_vocabulary(config.vocabulary_path)
    reverse_voc = {v: k for k, v in vocabulary.items()}
    data = get_data(data_dir=data_dir, vocabulary=vocabulary, data_limit=-1, batch_size=config.batch_size)["test"]
    model = RepeatQ(vocabulary, config)
    model.load_weights(model_dir)
    
    def to_string(tokens):
        return " ".join([reverse_voc[t] for t in tokens if t != 0])
    
    for feature, label in data:
        print("Target: " + to_string(label[0].numpy()))
        translated = model.infer(feature)
        print("Hypothesis: " + to_string(translated) + "\n")
        

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


def generate_vocabulary(ds, save_dir, voc_size, unk_token, pad_token, eos_token) -> Dict[str, int]:
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
    # Keeps special tokens as well as the voc_size - len(special words) most frequent words
    special_tokens = [pad_token, unk_token, eos_token]
    vocabulary = special_tokens + ["" for _ in range(voc_size - len(special_tokens))]
    frequencies = sorted(frequencies, key=frequencies.get, reverse=True)
    for i, word in enumerate(frequencies[:voc_size - len(special_tokens)]):
        vocabulary[i + len(special_tokens)] = word
    vocabulary = [w for w in vocabulary if len(w) > 0]
    # Saves the vocabulary
    with open(f"{save_dir}/{REPEAT_Q_VOCABULARY_FILENAME}", mode='w') as f:
        for word in vocabulary:
            f.write(word + "\n")
    info("Vocabulary saved.")
    return {vocabulary[index]: index for index in range(len(vocabulary))}


def preprocess(data_dirpath, save_dir, ds_name, voc_size, pretrained_embeddings_path=None):
    if not os.path.isdir(data_dirpath):
        err_message = f"Directory \"{data_dirpath}\" does not exist."
        raise ValueError(err_message)

    ds_test_path = f"{data_dirpath}/{ds_name}_test.json"
    if not os.path.isfile(ds_test_path):
        raise ValueError(f"Dataset '{ds_test_path}' does not exist.")
    ds_train_path = f"{data_dirpath}/{ds_name}_train.json"
    if not os.path.isfile(ds_train_path):
        raise ValueError(f"Dataset '{ds_train_path}' does not exist.")

    save_dir = f"{save_dir}/{ds_name}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(ds_test_path, mode='r') as f:
        ds_test = json.load(f)
    with open(ds_train_path, mode='r') as f:
        ds_train = json.load(f)

    pad_token = PAD_TOKEN
    unk_token = UNKNOWN_TOKEN
    eos_token = EOS_TOKEN

    vocab = generate_vocabulary(ds_test + ds_train, save_dir, voc_size, unk_token=unk_token, pad_token=pad_token,
                                eos_token=eos_token)
    # Keeps the embeddings for the words in the vocabulary and saves them to a file for later use
    if pretrained_embeddings_path is not None:
        embeddings = create_embedding_matrix(pretrained_embeddings_path, vocab, pad_token, unk_token)
        np.save(f"{save_dir}/{REPEAT_Q_EMBEDDINGS_FILENAME}", embeddings)

    def _save_ds(ds_type, data):
        with open(f"{save_dir}/{ds_type}.data.json", mode='w') as f:
            json.dump(data, f)

    _save_ds("test", ds_test)
    # Split train into train/dev
    d1 = int(0.9 * len(ds_train))
    ds_train_train = ds_train[:d1]
    ds_train_dev = ds_train[d1:]

    _save_ds("dev", ds_train_dev)
    _save_ds("train", ds_train_train)


if __name__ == '__main__':
    logging.getLogger(__name__).setLevel(logging.NOTSET)
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="train", type=str, choices=("translate", "preprocess", "train", "train_rl"))
    parser.add_argument("-data_dir", help="Used if action is train or translate. Directory path where all the data "
                                          "files are located.", type=str, required=False,
                        default=REPEAT_Q_SQUAD_DATA_DIR)
    parser.add_argument("-save_dir", help="Used if action is preprocess. Base directory where the processed files will "
                                          "be saved.", type=str, required=False, default=REPEAT_Q_DATA_DIR)
    parser.add_argument("-ds_name", help="Used if action is preprocess. A simple string indicating the name of the "
                                         "dataset to create. This will be used in addition to the 'save_dir' argument, "
                                         "if provided, or to the default data directory, to compute the final data "
                                         "paths",
                        required=False, default=None,
                        type=str)
    parser.add_argument("-data_limit", help="Number of examples to use for training. Default to -1, which means "
                                            "taking the whole dataset.", type=int, default=-1, required=False)
    parser.add_argument("-preprocess_data_dir", help="Used if action is preprocess. Directory path containing 2 JSON files "
                                              "with the schema presented in the README file. These files' names should "
                                              "contain the suffixes _test and _train.",
                        type=str, required=False, default=f"{REPEAT_Q_RAW_DATASETS}")
    parser.add_argument("-pretrained_embeddings_path", type=str,
                        help="Path to a pre-trained set of embeddings. If passed, it"
                             "will be used to generate an embedding file to use"
                             "for training, based on the created vocabulary.",
                        required=False, default=GLOVE_PATH)
    parser.add_argument("-voc_size", type=int, help="Number of words n to keep in the final vocabulary. The vocabulary "
                                                    "file will contain the n most frequent words", required=False,
                        default=30000)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-model_name", type=str, required=False)
    args = parser.parse_args()
    if args.action == "train":
        assert args.data_dir is not None
        train(args.data_dir, args.data_limit, args.batch_size)
        info("Training completed.")
    elif args.action == "preprocess":
        assert all(arg is not None for arg in (args.save_dir, args.data_dirpath, args.ds_name))
        preprocess(args.preprocess_data_dir, args.save_dir, args.ds_name, args.voc_size, args.pretrained_embeddings_path)
        info("Preprocessing completed successfully.")
    elif args.action == "translate":
        assert args.model_name is not None and args.data_dir is not None
        translate(f"{REPEAT_Q_TRAIN_CHECKPOINTS_DIR}/{args.model_name}", args.data_dir)
        info("Translation completed.")
