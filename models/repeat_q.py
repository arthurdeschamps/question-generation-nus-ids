import argparse
import json
import logging
import os
import sys
from logging import info
from typing import Dict, List
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from data_processing.class_defs import RepeatQExample
from data_processing.repeat_q_dataset import RepeatQDataset
from defs import UNKNOWN_TOKEN, REPEAT_Q_RAW_DATASETS, GLOVE_PATH, PAD_TOKEN, \
    REPEAT_Q_EMBEDDINGS_FILENAME, REPEAT_Q_VOCABULARY_FILENAME, REPEAT_Q_DATA_DIR, EOS_TOKEN, \
    REPEAT_Q_TRAIN_CHECKPOINTS_DIR, REPEAT_Q_FEATURE_VOCABULARY_FILENAME, ASS2S_DIR, \
    REPEAT_Q_PREDS_OUTPUT_DIR
from models.RepeatQ.model import RepeatQ
from models.RepeatQ.model_config import ModelConfiguration
from models.RepeatQ.trainer import RepeatQTrainer
sys.path.append(ASS2S_DIR + '/submodule/')
from mytools import remove_adjacent_duplicate_grams


def make_tf_dataset(examples: List[RepeatQExample], config, shuffle=True, drop_remainder=True, is_training=True):
    def _gen(synth_dataset):
        def _gen_ds():
            for example in examples:
                # When training, we separate the organic and synthetic datasets and train on them
                # in different epochs. For test and dev, we keep everything together
                if is_training and not config.mixed_data and ((not example.is_synthetic_data and synth_dataset) or
                                                              (example.is_synthetic_data and not synth_dataset)):
                    continue
                # For performance assessment, we only use organic data
                if not is_training and example.is_synthetic_data:
                    continue
                facts = example.facts
                f_features = example.facts_features
                base_question = example.base_question
                q_features = example.base_question_features
                if use_pos or use_ner:
                    f_features = [[tf.cast(feature[:tf.shape(facts)[1]], dtype=tf.float32)
                                   for feature in fact_features] for fact_features in f_features]
                    f_features = tf.stack(
                        [tf.stack([tf.pad(feature, paddings=[[0, tf.shape(facts)[1] - tf.shape(feature)[0]]])
                                   for feature in fact_features]) for fact_features in f_features],
                        name="facts_features"
                    )
                    f_features = tf.pad(f_features,
                                        paddings=([0, tf.shape(facts)[0] - tf.shape(f_features)[0]], [0, 0], [0, 0]))
                    # Make last dimension feature dimension (should be 2 for pos + entity)
                    f_features = tf.transpose(f_features, perm=[0, 2, 1])
                    q_features = tf.stack(
                        [tf.pad(tf.cast(feature, dtype=tf.float32), 
                                [(0, tf.shape(base_question)[0] - tf.shape(feature)[0])]) for feature in q_features],
                        axis=0,
                        name="base_question_features"
                    )
                    q_features = tf.transpose(q_features, perm=[1, 0])
                else:
                    # Creates 0-dimensional features (equivalent to not using them)
                    f_features = [[[] for _ in range(tf.shape(facts)[1])] for _ in range(tf.shape(facts)[0])]
                    q_features = [[] for _ in range(tf.shape(base_question)[0])]
                yield {
                          "facts": facts,
                          "facts_features": tf.cast(f_features, dtype=tf.float32),
                          "base_question": base_question,
                          "base_question_features": tf.cast(q_features, dtype=tf.float32),
                          "target_copy_indicator": example.target_question_copy_indicator,
                          "from_base_question": example.is_from_base_question,
                          "is_synthetic": example.is_synthetic_data
                }, example.rephrased_question

        return _gen_ds

    output_types = (
        {
            "facts": tf.int32,
            "facts_features": tf.float32,
            "base_question": tf.int32,
            "base_question_features": tf.float32,
            "target_copy_indicator": tf.int32,
            "from_base_question": tf.bool,
            "is_synthetic": tf.bool
        }, tf.int32
    )

    ds_synth = tf.data.Dataset.from_generator(_gen(True), output_types=output_types)
    ds_org = tf.data.Dataset.from_generator(_gen(False), output_types=output_types)

    # For mixed data, we simply mix the synthetic and organic datasets together
    if config.mixed_data:
        ds = ds_org.concatenate(ds_org)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(examples), reshuffle_each_iteration=True)
        return ds.batch(batch_size=config.batch_size, drop_remainder=drop_remainder)

    if shuffle:
        synth_size = len([ex for ex in examples if ex.is_synthetic_data])
        organic_size = len(examples) - synth_size

        if synth_size > 0:
            ds_synth = ds_synth.shuffle(buffer_size=synth_size, reshuffle_each_iteration=True)
        if organic_size > 0:
            ds_org = ds_org.shuffle(buffer_size=organic_size, reshuffle_each_iteration=True)
    ds_synth = ds_synth.batch(batch_size=config.batch_size, drop_remainder=drop_remainder)
    ds_org = ds_org.batch(batch_size=config.batch_size, drop_remainder=drop_remainder)
    return {"synthetic": ds_synth, "organic": ds_org}


def get_data(data_dir, vocabulary, feature_vocabulary, data_limit, config: ModelConfiguration,
             data_modes=("train", "dev", "test")):
    info("Preparing dataset...")
    datasets = {}
    for mode in data_modes:
        dataset = RepeatQDataset(
            f"{data_dir}/{mode}.data.json",
            vocabulary=vocabulary,
            feature_vocab=feature_vocabulary,
            data_limit=data_limit,
            use_ner_features=use_ner,
            use_pos_features=use_pos,
            reduced_ner_indicators=config.reduced_ner_indicators
        ).get_dataset()
        datasets[mode] = make_tf_dataset(
            examples=dataset,
            shuffle=mode != "test",
            drop_remainder=mode != "test",
            config=config,
            is_training=mode == "train",
        )
    info("Done.")
    return datasets


def build_vocabulary(vocabulary_path):
    token_to_id = {}
    with open(vocabulary_path, mode='r') as vocab_file:
        for i, token in enumerate(vocab_file.readlines()):
            token_to_id[token.strip()] = i
    return token_to_id


def train(args):
    config = ModelConfiguration.new() \
        .with_batch_size(args.batch_size) \
        .with_synth_supervised_epochs(args.synth_supervised_epochs) \
        .with_org_supervised_epochs(args.org_supervised_epochs) \
        .with_saving_model(args.save_model) \
        .with_save_directory_name(args.save_model_dir) \
        .with_dropout_rate(args.dropout_rate) \
        .with_attention_dropout(args.attention_dropout_rate) \
        .with_recurrent_dropout(args.recurrent_dropout_rate) \
        .with_pos_features(use_pos) \
        .with_ner_features(use_ner) \
        .with_mixed_data(args.mixed_data) \
        .with_reduced_ner_indicators(args.reduced_ner_indicators)\
        .with_question_encodings(not args.no_base_question_encodings)\
        .with_glove_embeddings(not args.no_glove)

    tf.print(str(config))
    if args.learning_rate is not None:
        config = config.with_learning_rate(args.learning_rate)
    if args.checkpoint_name is not None:
        config = config.with_restore_supervised_checkpoint().with_supervised_model_checkpoint_path(args.checkpoint_name)
    vocabulary = build_vocabulary(config.vocabulary_path)
    feature_vocabulary = build_vocabulary(config.feature_vocabulary_path)
    # Gets the default vocabulary from NQG from now
    data_dir = f"{REPEAT_Q_DATA_DIR}/{args.ds_name}"
    data = get_data(data_dir, vocabulary, feature_vocabulary, args.data_limit, config)
    training_data, dev_data, test_data = data["train"], data["dev"], data["test"]
    model = RepeatQ(vocabulary, config)
    trainer = RepeatQTrainer(config, model, training_data, dev_data, vocabulary)
    trainer.train()


def translate(model_dir, args):
    config = ModelConfiguration\
        .new()\
        .with_batch_size(32 if args.beam_search_size > 1 else 1)\
        .with_pos_features(use_pos)\
        .with_ner_features(use_ner)\
        .with_reduced_ner_indicators(args.reduced_ner_indicators)\
        .with_question_encodings(not args.no_base_question_encodings)

    save_path = f"{REPEAT_Q_PREDS_OUTPUT_DIR}/{args.ds_name}_predictions.txt"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    def _reverse_voc(voc):
        return {v: k for k, v in voc.items()}

    vocabulary = build_vocabulary(config.vocabulary_path)
    reverse_voc = _reverse_voc(vocabulary)
    feature_voc = build_vocabulary(config.feature_vocabulary_path)
    data = get_data(
        data_dir=f"{REPEAT_Q_DATA_DIR}/{args.ds_name}",
        vocabulary=vocabulary,
        feature_vocabulary=feature_voc,
        data_limit=-1,
        config=config,
        data_modes=["test"]
    )["test"]
    model = RepeatQ(vocabulary, config)
    model.load_weights(model_dir)

    def to_string(tokens, _reverse_voc=reverse_voc):
        tokens = tokens.numpy()
        return " ".join([_reverse_voc[t] for t in tokens]).replace(" <blank>", "")

    with open(save_path, mode='w+') as pred_file:
        for feature, labels in data["organic"]:
            if args.beam_search_size == 1:
                preds, _ = model.get_actions(feature, None, training=False, phase=RepeatQTrainer.supervised_phase)
            else:
                preds = model.beam_search(feature, beam_search_size=args.beam_search_size)
            for label, base_question, facts, pred in zip(labels, feature["base_question"], feature["facts"], preds):
                translated = remove_adjacent_duplicate_grams(to_string(pred))
                tf.print("Base question: ", to_string(base_question))
                for fact in facts:
                    tf.print("Fact: ", to_string(fact))
                tf.print("Target: ", to_string(label))
                #tf.print(to_string(label))
                tf.print("Prediction: ", translated, "\n")
                pred_file.write(translated + "\n")


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


def generate_feature_vocabulary(ds, save_dir, null_tag="O"):
    voc = set()
    feature_names = [k for k in ds[0].keys() if "tags" in k]

    def _extend_voc(feature):
        if isinstance(feature, list):
            [_extend_voc(f) for f in feature]
        else:
            for term in feature.split():
                voc.add(term)

    for dp in ds:
        [_extend_voc(dp[k]) for k in feature_names]
    # Place the null tag at the first position so that it will have id 0
    voc.remove(null_tag)
    with open(f"{save_dir}/{REPEAT_Q_FEATURE_VOCABULARY_FILENAME}", mode='w') as f:
        f.write(f"{null_tag}\n")
        for k in voc:
            f.write(f"{k}\n")
    info("Feature vocabulary saved.")


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
    ds_train_path = f"{data_dirpath}/{ds_name}_train.json"
    ds_dev_path = f"{data_dirpath}/{ds_name}_dev.json"

    for path in (ds_test_path, ds_dev_path, ds_train_path):
        if not os.path.isfile(path):
            raise ValueError(f"Dataset '{path}' does not exist.")

    save_dir = f"{save_dir}/{ds_name}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(ds_test_path, mode='r') as f:
        ds_test = json.load(f)
    with open(ds_train_path, mode='r') as f:
        ds_train = json.load(f)
    with open(ds_dev_path, mode='r') as f:
        ds_dev = json.load(f)

    pad_token = PAD_TOKEN
    unk_token = UNKNOWN_TOKEN
    eos_token = EOS_TOKEN

    all_data = ds_test + ds_train + ds_dev
    # Generate the feature vocabulary (POS tags, answer indicators, etc)
    generate_feature_vocabulary(all_data, save_dir)
    # Generate the word vocabulary
    vocab = generate_vocabulary(all_data, save_dir, voc_size, unk_token=unk_token, pad_token=pad_token,
                                eos_token=eos_token)
    # Keeps the embeddings for the words in the vocabulary and saves them to a file for later use
    if pretrained_embeddings_path is not None:
        embeddings = create_embedding_matrix(pretrained_embeddings_path, vocab, pad_token, unk_token)
        np.save(f"{save_dir}/{REPEAT_Q_EMBEDDINGS_FILENAME}", embeddings)

    def _save_ds(ds_type, data):
        with open(f"{save_dir}/{ds_type}.data.json", mode='w') as f:
            json.dump(data, f)

    _save_ds("test", ds_test)
    _save_ds("train", ds_train)
    _save_ds("dev", ds_dev)


if __name__ == '__main__':
    logging.getLogger(__name__).setLevel(logging.NOTSET)

    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="train", type=str, choices=("translate", "preprocess", "train"))
    parser.add_argument("-save_data_dir", help="If action is preprocess, base directory where the processed files will "
                                          "be saved.", type=str, required=False, default=REPEAT_Q_DATA_DIR)
    parser.add_argument("-save_model_dir", help="If action is train, name of the directory to save the checkpoints "
                                          "to (only the name of it, not the full path which is predetermined)",
                        type=str, default=None)
    parser.add_argument("-ds_name", help="Name of the dataset directory to work with. It will be looked for at the "
                                         "dedicated path in th project files. Please check the README to know where "
                                         "to place your dataset(s).",
                        required=False, default="squad",
                        type=str)
    parser.add_argument("-data_limit", help="Number of examples to use for training. Default to -1, which means "
                                            "taking the whole dataset.", type=int, default=-1, required=False)
    parser.add_argument("-preprocess_data_dir",
                        help="Used if action is preprocess. Directory path containing 2 JSON files with the schema "
                             "presented in the README file. These files' names should contain the suffixes _test and "
                             "_train.",
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
    parser.add_argument("-learning_rate", type=float, required=False,
                        help="Learning rate for the optimizer. Default is whatever default value is used by TF's Adam "
                             "optimizer implementation")
    parser.add_argument("-dropout_rate", type=float, required=False, default=0.5, help="Dropout rate used on feed"
                                                                                       " forward layers' inputs")
    parser.add_argument("-attention_dropout_rate", type=float, required=False, default=0.3, help="Dropout rate for"
                                                                                                 "attention.")
    parser.add_argument("-recurrent_dropout_rate", type=float, required=False, default=0.0,
                        help="Recurrent dropout rate used on RNN inputs")
    parser.add_argument("--no_pos_features", action="store_true",
                        help="POS features won't be used as extra input to the model.")
    parser.add_argument("--no_ner_indicators", action="store_true",
                        help="Named entity indicators won't be used as extra input to the model.")
    parser.add_argument("-synth_supervised_epochs", type=int, default=2,
                        help="Number of epochs to train the model in supervised mode for on the synthetically generated"
                             "dataset.")
    parser.add_argument("-org_supervised_epochs", type=int, default=18,
                        help="Number of epochs to train the model in supervised mode for on the Amazon Turk collected"
                             "dataset.")
    parser.add_argument("-nb_epochs", type=int, default=20, help="Total number of epochs to train for.", required=False)
    parser.add_argument("-beam_search_size", type=int, default=5, help="Beam search size to use during translation.")
    parser.add_argument("-checkpoint_name", type=str, required=False, default=None,
                        help="Name of a checkpoint of the model to resume from. When in translate mode, the model"
                             "will be loaded and directly used to make predictions. When in train or train_rl mode,"
                             " training will resume from the checkpoint and continue with the provided parameters.")
    parser.add_argument("--save_model", action="store_true", help="Add this flag to save checkpoints while training.")
    parser.add_argument("--mixed_data", action="store_true", help="Enables training on both the synthetic and the"
                                                                  " organic (Mturk) dataset within the same epoch. "
                                                                  "With this option activated we will run as many "
                                                                  "epochs as the sum of the number of synthetic and "
                                                                  "organic epochs.")
    parser.add_argument("--reduced_ner_indicators", action="store_true",
                        help="Whether to differentiate between the answer indicator and the other named entities from"
                             " the base question.")
    parser.add_argument("--no_base_question_encodings", action="store_true",
                        help="Use directly the base question embeddings in place of the base question encodings ("
                             "generated by the encoder RNN network)")
    parser.add_argument("--no_glove", action="store_true", help="Randomly initialize embedding matrix instead of using"
                                                                " pretrained GloVe embeddings.")
    parser.add_argument("-gpu_id", type=str, help="ID of the GPU to use.", default=None)
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    use_pos = not args.no_pos_features
    use_ner = not args.no_ner_indicators
    if args.action == "train":
        train(args)
        info("Training completed.")
    elif args.action == "preprocess":
        assert all(arg is not None for arg in (args.save_dir, args.ds_name))
        preprocess(args.preprocess_data_dir, args.save_data_dir, args.ds_name, args.voc_size,
                   args.pretrained_embeddings_path)
        info("Preprocessing completed successfully.")
    elif args.action == "translate":
        assert args.checkpoint_name is not None and args.ds_name is not None
        translate(f"{REPEAT_Q_TRAIN_CHECKPOINTS_DIR}/{args.checkpoint_name}", args)
        info("Translation completed.")
