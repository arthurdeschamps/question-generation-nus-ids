import json
import os
import pathlib
import shutil
import subprocess
from logging import info, warning, debug

import pandas as pd
import nltk
import stanza
from tqdm import tqdm

from data_processing.mpqg_dataset import MPQGDataset
from data_processing.parse import read_medquad_raw_dataset, read_squad_dataset, read_squad_facts_files, \
    read_squad_rewrites_files, read_squad_qmap_files
from data_processing.utils import array_to_string
from defs import NQG_MODEL_DIR, NQG_DATA_HOME, MEDQUAD_DIR, MEDQUAD_DEV, MEDQUAD_TRAIN, \
    MEDQA_HANDMADE_FILEPATH, MEDQA_HANDMADE_RAW_DATASET_FILEPATH, HOTPOT_QA_DEV_JSON, \
    HOTPOT_QA_DEV_TARGETS_PATH, ASS2S_PROCESSED_MPQG_DATA, SQUAD_TRAIN, SQUAD_DEV, \
    REPEAT_Q_RAW_DATASETS, SQUAD_FACTS_TRAIN, SQUAD_FACTS_DEV, SQUAD_REWRITES_DEV, SQUAD_REWRITES_TRAIN
from data_processing.nqg_dataset import NQGDataset
from data_processing.pre_processing import NQGDataPreprocessor
import numpy as np


def generate_ass2s_mpqg_features(ds_name):
    # The provided dataset for ASs2s has the following format:
    # 'textN': unused
    # 'annotation1' corresponds to the features of the context
    # 'annotation2' corresponds to the features of the question
    # 'annotation3' corresponds to the features of the answer
    # 'annotationN' has fields: 'raw_text', 'toks' (tokenized, cases remain untouched for answers and questions and
    # are lower-cased for contexts), 'POSs': ununsed,
    # 'positions': unused, 'NERs': tokens replaced with corresponding NER tag or O
    if ds_name == "squad":
        ds_train = MPQGDataset(mode="train")
        ds_test = MPQGDataset(mode="dev")

        c_dev, a_dev, q_dev, c_test, a_test, q_test = ds_test.get_split(0.5)

        def _generate_features(contexts, answers, questions, ds_type):
            features = []
            for context, answer, question in zip(contexts, answers, questions):
                def make_features(document, is_context=False):
                    if is_context:
                        tokens = " ".join([" ".join([token.text.lower() for token in sentence.tokens])
                                           for sentence in document.sentences])
                    else:
                        tokens = document.text
                    return {
                        'toks': tokens,
                        'NERs': [{
                            "entity": entity.text.lower() if is_context else entity.text,
                            "ent_type": entity.type
                        } for entity in document.entities]
                    }

                features.append({
                    'annotation1': make_features(context, is_context=True),
                    'annotation2': question,
                    'annotation3': answer
                })
            if not os.path.exists(ASS2S_PROCESSED_MPQG_DATA):
                pathlib.Path(ASS2S_PROCESSED_MPQG_DATA).mkdir(parents=True, exist_ok=True)
            with open(f"{ASS2S_PROCESSED_MPQG_DATA}/{ds_type}_sent_pre.json", mode='w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=2)

        _generate_features(*ds_train.get_dataset(), "train")
        _generate_features(c_dev, a_dev, q_dev, "dev")
        _generate_features(c_test, a_test, q_test, "test")
    else:
        raise NotImplementedError(f"Preprocessing for dataset {ds_name} not implemented yet.")


def generate_vocabulary_files(dataset_path, bio_path, vocab_size):
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
        f"{bio_path}/train/data.txt.bio",
        f"{bio_path}/train/bio.vocab.txt",
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


def generate_nqg_features(mode: str, dataset_name: str, enhanced_ner: bool = False):
    """
    Generates the feature files for the NQG model.
    :param mode: "Train", "Dev" or "Test". Will define which files will be used to generate the features.
    :param dataset_name: "squad", "medquad", "medqa_handmade", ...
    :param enhanced_ner: If the most up-to-date NER tags should be used, or the ones used by NQG.
    """
    if mode not in ("train", "dev", "test"):
        raise ValueError(f"mode should be one of 'train', 'dev' or 'test'")
    if dataset_name not in ("squad", "medquad", "medqa_handmade"):
        raise ValueError("dataset_name argument not recognized")

    ds = NQGDataset(dataset_name=dataset_name, mode=mode)
    if mode == 'dev':
        # Need to split into dev/test
        segments = ['dev', 'test']
        c_dev, a_dev, q_dev, c_test, a_test, q_test = ds.get_split(0.5)
        data = [(c_dev, a_dev, q_dev), (c_test, a_test, q_test)]
    elif mode == 'train':
        segments = ['train']
        data = [ds.get_dataset()]
    else:
        segments = ['test']
        data = [ds.get_dataset()]

    if enhanced_ner:
        dataset_name += "_+NER"

    for segment_type, segment_data in zip(segments, data):
        data_preprocessor = NQGDataPreprocessor(segment_data[0])
        answer_starts = np.array(list(answer.start_index for answer in segment_data[1]))
        answer_lengths = np.array(list(answer.nb_words for answer in segment_data[1]))
        ner = data_preprocessor.create_ner_sequences(enhanced_ner)
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


def generate_medquad_dataset():
    ds = read_medquad_raw_dataset()
    train_size = int(0.8 * len(ds))
    train = ds[:train_size]
    dev = ds[train_size:]
    if os.path.exists(MEDQUAD_DIR):
        shutil.rmtree(MEDQUAD_DIR)
    os.mkdir(MEDQUAD_DIR)
    dev_df = pd.DataFrame(dev)
    dev_df.to_csv(MEDQUAD_DEV, sep='|', index=False)
    train_df = pd.DataFrame(train)
    train_df.to_csv(MEDQUAD_TRAIN, sep='|', index=False)


def generate_medqa_handmade_dataset(ds_path):
    ds_raw = pd.read_csv(ds_path, sep='|')
    tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
    ds = []
    for question, answer in zip(ds_raw['question'], ds_raw['answer']):
        question_tokens = tokenizer.process(question).sentences[0].tokens
        paragraph = tokenizer.process(answer)
        for i in range(0, len(paragraph.sentences), 2):
            # Takes 2 sentences at a time
            if i + 1 < len(paragraph.sentences):
                tokens = paragraph.sentences[i].tokens + paragraph.sentences[i + 1].tokens
            else:
                tokens = paragraph.sentences[i].tokens
            answer_content = array_to_string(list(tok.text for tok in tokens))
            question_content = array_to_string(list(tok.text for tok in question_tokens)).lower()
            ds.append({
                'question': question_content,
                'answer': answer_content,
            })
    pd.DataFrame(ds).to_csv(MEDQA_HANDMADE_FILEPATH, index=False, sep="|")


def generate_bio_features(mode: str, ds_name: str, answer_mode: str):
    assert answer_mode in ("none", "guess")
    source_dir = f"{NQG_DATA_HOME}/{ds_name}/{mode}"
    target_dir = f"{NQG_DATA_HOME}/{ds_name}"
    if answer_mode == "none":
        target_dir += "_NA"
    else:
        target_dir += "_GA"
    assert os.path.exists(source_dir) and os.path.isdir(source_dir)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(f"{target_dir}/{mode}"):
        os.mkdir(f"{target_dir}/{mode}")

    if answer_mode == "none":
        bios = []
        source_passages = np.loadtxt(f"{source_dir}/data.txt.source.txt", dtype=str, delimiter='\n', comments=None)
        for passage in source_passages:
            bio = ["I" for _ in range(len(passage.split(" ")))]
            bio[0] = "B"
            bios.append(array_to_string(bio))

    if answer_mode == "guess":
        corpus_named_entities = np.loadtxt(f"{source_dir}/data.txt.ner", dtype=str, delimiter='\n', comments=None)
        corpus_pos_tags = np.loadtxt(f"{source_dir}/data.txt.pos", dtype=str, delimiter='\n', comments=None)
        bios = []
        for named_entities, pos_tags in zip(corpus_named_entities, corpus_pos_tags):
            named_entities = named_entities.split(' ')
            longest_ne_seq = []
            current_seq_length = []
            for i in range(len(named_entities)):
                ne = named_entities[i]
                if ne != 'O':
                    current_seq_length.append(i)
                else:
                    if len(current_seq_length) > len(longest_ne_seq):
                        longest_ne_seq = current_seq_length
                    current_seq_length = []
            if len(longest_ne_seq) == 0:
                # No named entities in this passage so we take the first noun phrase
                pos_tags = pos_tags.split(' ')
                try:
                    bio = ["O" for _ in range(len(pos_tags))]
                    i = 0
                    while i < len(pos_tags):
                        if pos_tags[i].startswith("NN"):
                            bio[i] = "B"
                            i += 1
                            break
                        i += 1
                    while i < len(pos_tags) and pos_tags[i].startswith("NN"):
                        bio[i] = "I"
                        i += 1
                except ValueError:
                    # No noun either, we fallback on using the full passage as the answer
                    bio = array_to_string(['B'] + ['I' for _ in range(len(named_entities) - 1)])
            else:
                bio = ['O' for _ in range(len(named_entities))]
                bio[longest_ne_seq[0]] = "B"
                for i in longest_ne_seq[1:]:
                    bio[i] = "I"
            bios.append(array_to_string(bio))

    np.savetxt(f"{target_dir}/{mode}/data.txt.bio", bios, fmt="%s")


def generate_hotpot_targets(json_data_path, savepath):
    with open(json_data_path) as f:
        hotpot_data = json.load(f)
        targets = []
        for example in hotpot_data:
            targets.append(" ".join(nltk.word_tokenize(example['question'].trim().lower())))
    np.savetxt(savepath, targets, fmt="%s", delimiter="\n", comments=None)


def generate_repeat_q_squad_raw():
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    #tokenize = stanza.Pipeline(lang='en', processors='tokenize')
    tokenize = nltk.word_tokenize

    facts_train = read_squad_facts_files(facts_dirpath=SQUAD_FACTS_TRAIN)
    rewrites_train = read_squad_rewrites_files(rewrites_dirpath=SQUAD_REWRITES_TRAIN)
    question_to_facts_map_train = read_squad_qmap_files(qmap_dirpath=SQUAD_FACTS_TRAIN)
    facts_dev = read_squad_facts_files(facts_dirpath=SQUAD_FACTS_DEV)
    rewrites_dev = read_squad_rewrites_files(rewrites_dirpath=SQUAD_REWRITES_DEV)
    question_to_facts_map_dev = read_squad_qmap_files(qmap_dirpath=SQUAD_FACTS_DEV)

    ds = []

    def _make_example(_base_question, _rewritten_question, _facts, _passage_id):
        # Create example placeholders and filter out irrelevant question words for future word matching
        def _tokenize(sentence):
            return " ".join(tokenize(sentence)).lower()
        tokenized_question = _tokenize(base_question)
        target = _tokenize(_rewritten_question)
        tokenized_facts = [_tokenize(fact["text"]) for fact in _facts]
        example = {
            "base_question": tokenized_question,
            "facts": tokenized_facts,
            "target": target,
            "passage_id": _passage_id
        }
        return example

    for question_to_facts_map, rewrites, facts, ds_type in [
        (question_to_facts_map_dev, rewrites_dev, facts_dev, "test"),
        (question_to_facts_map_train, rewrites_train, facts_train, "train")
    ]:
        for passage_id in tqdm(list(question_to_facts_map.keys())):
            questions = rewrites[passage_id]
            question_to_facts = question_to_facts_map[passage_id]
            for question in questions:
                base_question = question["base_question"]
                rephrased = question["rephrased"]
                if base_question in question_to_facts:
                    fact_ids = question_to_facts[base_question]
                    passage_facts = [facts[passage_id][fact_id] for fact_id in fact_ids]
                    ds.append(_make_example(base_question, rephrased, passage_facts, passage_id))
        with open(f"{REPEAT_Q_RAW_DATASETS}/squad_{ds_type}.json", mode='w') as f:
            json.dump(ds, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    args = parser.parse_args()

    if args.dataset_name == 'nqg_squad':
        generate_nqg_features('dev', 'squad')
        generate_nqg_features('train', 'squad')
    elif args.dataset_name == "nqg_squad_ner":
        generate_nqg_features('dev', 'squad', enhanced_ner=True)
        generate_nqg_features('train', 'squad', enhanced_ner=True)
    elif args.dataset_name == "nqg_squad_ga":
        generate_bio_features('dev', 'squad', 'guess')
        generate_bio_features('test', 'squad', 'guess')
        generate_bio_features('train', 'squad', 'guess')
    elif args.dataset_name == "nqg_squad_na":
        generate_bio_features('dev', 'squad', 'none')
        generate_bio_features('test', 'squad', 'none')
        generate_bio_features('train', 'squad', 'none')
    elif args.dataset_name == 'nqg_medquad':
        generate_nqg_features('dev', 'medquad')
        generate_nqg_features('train', 'medquad')
    elif args.dataset_name == "nqg_medqa_handmade":
        filepath = MEDQA_HANDMADE_RAW_DATASET_FILEPATH
        if not os.path.exists(filepath):
            generate_medqa_handmade_dataset(filepath)
        generate_nqg_features('test', 'medqa_handmade')
    elif args.dataset_name == "hotpotqa_dev_targets":
        dev_json = HOTPOT_QA_DEV_JSON
        result_save_path = HOTPOT_QA_DEV_TARGETS_PATH
        generate_hotpot_targets(dev_json, result_save_path)
    elif args.dataset_name == "repeat_q_squad":
        generate_repeat_q_squad_raw()
        info("Raw SQuAD dataset for RepeatQ generated.")
    else:
        raise ValueError("Non-existing dataset type")
    print("Done")
