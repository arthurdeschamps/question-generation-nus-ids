import json
import os
import pathlib
import shutil
import subprocess
from logging import info

import pandas as pd
import nltk
import stanza
from tqdm import tqdm

from data_processing.class_defs import RepeatQExample
from data_processing.mpqg_dataset import MPQGDataset
from data_processing.parse import read_medquad_raw_dataset, read_squad_rewrites, get_squad_question_to_answers_map
from data_processing.utils import array_to_string
from defs import NQG_MODEL_DIR, NQG_DATA_HOME, MEDQUAD_DIR, MEDQUAD_DEV, MEDQUAD_TRAIN, \
    MEDQA_HANDMADE_FILEPATH, MEDQA_HANDMADE_RAW_DATASET_FILEPATH, HOTPOT_QA_DEV_JSON, \
    HOTPOT_QA_DEV_TARGETS_PATH, REPEAT_Q_RAW_DATASETS, ASS2S_PROCESSED_SQUAD_MPQG_DATA, REPEAT_Q_SQUAD_DATA_DIR, \
    NQG_SQUAD_DATASET, SQUAD_REWRITE_MTURK_DIR, SQUAD_REWRITES_SYNTHETIC_JSON, ASS2S_PROCESSED_DIR, REPEAT_Q_DATA_DIR

from data_processing.nqg_dataset import NQGDataset
from data_processing.pre_processing import NQGDataPreprocessor
import numpy as np


def generate_ass2s_mpqg_features(ds_name, regular_squad):

    if not os.path.exists(ASS2S_PROCESSED_SQUAD_MPQG_DATA):
        pathlib.Path(ASS2S_PROCESSED_SQUAD_MPQG_DATA).mkdir(parents=True, exist_ok=True)

    def save_data(dir_name, ds_type, data):
        if not os.path.exists(dir_name):
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(f"{dir_name}/{ds_type}_sent_pre.json", mode='w+', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # The provided dataset for ASs2s has the following format:
    # 'textN': unused
    # 'annotation1' corresponds to the features of the context
    # 'annotation2' corresponds to the features of the question
    # 'annotation3' corresponds to the features of the answer
    # 'annotationN' has fields: 'raw_text', 'toks' (tokenized, cases remain untouched for answers and questions and
    # are lower-cased for contexts), 'POSs': ununsed,
    # 'positions': unused, 'NERs': tokens replaced with corresponding NER tag or O
    if regular_squad:
        ds_train = MPQGDataset(mode="train")
        ds_test = MPQGDataset(mode="dev")
        c_dev, a_dev, q_dev, c_test, a_test, q_test = ds_test.get_split(0.5)

        def _generate_features(contexts, answers, questions, ds_type):
            features = []
            for context, answer, question in zip(contexts, answers, questions):
                def make_features(document):
                    tokens = " ".join([" ".join([token.text.lower() for token in sentence.tokens])
                                       for sentence in document.sentences])
                    return {
                        'toks': tokens,
                        'NERs': [{
                            "entity": entity.text.lower(),
                            "ent_type": entity.type
                        } for entity in document.entities]
                    }

                features.append({
                    'annotation1': make_features(context),
                    'annotation2': question,
                    'annotation3': answer
                })
            save_data(ASS2S_PROCESSED_SQUAD_MPQG_DATA, ds_type, features)

        _generate_features(*ds_train.get_dataset(), "train")
        _generate_features(c_dev, a_dev, q_dev, "dev")
        _generate_features(c_test, a_test, q_test, "test")
    else:
        use_triples = "triples" in ds_name
        mapped_triples = "mapped" in ds_name
        processed_mpqg_data_dir = f"{ASS2S_PROCESSED_DIR}/{ds_name}"
        tokenize = stanza.Pipeline(processors="tokenize")
        modes = ("test", "dev", "train")
        question_to_answers = {}
        if not use_triples:
            for mode in modes:
                with open(f"{ASS2S_PROCESSED_SQUAD_MPQG_DATA}/{mode}_sent_pre.json") as data_file:
                    for data in tqdm(json.load(data_file)):
                        base_question = " ".join(word.text.lower() for word in tokenize(data["annotation2"]).iter_words())
                        question_to_answers[base_question] = data["annotation3"].lower()
        for mode in modes:
            data_dir = f"{REPEAT_Q_DATA_DIR}/{ds_name}"
            with open(f"{data_dir}/{mode}.data.json") as data_file:
                examples = json.load(data_file)

            synthetic_examples = []
            organic_examples = []

            def get_ners(ners, base):
                mapped_ners = []
                current_word = []
                current_ner = None
                for ner, base_word in zip(ners.split(), base.split()):
                    if ner == "O":
                        if len(current_word) > 0:
                            mapped_ners.append({"entity": " ".join(current_word), "ent_type": current_ner})
                            current_word = []
                    else:
                        current_ner = ner
                        current_word.append(base_word)
                return mapped_ners

            for example in tqdm(examples):
                if mode == "test" and example["is_synthetic"]:
                    continue
                bq = example["base_question"]
                bq_ners = get_ners(example["base_question_ner"], bq)
                facts = example["facts"]
                facts_ners = [get_ners(fact_ner, fact) for fact_ner, fact in zip(example["facts_ner"], facts)]
                flattened_facts_ners = [ner for fact_ners in facts_ners for ner in fact_ners ]
                context = bq + " gsdassep " + " ".join(facts)
                context_ners = bq_ners + [{"entity": "gsdassep", "ent_type": "GSDASSEP"}] + flattened_facts_ners
                if (use_triples and mapped_triples) or bq not in question_to_answers:
                    answer = facts_ners[0][0]["entity"] if len(facts_ners[0]) > 0 else facts[0]
                    for fact_ner in facts_ners[0]:
                        if fact_ner["entity"] in facts[0]:
                            answer = fact_ner["entity"]
                            break
                else:
                    original_answer = question_to_answers[bq]
                    found = False
                    for fact_ners in facts_ners:
                        if not found:
                            for fact_ner in fact_ners:
                                if fact_ner in original_answer or fact_ner in bq_ners:
                                    answer = fact_ner
                                    found = True
                                    break
                    if not found:
                        print(f"No matching entity found for: \"{original_answer}. {base_question}\"")
                        continue
                feature = {
                    "annotation1": {
                        "toks": context,
                        "NERs": context_ners
                    },
                    "annotation2": example["target"],
                    "annotation3": answer
                }
                if example["is_synthetic"]:
                    synthetic_examples.append(feature)
                else:
                    organic_examples.append(feature)

            save_data(processed_mpqg_data_dir + "/mpqg_data", mode, synthetic_examples + organic_examples)
            save_data(processed_mpqg_data_dir + "_mturk/mpqg_data", mode, organic_examples)


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
    if dataset_name not in ("squad", "medquad", "medqa_handmade", "hotpotqa"):
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


def generate_repeat_q_squad_raw(use_triples: bool, mapped_triples: bool):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,ner')

    question_to_answers_map = get_squad_question_to_answers_map()

    def _get_tokens(words):
        return " ".join([w.text.lower() for w in words])

    def _get_pos_sequence(words):
        return " ".join([w.xpos for w in words])

    def _get_tags(words, sought_after_tokens, beg_tag, inside_tag, tag_list=None):
        if tag_list is None:
            tags = ["O" for _ in range(len(words))]
        else:
            tags = tag_list
        for i in range(len(words)):
            if words[i].text == sought_after_tokens[0]:
                complete_match = True
                for j in range(len(sought_after_tokens)):
                    if i+j >= len(words) or sought_after_tokens[j] != words[i+j].text:
                        complete_match = False
                        break
                if complete_match:
                    tags[i] = beg_tag
                    for j in range(i+1, i+len(sought_after_tokens)):
                        tags[j] = inside_tag
        return tags

    def _get_entity_tags(question_doc, answers, facts_sentences):
        q_entities = [ent.text for ent in question_doc.entities]
        q_ent_tags = ["O" for _ in range(question_doc.num_words)]
        facts_ent_tags = [["O" for _ in range(len(facts_sentences[i].words))] for i in range(len(facts_sentences))]
        for q_entity in q_entities:
            q_entity_toks = q_entity.split()
            # Marks named entities in question
            q_ent_tags = _get_tags(list(question_doc.iter_words()), q_entity_toks, beg_tag="BN", inside_tag="IN",
                                   tag_list=q_ent_tags)
            # Marks NEs from the question in facts
            facts_ent_tags = [_get_tags(facts_sentences[i].words, q_entity_toks, "BN", "IN", facts_ent_tags[i])
                              for i in range(len(facts_sentences))]
        # Overwrites tags with answer tags if any in facts
        for answer in answers:
            answer_tokens = answer.lower().split()
            facts_ent_tags = [_get_tags(facts_sentences[i].words, answer_tokens, "BA", "IA", facts_ent_tags[i])
                              for i in range(len(facts_sentences))]
        return " ".join(q_ent_tags), [" ".join(f) for f in facts_ent_tags]

    def _get_cases(words):
        return " ".join(["UP" if word.text[0].isupper() else "LOW" for word in words])

    def _make_example(_base_question, _rewritten_question, _facts, _answers, _is_synthetic):
        try:
            # Create example placeholders and filter out irrelevant question words for future word matching
            analyzed_question = nlp(_base_question)
            analyzed_target = nlp(_rewritten_question)
            if use_triples:
                if mapped_triples:
                    analyzed_facts = [nlp(triple).sentences[0] for triple in _facts]
                else:
                    analyzed_facts = [nlp(triple).sentences[0] for fact in _facts for triple in fact]
            else:
                analyzed_facts = [fact_sentence for fact in _facts for fact_sentence in nlp(fact).sentences]
            base_question_entity_tags, facts_entity_tags = _get_entity_tags(analyzed_question, _answers, analyzed_facts)
            question_words = list(analyzed_question.iter_words())
            example = {
                "base_question": _get_tokens(question_words),
                "base_question_pos_tags": _get_pos_sequence(question_words),
                "base_question_entity_tags": base_question_entity_tags,
                "base_question_letter_cases": _get_cases(question_words),
                "base_question_ner": NQGDataPreprocessor.create_ner_sequence(True, question_words),
                "facts": [_get_tokens(fact.words) for fact in analyzed_facts],
                "facts_entity_tags": facts_entity_tags,
                "facts_pos_tags": [_get_pos_sequence(fact.words) for fact in analyzed_facts],
                "facts_letter_cases": [_get_cases(sentence.words) for sentence in analyzed_facts],
                "facts_ner": [NQGDataPreprocessor.create_ner_sequence(True, fact.words)
                              for fact in analyzed_facts],
                "target": _get_tokens(analyzed_target.iter_words()),
                "is_synthetic": _is_synthetic
            }
            return example
        except Exception as e:
            print(e)
            print("Question:")
            print(_base_question)
            print("Target:")
            print(_rewritten_question)
            print("Facts:")
            [print(f) for f in _facts]
        return None

    for mode in ("test", "dev", "train"):
        ds = []
        orga_filepath = f"{SQUAD_REWRITE_MTURK_DIR}/{mode}.json"
        examples = {
            "organic": read_squad_rewrites(
                dataset_path=orga_filepath, use_triples=use_triples, mapped_triples=mapped_triples
            ),
            "synthetic": read_squad_rewrites(
               SQUAD_REWRITES_SYNTHETIC_JSON, use_triples=use_triples, mapped_triples=mapped_triples
            ) if mode == "train" else []
        }

        for data_type in ("organic", "synthetic"):
            for example in tqdm(examples[data_type]):
                answers = question_to_answers_map[example["base_question"].strip()]
                ex = _make_example(
                    _base_question=example["base_question"],
                    _rewritten_question=example["target"],
                    _facts=example["facts"],
                    _answers=answers,
                    _is_synthetic=data_type == "synthetic"
                )
                if ex is not None:
                    ds.append(ex)

        if not os.path.exists(REPEAT_Q_RAW_DATASETS):
            os.mkdir(REPEAT_Q_RAW_DATASETS)
        ds_filename = f"{REPEAT_Q_RAW_DATASETS}/squad"
        if mapped_triples:
            ds_filename = f"{ds_filename}_mapped"
        if use_triples:
            ds_filename = f"{ds_filename}_triples"
        ds_filename = f"{ds_filename}_{mode}.json"
        with open(ds_filename, mode='w') as f:
            json.dump(ds, f, indent=4)


def repeat_q_to_nqg_squad(organic_only):
    data_dir = REPEAT_Q_SQUAD_DATA_DIR
    target_dir = f"{NQG_SQUAD_DATASET}_repeat_q"
    if organic_only:
        target_dir = f"{target_dir}_mturk_only"

    os.mkdir(target_dir)

    def _read_data(mode):
        with open(f"{data_dir}/{mode}.data.json", mode='r') as f:
            ds = RepeatQExample.from_json(json.load(f))
            return ds

    for mode in ("train", "dev", "test"):
        sep_tok = "GSDASSEP"
        bios, ners, poss, sources, targets, cases = [], [], [], [], [], []
        for data in _read_data(mode):
            if organic_only and data.is_synthetic_data:
                continue
            sources.append(f"{data.base_question} {sep_tok.lower()} {' '.join([fact for fact in data.facts])}")
            targets.append(data.rephrased_question)
            ners.append(
                f"{data.base_question_features.ner} {sep_tok} {' '.join([feat.ner for feat in data.facts_features])}")
            poss.append(
                f"{data.base_question_features.pos_tags} {sep_tok} {' '.join([feat.pos_tags for feat in data.facts_features])}")
            cases.append(
                f"{data.base_question_features.letter_cases} {sep_tok} {' '.join([feat.letter_cases for feat in data.facts_features])}")
            bios.append(
                f"{data.base_question_features.entity_tags} {sep_tok} {' '.join([feat.entity_tags for feat in data.facts_features])}")
            if not(len(ners[-1].split()) == len(poss[-1].split()) == len(cases[-1].split()) == len(bios[-1].split()) == len(sources[-1].split())):
                exit(-1)

        save_dir_name = f"{target_dir}/{mode}"
        os.mkdir(save_dir_name)
        for data_name, content in (("source.txt", sources), ("target.txt", targets), ("bio", bios), ("case", cases),
                                   ("ner", ners), ("pos", poss)):
            fname = f"{save_dir_name}/data.txt.{data_name}"
            np.savetxt(fname, content, fmt="%s")


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
    elif args.dataset_name == "nqg_hotpotqa":
        generate_nqg_features('dev', 'hotpotqa')
        generate_nqg_features('train', 'hotpotqa')
    elif "squad_repeat_q" in args.dataset_name:
        use_triples = False
        mapped_triples = False
        if "triples" in args.dataset_name:
            use_triples = True
        if "mapped" in args.dataset_name:
            mapped_triples = True
        generate_repeat_q_squad_raw(use_triples=use_triples, mapped_triples=mapped_triples)
        info(f"Raw SQuAD dataset for {args.dataset_name} generated.")
    elif "nqg_repeat_q_squad" in args.dataset_name:
        organic_only = False
        if "mturk_only" in args.dataset_name:
            organic_only = True
        repeat_q_to_nqg_squad(organic_only)
        info("RepeatQ SQuAD for NQG generated.")
    else:
        raise ValueError("Non-existing dataset type")
    print("Done")
