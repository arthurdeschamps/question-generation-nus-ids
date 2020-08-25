import ast
import json
import logging
import random
import uuid
import xml.etree.ElementTree as ET
from typing import List, Dict
import pandas as pd
import stanza
from tqdm import tqdm
from transformers import BertConfig

from data_processing.utils import array_to_string
from defs import PRETRAINED_MODELS_DIR, MEDQUAD_RAW_DIR, SQUAD_TRAIN, SQUAD_DEV
from data_processing.class_defs import SquadExample, Question, Answer, SquadMultiQAExample
import os
from data_processing.class_defs import QAExample


def read_squad_dataset(dataset_path: str, example_cls=SquadExample, limit=-1):
    """
    Loads a squad dataset (json format) from the given path.
    :param dataset_path: Path to a json formatted SQuAD dataset.
    :param example_cls: Class of the examples to parse.
    :param limit: Limit to the number of paragraphs to load.
    :return: A list of SquadExample objects.
    """
    ds = pd.read_json(dataset_path)
    ds = ds["data"][:limit]
    squad_examples = []
    logging.info("Read squad examples...")
    for i, examples in tqdm(enumerate(ds)):
        squad_examples.extend(example_cls.from_json(examples))
        if i == limit:
            break
    return squad_examples


def read_bert_config(model_dir) -> BertConfig:
    with open(f"{PRETRAINED_MODELS_DIR}/{model_dir}/bert_config.json") as config:
        parsed_config = json.load(config)
    if parsed_config is None:
        raise AssertionError(f"Could not read config at {model_dir}")
    return BertConfig(**parsed_config)


def read_medquad_raw_dataset() -> List[Dict]:
    logging.basicConfig(level=logging.INFO)

    tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
    ds = []
    nb_generate_data = 0

    for subset_dir in os.listdir(MEDQUAD_RAW_DIR):
        dirpath = f"{MEDQUAD_RAW_DIR}/{subset_dir}"
        if os.path.isdir(dirpath):
            for xml_file in os.listdir(dirpath):
                filepath = f"{dirpath}/{xml_file}"
                if os.path.isfile(filepath) and xml_file.endswith(".xml"):
                    parsed = ET.parse(filepath)
                    qa_pairs = parsed.getroot().find('QAPairs')
                    pair_tag = "QAPair"
                    q_tag = "Question"
                    a_tag = "Answer"
                    if qa_pairs is None:
                        # Some documents have XML tags although having the same structure
                        qa_pairs = parsed.getroot().find('qaPairs')
                        pair_tag = "pair"
                        q_tag = "question"
                        a_tag = "answer"
                        if qa_pairs is None:
                            logging.warning(f"No QAPairs tag in {ET.tostring(parsed.getroot())}")
                            continue
                    for qa in qa_pairs.findall(pair_tag):
                        question = qa.find(q_tag).text
                        answer = qa.find(a_tag).text
                        if not isinstance(question, str) or not isinstance(answer, str) or len(question) == 0 or \
                                len(answer) == 0:
                            logging.warning(f"Issue with QA pair: \n'{question}' \n'{answer}")
                            continue
                        question_tokens = tokenizer.process(question).sentences[0].tokens
                        paragraph = tokenizer.process(answer)
                        for i in range(0, len(paragraph.sentences), 2):
                            # Takes 2 sentences at a time
                            if i + 1 < len(paragraph.sentences):
                                tokens = paragraph.sentences[i].tokens + paragraph.sentences[i+1].tokens
                            else:
                                tokens = paragraph.sentences[i].tokens
                            answer_content = array_to_string(list(tok.text for tok in tokens))
                            question_content = array_to_string(list(tok.text for tok in question_tokens)).lower()
                            ds.append({
                                'question': question_content,
                                'answer': answer_content,
                                'sub_dataset': subset_dir,
                                'filename': xml_file
                            })
                            nb_generate_data += 1
                            if nb_generate_data % 10 == 0:
                                logging.info(f"Processed {nb_generate_data}")
    random.shuffle(ds)
    return ds


def read_qa_dataset(ds_path: str, limit=-1) -> List[QAExample]:
    ds = pd.read_csv(ds_path, sep='|', index_col=None, nrows=None if limit == -1 else limit)
    exs = []
    for _, datapoint in ds.iterrows():
        if 'sub_dataset' not in datapoint.keys() or 'filename' not in datapoint.keys():
            question_id = uuid.uuid4()
        else:
            question_id = f"{datapoint['sub_dataset']}/{datapoint['filename']}"
        exs.append(QAExample(
            question=Question(
                question=datapoint['question'],
                question_id=question_id
            ),
            answer=Answer(text=datapoint['answer'], answer_start=0)
        ))
    return exs


def next_chunk(file_reader):
    next_content = None
    next_line = file_reader.readline()
    while next_line and next_line == "\n":
        next_line = file_reader.readline()
    while next_line and next_line != "\n":
        next_line = next_line.strip()
        if next_content is None:
            next_content = [next_line]
        else:
            next_content.append(next_line)
        next_line = file_reader.readline()
    return next_content, next_line


def read_squad_facts_files(facts_dirpath):
    fact_dataset = {}
    assert os.path.isdir(facts_dirpath)
    print(f"Parsing {facts_dirpath}...")
    for filename in tqdm(os.listdir(facts_dirpath)):
        if "facts" in filename:
            passage_facts = []
            passage_id = int(filename.replace("facts.", "").replace(".txt", ""))
            with open(os.path.join(facts_dirpath, filename), mode='r') as f:
                last_line = ""
                while last_line is not None:
                    next_content, last_line = next_chunk(f)
                    if next_content is None:
                        break
                    fact_id = next_content[0]
                    g_type = next_content[1]
                    g_name = next_content[2]
                    g_description = next_content[3]
                    g_article_text = next_content[4]

                    fields_with_keys_and_renamed_keys = zip(
                        [fact_id, g_type, g_name, g_description, g_article_text],
                        ["FACTID", "GKGTYPE", "GKGNAME", "GKGDESC", "GKGARTTEXT"],
                        ["fact_id", "type", "name", "description", "text"]
                    )
                    fact = {}
                    for field, key, renamed_key in fields_with_keys_and_renamed_keys:
                        assert key in field
                        fact[renamed_key] = field.replace(key, "").strip()
                    passage_facts.append(fact)

            fact_dataset[passage_id] = passage_facts
    return fact_dataset


def read_squad_rewrites_files(rewrites_dirpath):
    assert os.path.isdir(rewrites_dirpath)
    rewrites = {}
    print(f"Parsing {rewrites_dirpath}...")
    for filename in tqdm(os.listdir(rewrites_dirpath)):
        if "qw" in filename and "old" not in filename:
            passage_rewrites = []
            passage_id = int(filename.replace("qw.", "").replace(".list", ""))
            with open(os.path.join(rewrites_dirpath, filename), mode='r') as f:
                next_line = ""
                while next_line is not None:
                    next_content, next_line = next_chunk(f)
                    if next_content is None:
                        break
                    base_question = next_content[0]
                    # Some questions have more than one rewrites, we create one example per rewrite
                    rewritten_questions = next_content[1:]
                    for rewritten_question in rewritten_questions:
                        passage_rewrites.append({
                            "base_question": base_question,
                            "rephrased": rewritten_question
                        })
            rewrites[passage_id] = passage_rewrites
    return rewrites


def read_squad_qmap_files(qmap_dirpath):
    assert os.path.isdir(qmap_dirpath)
    question_to_facts = {}
    for filename in tqdm(os.listdir(qmap_dirpath)):
        if "qmap" in filename:
            passage_id = int(filename.replace("qmap.", "").replace(".txt", ""))
            q_maps = {}
            with open(os.path.join(qmap_dirpath, filename), mode='r') as f:
                next_line = ""
                while next_line is not None:
                    next_content, next_line = next_chunk(f)
                    if next_content is None:
                        break
                    question = next_content[0].replace("QUESTION ", "")
                    fact_ids = ast.literal_eval(next_content[1].replace("facts ", ""))
                    q_maps[question] = fact_ids
            question_to_facts[passage_id] = q_maps
    return question_to_facts


def read_squad_rewrites_human_made(dirpath, use_triples=False, mapped_triples=False):
    with open(dirpath, mode="r") as f:
        data = json.load(f)
        questions_to_facts = {}
        questions_to_rewrites = {}
        ds = []

        if mapped_triples:
            for datapoint in data:
                for rf_pair in datapoint["rfpairs"]:
                    ds.append({
                        "base_question": datapoint["question"],
                        "facts": [rf_pair["triple"]],
                        "target": rf_pair["rewrite"]
                    })
        else:
            for datapoint in data:
                q = datapoint["question"]
                fact = datapoint["triples"] if use_triples else datapoint["fact"]
                rewrites = datapoint["rewrites"]
                if q in questions_to_facts:
                    questions_to_facts[q].append(fact)
                    questions_to_rewrites[q].extend(rewrites)
                else:
                    questions_to_facts[q] = [fact]
                    questions_to_rewrites[q] = rewrites
            for q in questions_to_facts.keys():
                for target in questions_to_rewrites[q]:
                    ds.append({
                        "base_question": q,
                        "facts": questions_to_facts[q],
                        "target": target
                    })
    return ds


def get_squad_question_to_answers_map():
    qa_map = {}

    def _read_qas(path):
        ds = pd.read_json(path)["data"]
        for article in ds:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    qa_map[qa["question"].strip()] = [answer["text"] for answer in qa["answers"]]
    _read_qas(SQUAD_DEV)
    _read_qas(SQUAD_TRAIN)
    return qa_map
