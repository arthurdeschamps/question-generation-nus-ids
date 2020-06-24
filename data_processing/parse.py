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
from defs import PRETRAINED_MODELS_DIR, MEDQUAD_RAW_DIR
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
