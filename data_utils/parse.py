import json

import pandas as pd
from transformers import BertConfig
from defs import PRETRAINED_MODELS_DIR
from data_utils.class_defs import SquadExample


def parse_squad_example(raw_example):
    return SquadExample.from_json(raw_example)


def read_squad_dataset(dataset_path: str, limit=-1):
    """
    Loads a squad dataset (json format) from the given path.
    :param dataset_path: Path to a json formatted SQuAD dataset.
    :param limit: Limit to the number of paragraphs to load.
    :return: A list of SquadExample objects.
    """
    ds = pd.read_json(dataset_path)
    ds = ds["data"][:limit]
    ds = list(parsed for parsed in (parse_squad_example(example) for example in ds) if parsed is not None)
    return ds


def read_bert_config(model_dir) -> BertConfig:
    with open(f"{PRETRAINED_MODELS_DIR}/{model_dir}/bert_config.json") as config:
        parsed_config = json.load(config)
    if parsed_config is None:
        raise AssertionError(f"Could not read config at {model_dir}")
    return BertConfig(**parsed_config)
