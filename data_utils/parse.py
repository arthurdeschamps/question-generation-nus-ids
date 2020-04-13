import pandas as pd
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
    ds = list(parse_squad_example(example) for example in ds)
    return ds
