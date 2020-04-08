import pandas as pd
from defs import SQUAD_DEV, SQUAD_TRAIN
from data_utils.class_defs import SquadExample


def parse_squad_example(raw_example):
    return SquadExample.from_json(raw_example)


def read_square_dataset(dataset_path: str):
    ds = pd.read_json(dataset_path)
    ds = ds["data"][:10]
    ds = list(parse_squad_example(example) for example in ds)
    return ds
