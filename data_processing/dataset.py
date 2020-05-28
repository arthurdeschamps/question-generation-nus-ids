import random
from abc import ABC

from data_processing.class_defs import SquadExample
from defs import SQUAD_TRAIN, SQUAD_DEV, MEDQUAD_TRAIN, MEDQUAD_DEV, MEDQA_HANDMADE_FILEPATH


class Dataset(ABC):

    def __init__(self, dataset_name="squad", mode="train", data_limit=-1, ):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        self.data_limit = data_limit
        self.datapath = Dataset.determine_correct_datapath(dataset_name, mode)

    def get_split(self, first_part_size_ratio: float):
        """
        :param first_part_size_ratio: Size ratio of the first returned dataset from the original one.
        :return: A tuple (ds1, ds2) where ds1 is `first_part_size_ratio` of the original dataset
        and ds2 the rest of it.
        """
        c, a, q = self.get_dataset()
        ds = list(zip(c, a, q))
        random.shuffle(ds)
        c, a, q = zip(*ds)
        ds_size = len(c)
        first_part_size = int(first_part_size_ratio * ds_size)
        return c[:first_part_size], a[:first_part_size], q[:first_part_size], c[first_part_size:], \
               a[first_part_size:], q[first_part_size:]

    def get_dataset(self):
        raise NotImplementedError()

    @staticmethod
    def determine_correct_datapath(ds_name, mode):
        if ds_name == "squad":
            if mode == "train":
                return SQUAD_TRAIN
            elif mode == "dev":
                return SQUAD_DEV
            else:
                raise ValueError()
        elif ds_name == "medquad":
            if mode == "train":
                return MEDQUAD_TRAIN
            elif mode == "dev":
                return MEDQUAD_DEV
            else:
                raise ValueError()
        elif ds_name == "medqa_handmade":
            if mode == "test":
                return MEDQA_HANDMADE_FILEPATH
            else:
                raise ValueError()
        else:
            raise NotImplementedError()
