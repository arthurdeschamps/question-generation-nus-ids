import random
from abc import ABC

from data_processing.class_defs import SquadExample, QAExample, SquadMultiQAExample
from data_processing.parse import read_squad_dataset, read_qa_dataset
from defs import SQUAD_TRAIN, SQUAD_DEV, MEDQUAD_TRAIN, MEDQUAD_DEV, MEDQA_HANDMADE_FILEPATH


class Dataset(ABC):

    def __init__(self, dataset_name="squad", mode="train", data_limit=-1, break_up_paragraphs=True):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        self.break_up_paragraphs = break_up_paragraphs
        self.data_limit = data_limit
        self.datapath = Dataset.determine_correct_datapath(dataset_name, mode)
        self.ds, self.datatype = None, None
        self.read_dataset()

    def get_split(self, first_part_size_ratio: float):
        """
        :param first_part_size_ratio: Size ratio of the first returned dataset from the original one.
        :return: A tuple (ds1, ds2) where ds1 is `first_part_size_ratio` of the original dataset
        and ds2 the rest of it.
        """
        # c, a, q = self.get_dataset()
        # ds = list(zip(c, a, q))
        # random.shuffle(ds)
        # c, a, q = zip(*ds)
        # ds_size = len(c)
        # first_part_size = int(first_part_size_ratio * ds_size)
        # return c[:first_part_size], a[:first_part_size], q[:first_part_size], c[first_part_size:], \
        #        a[first_part_size:], q[first_part_size:]
        elems = self.get_dataset()
        ds = list(zip(*elems))
        random.shuffle(ds)
        first_part_size = int(first_part_size_ratio * len(ds))
        ds1 = list(zip(*ds[:first_part_size]))
        ds2 = list(zip(*ds[first_part_size:]))
        return (*ds1, *ds2)

    def read_dataset(self):
        # Default implementation
        if self.dataset_name == "squad":
            self.datatype = SquadExample if self.break_up_paragraphs else SquadMultiQAExample
            self.ds = read_squad_dataset(self.datapath, example_cls=self.datatype, limit=self.data_limit)
        elif self.dataset_name in ("medquad", "medqa_handmade"):
            self.datatype = QAExample
            self.ds = read_qa_dataset(self.datapath, limit=self.data_limit)
        else:
            raise NotImplementedError()

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
            raise NotImplementedError(ds_name)
