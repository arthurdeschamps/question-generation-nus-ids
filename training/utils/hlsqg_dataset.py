from logging import info
from typing import List

from data_processing.class_defs import SquadExample
from data_processing.nqg_dataset import NQGDataset
from data_processing.parse import read_squad_dataset
from data_processing.pre_processing import pad_data
import tensorflow as tf

from defs import SQUAD_DEV, SQUAD_TRAIN
import numpy as np
from defs import NQG_SQUAD_DATASET


class HlsqgDataset:
    """
    HLSQG dataset manager.
    """

    def __init__(self,
                 max_sequence_length,
                 max_generated_question_length,
                 embedder,
                 nb_epochs,
                 batch_size,
                 limit_train_data,
                 limit_dev_data):
        super(HlsqgDataset, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.max_generated_question_length = max_generated_question_length
        self.embedder = embedder
        self.epochs = nb_epochs
        self.batch_size = batch_size
        self.limit_train_data = limit_train_data
        self.limit_dev_data = limit_dev_data

    def get_train_set(self):
        ds, ds_size = self._prepare_data("train")
        return ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)\
            .repeat(self.epochs)\
            .batch(self.batch_size, drop_remainder=True)

    def get_dev_set(self):
        return self._prepare_data("dev")[0].batch(1).repeat(self.epochs)

    def _prepare_data(self, mode):
        def load_txt(name):
            limit = self.limit_train_data if mode == "train" else self.limit_dev_data
            if limit == -1:
                limit = None
            return np.loadtxt(f"{NQG_SQUAD_DATASET}/{mode}/{name}", dtype=np.str, delimiter='\n', max_rows=limit,
                              comments=None)
        contexts = np.reshape(load_txt("data.txt.source.txt"), (-1))
        questions = np.reshape(load_txt("data.txt.target.txt"), (-1))
        bio = np.reshape(load_txt("data.txt.bio"), (-1))
        assert contexts.shape[0] == questions.shape[0] == bio.shape[0]

        info(f"Loaded {contexts.shape[0]} examples.")
        contexts, questions = self.embedder.generate_bert_hlsqg_dataset(
            contexts, questions, bio, self.max_sequence_length, self.max_generated_question_length
        )
        contexts = pad_data(contexts, self.embedder.padding_token)
        questions = pad_data(questions, self.embedder.padding_token)
        return tf.data.Dataset.from_tensor_slices((contexts, questions)), len(contexts)
