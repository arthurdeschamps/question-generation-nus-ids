from data_utils.parse import read_squad_dataset
from data_utils.pre_processing import pad_data
import tensorflow as tf

from defs import SQUAD_DEV, SQUAD_TRAIN


class SquadDataset:
    """
    SQuAD dataset manager.
    """

    def __init__(self,
                 max_sequence_length,
                 max_generated_question_length,
                 embedder,
                 nb_epochs,
                 batch_size,
                 limit_train_data,
                 limit_dev_data):
        super(SquadDataset, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.max_generated_question_length = max_generated_question_length
        self.embedder = embedder
        self.epochs = nb_epochs
        self.batch_size = batch_size
        self.limit_train_data = limit_train_data
        self.limit_dev_data = limit_dev_data

    def get_train_set(self):
        ds = self._prepare_data(read_squad_dataset(SQUAD_TRAIN, limit=self.limit_train_data),
                                limit=self.limit_train_data)
        return ds.shuffle(buffer_size=256, reshuffle_each_iteration=True).repeat(self.epochs).batch(self.batch_size)

    def get_dev_set(self):
        return self._prepare_data(read_squad_dataset(SQUAD_DEV, limit=self.limit_dev_data),
                                  limit=self.limit_dev_data).batch(1).repeat(self.epochs)

    def _prepare_data(self, data, limit):
        x, y = self.embedder.generate_bert_hlsqg_dataset(
            data, self.max_sequence_length, self.max_generated_question_length)
        padding_value = self.embedder.tokenizer.pad_token_id
        x = pad_data(x, padding_value)[:limit]
        y = pad_data(y, padding_value)[:limit]
        return tf.data.Dataset.from_tensor_slices((x, y))
