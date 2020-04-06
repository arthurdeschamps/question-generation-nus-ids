from typing import List
from transformers import TFBertModel
from data_utils.embeddings import Embedder
import tensorflow as tf

from data_utils.parse import read_square_dataset
from defs import SQUAD_DEV


class Model:
    def generate_questions(self, tokens: List[int]):
        raise NotImplementedError()


class Bert(Model):

    def __init__(self):
        super(Bert, self).__init__()
        self.embedder = Embedder()
        self.pretrained_weights_name = 'bert-base-uncased'
        self.model = TFBertModel.from_pretrained(self.pretrained_weights_name)

    def generate_questions(self, tokens: List[int]):
        tokens_tensor = tf.Variable([tokens], dtype=tf.int32)
        hidden_states, attentions = self.model(tokens_tensor)
        print(hidden_states)
        print(attentions)


class NQG(Model):
    def generate_questions(self, tokens: List[int]):
        pass


def generate_questions(tokens: List[int], model: Model):
    return model.generate_questions(tokens)


#ds = parse.read_square_dataset(SQUAD_DEV)
ex_tokens = [101, 1996, 5879, 2015, 1006, 5879, 1024, 2053, 3126, 2386, 5104, 1025, 2413, 1024, 5879, 5104, 1025, 3763,
             1024, 5879, 3490, 1007, 2020, 1996, 2111, 2040, 1999, 1996, 6049, 1998, 6252, 4693, 2435, 2037, 2171, 2000,
             13298, 1010, 1037, 2555, 1999, 100, 2605, 100, 1996, 5879, 2015, 1006, 5879, 1024, 2053, 3126, 2386, 5104,
             1025, 2413, 1024, 5879, 5104, 1025, 3763, 1024, 5879, 3490, 1007, 2020, 1996, 2111, 2040, 1999, 1996, 6049,
             1998, 6252, 4693, 2435, 2037, 2171, 2000, 13298, 1010, 1037, 2555, 1999, 2605, 102, 103]


#data = read_square_dataset(SQUAD_DEV)
#ds = Embedder().generate_bert_hlsqg_dataset(data)
Bert().generate_questions(ex_tokens)
