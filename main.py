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

        initializer = tf.initializers.glorot_uniform()
        hidden_state_size = 768

        self.W_sqg = tf.Variable(shape=(hidden_state_size, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initial_value=initializer(shape=(hidden_state_size, self.embedder.vocab_size())))
        self.b_sqg = tf.Variable(shape=(1, self.embedder.vocab_size()),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initial_value=initializer(shape=(1, self.embedder.vocab_size())))

    def generate_questions(self, tokens: List[int]):
        print(self.embedder.vocab_lookup(tokens))
        generated_question = []
        while len(generated_question) == 0 or generated_question[-1] != self.embedder.tokenizer.sep_token:
            tokens_tensor = tf.Variable([tokens], dtype=tf.int32)
            hidden_states, attentions = self.model(tokens_tensor)
            mask_state = hidden_states[:, -1]
            word_distribution = tf.math.softmax(tf.add(tf.matmul(mask_state, self.W_sqg), self.b_sqg))
            next_predicted_word = self.embedder.vocab_lookup(tf.argmax(word_distribution, axis=1).numpy())
            print(next_predicted_word)
            generated_question.append(next_predicted_word)
            tokens = self.embedder.generate_next_tokens(tokens, next_predicted_word)

        exit()


class NQG(Model):
    def generate_questions(self, tokens: List[int]):
        pass


def generate_questions(tokens: List[int], model: Model):
    return model.generate_questions(tokens)


ex_tokens = [101, 1037, 2744, 2109, 2761, 1999, 4315, 19969, 1010, 8549, 24997, 4140, 2038, 10599, 7321, 1012, 2536, 1044, 22571, 14573, 23072, 2031, 2042, 3755, 1012, 1996, 8367, 2089, 2031, 2042, 1037, 4117, 4431, 2000, 1996, 5364, 3761, 2022, 8791, 8663, 8549, 15808, 1006, 2351, 16710, 2475, 1007, 1998, 1996, 3412, 2135, 4736, 2098, 3267, 1997, 5364, 3951, 2964, 1999, 2010, 2051, 1010, 2478, 1037, 12266, 4315, 18170, 7062, 26136, 2006, 1996, 2171, 8549, 15808, 2011, 2126, 1997, 1996, 3803, 2773, 17504, 28745, 16515, 6528, 1006, 6719, 28152, 1007, 1010, 7727, 2000, 1996, 9530, 17048, 10708, 1997, 1037, 5399, 3141, 2773, 1999, 2446, 1041, 13623, 15460, 3366, 1006, 24627, 2004, 1999, 1000, 1037, 6926, 1997, 2028, 1997, 1996, 2163, 1997, 1996, 5364, 18179, 1000, 1007, 1012, 2, 9810, 2, 2001, 2198, 11130, 1005, 1055, 4233, 2188, 1998, 1996, 2803, 1997, 1996, 11130, 2923, 2929, 1012, 1999, 9810, 1010, 8549, 15808, 1010, 2295, 3234, 1010, 2001, 1037, 3003, 1997, 1996, 1000, 8055, 2283, 1000, 1010, 2061, 2170, 2138, 2009, 16822, 4336, 2013, 1996, 3804, 1997, 16394, 2083, 2019, 4707, 2090, 1996, 2103, 1011, 2110, 1997, 9810, 1998, 1996, 5364, 11078, 1012, 1996, 3830, 8549, 24997, 4140, 2001, 27023, 2135, 2034, 4162, 1999, 2605, 2000, 2216, 9530, 13102, 7895, 6591, 1006, 2035, 1997, 2068, 19774, 2372, 1997, 1996, 9114, 2277, 1007, 2920, 1999, 1996, 2572, 5092, 5562, 5436, 1997, 29185, 1024, 1037, 17910, 2098, 3535, 2000, 23277, 4355, 2373, 1999, 2605, 2013, 1996, 6383, 2160, 1997, 21980, 1012, 1996, 2693, 2052, 2031, 2018, 1996, 2217, 3466, 1997, 6469, 2075, 4262, 2007, 1996, 5364, 1012, 2947, 1010, 8549, 15808, 4606, 1041, 13623, 15460, 3366, 2011, 2126, 1997, 17504, 28745, 16515, 6528, 10743, 2150, 8549, 24997, 4140, 1010, 1037, 8367, 4632, 10085, 15370, 1996, 8330, 3426, 2007, 4331, 19657, 1999, 2605, 1012, 1031, 11091, 2734, 1033, 102, 103]


# data = read_square_dataset(SQUAD_DEV)
# ds = Embedder().generate_bert_hlsqg_dataset(data)
# print(ds[0])
# exit()
Bert().generate_questions(ex_tokens)
