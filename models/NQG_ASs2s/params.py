import tensorflow as tf


class HParams:
    def __init__(self):
        super(HParams).__init__()
        self.base_config = self
        self.dtype = tf.float32
        self.voca_size = 34004
        self.embedding_trainable = False
        self.hidden_size = 512
        self.encoder_layer = 1
        self.decoder_layer = 1
        self.answer_layer = 1
        self.dec_init_ans = True

        self.maxlen_q_train = 32
        self.maxlen_q_dev = 27
        self.maxlen_q_test = 27

        self.rnn_dropout = 0.5

        self.start_token = 1  # <GO> index
        self.end_token = 2  # <EOS> index

        # Keyword-net related parameters
        self.use_keyword = 4

        # Attention related parameters
        self.attn = 'normed_bahdanau'

        # Output layer related parameters
        self.if_wean = True

        # Training related parameters
        self.batch_size = 128
        self.learning_rate = 0.001
        self.decay_step = None
        self.decay_rate = 0.4 #0.5

        # Beam Search
        self.beam_width = 10
        self.length_penalty_weight = 2.1

    def values(self):
        return self.__dict__


def basic_params():
    '''A set of basic hyperparameters'''

    return HParams()


def h200_batch64():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 64
    return params


def h512_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    return params
