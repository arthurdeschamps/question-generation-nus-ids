import functools

import tensorflow as tf
import tensorflow_addons as tfa
import sys
from defs import ASS2S_DIR
from encoder import Encoder

sys.path.append(ASS2S_DIR + '/submodule/')
from mytools import *
import rnn_wrapper as wrapper


class ASs2s(tf.keras.Model):

    def __init__(self, params, *args, **kwargs):
        super(ASs2s, self).__init__(*args, **kwargs)
        self.params = params
        self.batch_size = params['batch_size']
        self.hidden_size = params['hidden_size']
        self.vocab_size = params['voca_size']
        self.features_d_type = params['dtype']
        self.beam_width = params['beam_width']
        self.length_penalty_weight = params['length_penalty_weight']
        self.rnn_dropout = self.params['rnn_dropout']

        self.embedding_matrix = None
        self.embedding_layer = self._build_embedding_layer()
        self.answer_encoder = Encoder(params['answer_layer'], self.hidden_size, self.rnn_dropout,
                                      name='answer_encoder', dtype=self.features_d_type)
        self.passage_encoder = Encoder(params['encoder_layer'], self.hidden_size, self.rnn_dropout,
                                       name='passage_encoder', dtype=self.features_d_type)
        self.decoder = self._build_decoder(params['decoder_layer'])
        self.attention_mechanism = self._build_attention_mechanism()
        self.attention_layer = self._build_attention_layer()
        self.output_layer = None if params['if_wean'] else tf.keras.layers.Dense(units=self.voca_size)

        self.scheduled_sampler = tfa.seq2seq.ScheduledEmbeddingTrainingSampler(
                sampling_probability=0.25,
                embedding_fn=self.embedding_layer
        )
        self.greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(self.embedding_layer)
        self.basic_decoder = lambda sampler: tfa.seq2seq.BasicDecoder(
            cell=self.attention_layer,
            sampler=sampler,
            output_layer=self.output_layer
        )
        self.beam_search_decoder = tfa.seq2seq.BeamSearchDecoder(
            cell=self.attention_layer,
            beam_width=self.beam_width,
            length_penalty_weight=self.length_penalty_weight,
            output_layer=self.output_layer,
            embedding_fn=self.embedding_layer
        )

    def call(self, inputs, training=True, mask=None):
        features = inputs
        sentence = features['s']  # [batch, length]
        len_s = tf.squeeze(features['len_s'], name="len_s")
        answer = features['a']

        # batch_size should not be specified
        # if fixed, then the redundant eval_data will make error
        # it may related to bug of tensorflow api
        batch_size = tf.shape(sentence)[0]

        if training:
            question = features['q']
            q_shape = (self.batch_size, None)
            tf.ensure_shape(question, shape=q_shape)
            question.set_shape(shape=q_shape)
            len_q = tf.squeeze(tf.cast(features['len_q'], tf.int32), name="len_q")
        else:
            question = None
            len_q = None

        embd_s = self.embedding_layer(sentence)
        embd_a = self.embedding_layer(answer)
        embd_q = self.embedding_layer(question) if question is not None else None

        answer_encoder_outputs, answer_encoder_state = self.answer_encoder(embd_a, training=training)
        passage_encoder_outputs, passage_encoder_state = self.passage_encoder(embd_s, training=training)

        if self.params['dec_init_ans'] and self.params['decoder_layer'] == self.params['answer_layer']:
            copy_state = answer_encoder_state
        elif self.params['encoder_layer'] == self.params['decoder_layer']:
            copy_state = passage_encoder_state
        else:
            copy_state = None
        if not training and self.beam_width > 0:
            passage_encoder_outputs = tfa.seq2seq.tile_batch(passage_encoder_outputs, self.beam_width)
            len_s = tfa.seq2seq.tile_batch(len_s, self.beam_width)

        if not training and self.beam_width > 0 and copy_state is not None:
            copy_state = tfa.seq2seq.tile_batch(copy_state, self.beam_width)

        if training:
            start_tokens = None
            sampler = self.scheduled_sampler
            sampler.initialize(
                inputs=embd_q,
                sequence_length=len_q,
            )
        else:  # EVAL & TEST
            start_tokens = self.params['start_token'] * tf.ones([batch_size], dtype=tf.int32)
            sampler = self.greedy_sampler
            sampler.initialize(embd_q, start_tokens, self.params['end_token'])

        self.attention_mechanism.setup_memory(
            memory=passage_encoder_outputs,
            memory_sequence_length=len_s
        )
        # Multi-layer Keyword Net (see ASs2s paper)
        self.attention_layer.cell._cell_input_fn = self._keyword_net(answer_encoder_outputs, training=training)
        decoder_cell = self.attention_layer
        # Decoder
        if training or self.beam_width == 0:
            initial_state = decoder_cell.zero_state(dtype=self.features_d_type, batch_size=batch_size)
            decoder = self.basic_decoder(sampler)
        else:
            initial_state = decoder_cell.zero_state(dtype=self.features_d_type,
                                                    batch_size=batch_size * self.beam_width)
            decoder = self.beam_search_decoder

        if copy_state is not None:
            initial_state = initial_state.clone(cell_state=copy_state)

        # Dynamic decoding
        dynamic_decode = functools.partial(
            tfa.seq2seq.dynamic_decode, decoder_init_input=embd_q, training=training
        )
        if not training:
            dynamic_decode = functools.partial(
                dynamic_decode,
                decoder_init_kwargs={
                    "initial_state": initial_state,
                    "start_tokens": start_tokens,
                    "end_token": self.params["end_token"]
                }
            )
        else:
            dynamic_decode = functools.partial(
                dynamic_decode,
                impute_finished=True, maximum_iterations=None,
                decoder_init_kwargs={
                    "initial_state": initial_state,
                }
            )

        predictions_q, logits_q = None, None

        if training:
            outputs, _, _ = dynamic_decode(decoder)
            logits_q = outputs.rnn_output

        elif not training and self.beam_width > 0:
            outputs, _, _ = dynamic_decode(
                decoder, impute_finished=False, maximum_iterations=self.params['maxlen_q_test']
            )
            predictions_q = outputs.predicted_ids  # [batch, length, beam_width]
            predictions_q = tf.transpose(predictions_q, [0, 2, 1])  # [batch, beam_width, length]
            predictions_q = predictions_q[:, 0, :]  # [batch, length]
        else:
            max_iter = self.params['maxlen_q_test'] if not training else self.params['maxlen_q_dev']
            outputs, _, _ = dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_iter)
            logits_q = outputs.rnn_output
            softmax_q = tf.nn.softmax(logits_q)
            predictions_q = tf.argmax(softmax_q, axis=-1)

        if not training:
            return predictions_q

        maxlen_q = self.params['maxlen_q_train'] if training else self.params['maxlen_q_dev']
        current_length = tf.shape(logits_q)[1]

        def concat_padding():
            num_pad = maxlen_q - current_length
            padding = tf.zeros([batch_size, num_pad, self.vocab_size], dtype=self.features_d_type)

            return tf.concat([logits_q, padding], axis=1)

        def slice_to_maxlen():
            return tf.slice(logits_q, [0, 0, 0], [batch_size, maxlen_q, self.vocab_size])

        logits_q = tf.cond(current_length < maxlen_q, concat_padding, slice_to_maxlen)
        return logits_q

    def _build_attention_layer(self):
        # Create an attention mechanism
        attention_wrapper = tfa.seq2seq.AttentionWrapper(
            self.decoder,
            self.attention_mechanism,
            attention_layer_size=2 * self.hidden_size,
            initial_cell_state=None,
            name="attention_wrapper"
        )

        if self.params['if_wean']:
            assert self.embedding_matrix is not None
            return wrapper.WeanWrapper(self.embedding_matrix, cell=attention_wrapper)
        return attention_wrapper

    def _build_decoder(self, nb_layers):
        def _cell():
            return tf.keras.layers.LSTMCell(
                units=2 * self.hidden_size,
                dropout=self.rnn_dropout
            )
        if nb_layers == 1:
            return _cell()
        else:
            return tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [_cell() for _ in range(nb_layers)]
            )

    def _build_embedding_layer(self):
        if self.params['embedding'] is None:
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.hidden_size,
                dtype=self.features_d_type,
                trainable=self.params['embedding_trainable'],
                name='embedding'
            )
        else:
            self.embedding_matrix = tf.Variable(
                np.load(self.params['embedding']).astype(np.float),
                dtype=tf.float32,
                trainable=self.params['embedding_trainable'],
                name='embedding_matrix'
            )
            embedding_layer = tf.keras.layers.Lambda(lambda idx: tf.nn.embedding_lookup(self.embedding_matrix, idx))
        return embedding_layer

    def _keyword_net(self, answer_encoder_outputs, training=False):
        if self.params['use_keyword'] > 0:
            # cell_input_fn = lambda inputs, attention : tf.concat([inputs, attention, o_s], -1)
            def cell_input_fn(inputs, attention):
                if not training and self.beam_width > 0:
                    last_attention = attention[0::self.beam_width]
                else:
                    last_attention = attention
                o_s = last_attention
                h_a = answer_encoder_outputs
                for _ in range(self.params['use_keyword']):
                    o_s = tf.expand_dims(o_s, 2)
                    p_s = tf.nn.softmax(tf.matmul(h_a, o_s))
                    o_s = tf.reduce_sum(p_s * h_a, axis=1)

                if not training and self.beam_width > 0:
                    o_s = tfa.seq2seq.tile_batch(o_s, self.beam_width)
                return tf.concat([inputs, o_s], -1)
            return cell_input_fn
        return None

    def _build_attention_mechanism(self):
        if self.params['attn'] == 'bahdanau':
            return tfa.seq2seq.BahdanauAttention(
                units=self.hidden_size * 2
            )
        elif self.params['attn'] == 'normed_bahdanau':
            return tfa.seq2seq.BahdanauAttention(
                units=self.hidden_size * 2,
                normalize=True)

        elif self.params['attn'] == 'luong':
            return tfa.seq2seq.LuongAttention(
                units=self.hidden_size * 2
            )

        elif self.params['attn'] == 'scaled_luong':
            return tfa.seq2seq.LuongAttention(
                units=self.hidden_size * 2,
                scale=True)
        else:
            raise ValueError('Unknown attention mechanism : %s' % self.params['attn'])
