import functools

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sys
from defs import ASS2S_DIR

sys.path.append(ASS2S_DIR + '/submodule/')
from mytools import *
import rnn_wrapper as wrapper


def _attention(params, memory, memory_length):
    if params['attn'] == 'bahdanau':
        return tfa.seq2seq.BahdanauAttention(
            units=params['hidden_size'] * 2,
            memory=memory,
            memory_sequence_length=memory_length)
    elif params['attn'] == 'normed_bahdanau':
        return tfa.seq2seq.BahdanauAttention(
            units=params['hidden_size'] * 2,
            memory=memory,
            memory_sequence_length=memory_length,
            normalize=True)

    elif params['attn'] == 'luong':
        return tfa.seq2seq.LuongAttention(
            units=params['hidden_size'] * 2,
            memory=memory,
            memory_sequence_length=memory_length)

    elif params['attn'] == 'scaled_luong':
        return tfa.seq2seq.LuongAttention(
            units=params['hidden_size'] * 2,
            memory=memory,
            memory_sequence_length=memory_length,
            scale=True)
    else:
        raise ValueError('Unknown attention mechanism : %s' % params['attn'])


def q_generation(features, labels, mode, params):
    dtype = params['dtype']
    hidden_size = params['hidden_size']
    voca_size = params['voca_size']

    sentence = features['s']  # [batch, length]
    len_s = features['len_s']

    answer = features['a']
    len_a = features['len_a']

    beam_width = params['beam_width']
    length_penalty_weight = params['length_penalty_weight']

    # batch_size should not be specified
    # if fixed, then the redundant eval_data will make error
    # it may related to bug of tensorflow api
    batch_size = tf.shape(sentence)[0]

    if mode != tf.estimator.ModeKeys.PREDICT:
        question = tf.cast(features['q'], tf.int32, name="question")  # label
        len_q = tf.cast(features['len_q'], tf.int32, name="len_q")
        q_shape = (params['batch_size'], None)
        tf.ensure_shape(question, shape=q_shape)
        question.set_shape(shape=q_shape)
    else:
        question = None
        len_q = None

    # Embedding for sentence, question and rnn encoding of sentence
    with tf.compat.v1.variable_scope('SharedScope'):
        # Embedded inputs
        # Same name == embedding sharing
        embd_s = embed_op(sentence, params, name='embedding')
        embd_a = embed_op(answer, params, name='embedding')
        if question is not None:
            embd_q = embed_op(question, params, name='embedding')

        def _lstm(size, go_backwards):
            return tf.keras.layers.LSTM(
                size,
                go_backwards=go_backwards,
                dropout=params['rnn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0,
                return_sequences=True,
                return_state=True
            )

        def lstm_cell(nb_hidden_units):
            dropout = params['rnn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0
            tf.print("Dropout rate: ", dropout)
            return tf.keras.layers.LSTMCell(
                units=nb_hidden_units,
                dropout=dropout
            )

        # Build encoder cell
        def lstm_enc(go_backwards: bool):
            return _lstm(hidden_size, go_backwards)

        def multi_layer_lstm_enc(go_backwards: bool, nb_layers: int):
            _encoder = lstm_enc(go_backwards)
            for _ in range(nb_layers) - 1:
                _encoder = lstm_enc(go_backwards)(_encoder)
            return _encoder

        if params['encoder_layer'] == 1:
            encoder_cell_fw = lstm_enc(False)
            encoder_cell_bw = lstm_enc(True)
        else:
            encoder_cell_fw = multi_layer_lstm_enc(False, params['encoder_layer'])
            encoder_cell_bw = multi_layer_lstm_enc(True, params['encoder_layer'])

        encoder_outputs, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
            encoder_cell_fw,
            backward_layer=encoder_cell_bw,
            dtype=dtype,
            merge_mode='concat'
        )(inputs=embd_s)

        if params['encoder_layer'] == 1:
            encoder_state_c = tf.concat([forward_c, backward_c], axis=1)
            encoder_state_h = tf.concat([forward_h, backward_h], axis=1)
            encoder_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        else:
            # Warning: this won't work
            _encoder_state = list()
            for state_fw, state_bw in zip([forward_c, forward_h], [backward_c, backward_h]):
                partial_state_c = tf.concat([state_fw.c, state_bw.c], axis=1)
                partial_state_h = tf.concat([state_fw.h, state_bw.h], axis=1)
                partial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=partial_state_c, h=partial_state_h)
                _encoder_state.append(partial_state)
            encoder_state = tuple(_encoder_state)

        if params['answer_layer'] == 1:
            answer_cell_fw = lstm_enc(go_backwards=False)
            answer_cell_bw = lstm_enc(go_backwards=True)

        else:
            answer_cell_fw = multi_layer_lstm_enc(False, params['answer_layer'])
            answer_cell_bw = multi_layer_lstm_enc(True, params['answer_layer'])

        with tf.compat.v1.variable_scope('answer_scope'):
            answer_outputs, answer_forward_c, answer_forward_h, answer_backward_c, answer_backward_h = \
                tf.keras.layers.Bidirectional(
                    answer_cell_fw,
                    backward_layer=answer_cell_bw,
                    dtype=dtype,
                )(inputs=embd_a,)

        if params['answer_layer'] == 1:
            answer_state_c = tf.concat([answer_forward_c, answer_backward_c], axis=1)
            answer_state_h = tf.concat([answer_forward_h, answer_backward_h], axis=1)
            # answer_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=answer_state_c, h=answer_state_h)
            answer_state = [answer_state_c, answer_state_h]
        else:
            # Warning: this won't work
            _answer_state = list()
            for state_fw, state_bw in zip([answer_forward_c, answer_forward_h], [answer_backward_c, answer_forward_h]):
                partial_state_c = tf.concat([state_fw.c, state_bw.c], axis=1)
                partial_state_h = tf.concat([state_fw.h, state_bw.h], axis=1)
                partial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=partial_state_c, h=partial_state_h)
                _answer_state.append(partial_state)
            answer_state = tuple(_answer_state)

        if params['dec_init_ans'] and params['decoder_layer'] == params['answer_layer']:
            copy_state = answer_state
        elif params['encoder_layer'] == params['decoder_layer']:
            copy_state = encoder_state
        else:
            copy_state = None

        if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_width)
            len_s = tfa.seq2seq.tile_batch(len_s, beam_width)

        if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0 and copy_state is not None:
            copy_state = tfa.seq2seq.tile_batch(copy_state, beam_width)

    with tf.compat.v1.variable_scope('SharedScope/EmbeddingScope', reuse=True):
            embedding_q = tf.compat.v1.get_variable('embedding', shape=[params['voca_size'], 300])
    # Rnn decoding of sentence with attention 
    with tf.compat.v1.variable_scope('QuestionGeneration'):
        # Memory for attention
        attention_states = encoder_outputs

        # Create an attention mechanism
        attention_mechanism = _attention(params, attention_states, len_s)

        # Build decoder cell
        decoder_cell = lstm_cell(2*hidden_size) if params['decoder_layer'] == 1 \
            else tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(2*hidden_size) for _ in range(params['decoder_layer'])]
        )

        if params['use_keyword'] > 0:
            # cell_input_fn = lambda inputs, attention : tf.concat([inputs, attention, o_s], -1)
            def cell_input_fn(inputs, attention):
                if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
                    last_attention = attention[0::beam_width]
                else:
                    last_attention = attention
                o_s = last_attention
                h_a = answer_outputs
                for _ in range(params['use_keyword']):
                    o_s = tf.expand_dims(o_s, 2)
                    p_s = tf.nn.softmax(tf.matmul(h_a, o_s), name='p_s')
                    o_s = tf.reduce_sum(p_s * h_a, axis=1)

                if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
                    o_s = tfa.seq2seq.tile_batch(o_s, beam_width)

                return tf.concat([inputs, o_s], -1)

        else:
            cell_input_fn = None

        decoder_cell = tfa.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=2*hidden_size,
            cell_input_fn=cell_input_fn,
            initial_cell_state=None,
            name="attention_wrapper"
        )
        # initial_cell_state = encoder_state if params['encoder_layer'] == params['decoder_layer'] else None)

        if params['if_wean']:
            decoder_cell = wrapper.WeanWrapper(decoder_cell, embedding_q)
            output_layer = None
        else:
            output_layer = tf.keras.layers.Dense(units=voca_size)

        # Helper for decoder cell
        def emb_fn(idx):
            return tf.nn.embedding_lookup(embedding_q, idx)
        if mode == tf.estimator.ModeKeys.TRAIN:
            sampler = tfa.seq2seq.ScheduledEmbeddingTrainingSampler(
                sampling_probability=0.25,
                embedding_fn=emb_fn
            )
            sampler.initialize(
                inputs=embd_q,
                sequence_length=len_q,
                embedding=embedding_q,
            )
        else:  # EVAL & TEST
            start_tokens = params['start_token'] * tf.ones([batch_size], dtype=tf.int32)
            sampler = tfa.seq2seq.GreedyEmbeddingSampler(emb_fn)
            sampler.initialize(embd_q, start_tokens, params['end_token'])

        # Decoder
        if mode != tf.estimator.ModeKeys.PREDICT or beam_width == 0:
            initial_state = decoder_cell.zero_state(dtype=dtype, batch_size=batch_size)
            if copy_state is not None:
                initial_state = initial_state.clone(cell_state=copy_state)
            decoder = tfa.seq2seq.BasicDecoder(
                cell=decoder_cell,
                sampler=sampler,
                output_layer=output_layer
            )

        else:
            initial_state = decoder_cell.zero_state(dtype=dtype, batch_size=batch_size * beam_width)
            if copy_state is not None:
                initial_state = initial_state.clone(cell_state=copy_state)
            decoder = tfa.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=embedding_q,
                start_tokens=start_tokens,
                end_token=params['end_token'],
                initial_state=initial_state,
                beam_width=beam_width,
                length_penalty_weight=length_penalty_weight)

        # Dynamic decoding
        dynamic_decode = functools.partial(
            tfa.seq2seq.dynamic_decode, decoder_init_input=embd_q
        )
        if mode == tf.estimator.ModeKeys.TRAIN:
            outputs, _, _ = dynamic_decode(
                decoder, impute_finished=True, maximum_iterations=None, decoder_init_kwargs={
                    "initial_state": initial_state,
                    "embedding": embedding_q
                }
            )
            logits_q = outputs.rnn_output
            softmax_q = tf.nn.softmax(logits_q)
            predictions_q = tf.argmax(softmax_q, axis=-1)
        elif mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
            outputs, _, _ = tfa.seq2seq.dynamic_decode(
                decoder, impute_finished=False, maximum_iterations=params['maxlen_q_test'],
                decoder_init_kwargs={
                    "initial_state": initial_state,
                }
            )
            predictions_q = outputs.predicted_ids  # [batch, length, beam_width]
            predictions_q = tf.transpose(predictions_q, [0, 2, 1])  # [batch, beam_width, length]
            predictions_q = predictions_q[:, 0, :]  # [batch, length]
        else:
            max_iter = params['maxlen_q_test'] if mode == tf.estimator.ModeKeys.PREDICT else params['maxlen_q_dev']
            outputs, _, _ = dynamic_decode(
                decoder, impute_finished=True, maximum_iterations=max_iter,
                decoder_init_kwargs={
                    "initial_state": initial_state,
                    "start_tokens": start_tokens,
                    "end_token": params["end_token"]
                }
            )
            logits_q = outputs.rnn_output
            softmax_q = tf.nn.softmax(logits_q)
            predictions_q = tf.argmax(softmax_q, axis=-1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'question': predictions_q
            })
    # Loss
    label_q = tf.concat([question[:, 1:], tf.zeros([batch_size, 1], dtype=tf.int32)], axis=1, name='label_q')
    maxlen_q = params['maxlen_q_train'] if mode == tf.estimator.ModeKeys.TRAIN else params['maxlen_q_dev']
    current_length = tf.shape(logits_q)[1]

    def concat_padding():
        num_pad = maxlen_q - current_length
        padding = tf.zeros([batch_size, num_pad, voca_size], dtype=dtype)

        return tf.concat([logits_q, padding], axis=1)

    def slice_to_maxlen():
        return tf.slice(logits_q, [0, 0, 0], [batch_size, maxlen_q, voca_size])

    logits_q = tf.cond(current_length < maxlen_q,
                       concat_padding,
                       slice_to_maxlen)

    weight_q = tf.sequence_mask(len_q, maxlen_q, dtype)

    loss_q = tfa.seq2seq.sequence_loss(
        logits_q,
        label_q,
        weight_q,  # [batch, length]
        average_across_timesteps=True,
        average_across_batch=True,
        softmax_loss_function=None  # default : sparse_softmax_cross_entropy
    )

    loss = loss_q

    # eval_metric for estimator
    eval_metric_ops = {
        'bleu': bleu_score(label_q, predictions_q)
    }

    # Summary
    tf.summary.scalar('loss_question', loss_q)
    tf.summary.scalar('total_loss', loss)

    # Optimizer
    learning_rate = params['learning_rate']
    if params['decay_step'] is not None:
        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate,
            tf.compat.v1.train.get_global_step(),
            params['decay_step'],
            params['decay_rate'],
            staircase=True
        )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    grad_and_var = optimizer.compute_gradients(loss, tf.compat.v1.trainable_variables())
    grad, var = zip(*grad_and_var)
    train_op = optimizer.apply_gradients(zip(grad, var), global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
