# Following are functions that I always use
# Made by Yanghoon Kim, SNU
# Tensorflow 1.4
# 2018.03.01

import nltk
import numpy as np
import tensorflow as tf
import nltk


def embed_op(inputs, params, name='embedding'):
    if params['embedding'] is None:
        with tf.compat.v1.variable_scope('EmbeddingScope', reuse=tf.compat.v1.AUTO_REUSE):
            embedding = tf.compat.v1.get_variable(
                name,
                [params['voca_size'], params['hidden_size']],
                dtype=params['dtype'],

            )
    else:
        glove = np.load(params['embedding'])
        with tf.compat.v1.variable_scope('EmbeddingScope', reuse=tf.compat.v1.AUTO_REUSE):
            init = tf.constant_initializer(glove)
            embedding = tf.compat.v1.get_variable(
                name,
                [params['voca_size'], 300],
                initializer=init,
                dtype=params['dtype'],
                trainable=params['embedding_trainable']
            )

    tf.summary.histogram(embedding.name + '/value', embedding)
    return tf.nn.embedding_lookup(embedding, inputs)


def conv_op(embd_inp, params):
    fltr = tf.compat.v1.get_variable(
        'conv_fltr',
        params['kernel'],
        params['dtype'],
        regularizer=tf.keras.regularizers.l2(1.0)
    )

    convout = tf.nn.conv1d(embd_inp, fltr, params['stride'], params['conv_pad'])
    return convout


def ffn_op(x, params):
    out = x
    if params['ffn_size'] is None:
        ffn_size = []
    else:
        ffn_size = params['ffn_size']
    for unit_size in ffn_size[:-1]:
        out = tf.keras.layers.Dense(
            unit_size,
            activation=tf.tanh,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(1.0)
        )(out)
    return tf.keras.layers.Dense(
        params['label_size'],
        activation=None,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(1.0)
    )(out)


def dot_product_attention(q, k, v, bias, dropout_rate=0.0, name=None):
    with tf.compat.v1.variable_scope(name, default_name='dot_product_attention'):
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name='attention_weights')
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    return tf.matmul(weights, v)


def multihead_attention(query, memory, bias, num_heads, output_depth, dropout_rate, name=None):
    def split_heads(x, num_heads):
        def split_last_dimension(x, n):
            old_shape = x.get_shape().dims
            last = old_shape[-1]
            new_shape = old_shape[:-1] + [n] + [last // n if last else None]
            ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
            ret.set_shape(new_shape)
            return ret

        return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])

    def combine_heads(x):
        def combine_last_two_dimensions(x):
            old_shape = x.get_shape().dims
            a, b = old_shape[-2:]
            new_shape = old_shape[:-2] + [a * b if a and b else None]
            ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
            ret.set_shape(new_shape)
            return ret

        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

    with tf.compat.v1.variable_scope(name, default_name='multihead_attention'):
        depth_q = query.get_shape().as_list()[-1]
        if memory is None:
            # self attention
            combined = tf.keras.layers.Dense(
                units=3 * depth_q,
                name='qkv_transform'
            )(query)
            q, k, v = tf.split(combined, [depth_q, depth_q, depth_q], axis=2)

        else:
            depth_m = memory.get_shape().as_list()[-1]
            q = query
            combined = tf.keras.layers.Dense(
                units=2 * depth_m,
                name='kv_transform'
            )(memory)
            k, v = tf.split(combined, [depth_m, depth_m], axis=2)
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        depth_per_head = depth_q // num_heads
        q *= depth_per_head ** -0.5
        x = dot_product_attention(q, k, v, bias, dropout_rate, name)
        x = combine_heads(x)
        x = tf.keras.layers.Dense(output_depth, name='output_transform')(x)
        return x


def attention_bias_ignore_padding(memory_length, maxlen):
    mask = tf.sequence_mask(memory_length, maxlen, tf.int32)
    memory_padding = tf.equal(mask, 0)
    ret = tf.float64(memory_padding) * -1e9
    return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def bleu_score(labels, predictions,
               weights=None, metrics_collections=None,
               updates_collections=None, name=None):
    def _nltk_blue_score(labels, predictions):

        # slice after <eos>
        predictions = predictions.numpy()
        for i in range(len(predictions)):
            prediction = predictions[i]
            if 2 in prediction:  # 2: EOS
                predictions[i] = prediction[:prediction.index(2) + 1]

        labels = [
            [[w_id for w_id in label if w_id != 0]]  # 0: PAD
            for label in labels.numpy()]
        predictions = [
            [w_id for w_id in prediction]
            for prediction in predictions]

        return float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))

    score = tf.py_function(_nltk_blue_score, (labels, predictions), tf.float64)
    return tf.compat.v1.metrics.mean(score * 100)
