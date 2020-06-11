import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, nb_layers, hidden_size, rnn_dropout, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        if nb_layers == 1:
            encoder_cell_fw = self._lstm_enc(False)
            encoder_cell_bw = self._lstm_enc(True)
        else:
            encoder_cell_fw = self._multi_layer_lstm_enc(False, nb_layers)
            encoder_cell_bw = self._multi_layer_lstm_enc(True, nb_layers)

        self.bidirectional_lstm = tf.keras.layers.Bidirectional(
            encoder_cell_fw, backward_layer=encoder_cell_bw, merge_mode='concat', name="bidirectional_lstm"
        )

    def call(self, inputs, **kwargs):
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm(inputs, **kwargs)
        if self.nb_layers == 1:
            encoder_state_c = tf.concat([forward_c, backward_c], axis=1)
            encoder_state_h = tf.concat([forward_h, backward_h], axis=1)
            # tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            encoder_state = [encoder_state_c, encoder_state_h]
        else:
            # Warning: this won't work
            _encoder_state = list()
            for state_fw, state_bw in zip([forward_c, forward_h], [backward_c, backward_h]):
                partial_state_c = tf.concat([state_fw.c, state_bw.c], axis=1)
                partial_state_h = tf.concat([state_fw.h, state_bw.h], axis=1)
                partial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=partial_state_c, h=partial_state_h)
                _encoder_state.append(partial_state)
            encoder_state = tuple(_encoder_state)
        return encoder_outputs, encoder_state

    # Build encoder cell
    def _lstm_enc(self, go_backwards: bool):
        return self._lstm(self.hidden_size, go_backwards)

    def _multi_layer_lstm_enc(self, go_backwards: bool, nb_layers: int):
        _encoder = self._lstm_enc(go_backwards)
        for _ in range(nb_layers) - 1:
            _encoder = self._lstm_enc(go_backwards)(_encoder)
        return _encoder

    def _lstm(self, size, go_backwards):
        return tf.keras.layers.LSTM(
            size,
            go_backwards=go_backwards,
            dropout=self.rnn_dropout,
            return_sequences=True,
            return_state=True
        )
