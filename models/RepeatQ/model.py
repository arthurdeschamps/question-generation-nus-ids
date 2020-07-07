from collections import namedtuple

import tensorflow as tf
from tf_agents.specs import TensorSpec, BoundedTensorSpec

from defs import REPEAT_Q_EMBEDDINGS_FILENAME, PAD_TOKEN
from models.RepeatQ.layers.decoder import Decoder
from models.RepeatQ.layers.embedding import Embedding
from models.RepeatQ.layers.fact_encoder import FactEncoder
from models.RepeatQ.layers.attention import Attention
import tf_agents.networks as networks
from models.RepeatQ.model_config import ModelConfiguration


class RepeatQ(tf.keras.models.Model):

    NetworkState = namedtuple("NetworkState", (
        "base_question",
        "facts",
        "base_question_embeddings",
        "facts_encodings",
        "decoder_states",
        "observation",
        "is_first_step"
    ))

    def __init__(self,
                 voc_word_to_id,
                 config: ModelConfiguration,
                 *args,
                 **kwargs
                 ):
        super(RepeatQ, self).__init__(*args, **kwargs)

        self.config = config
        self.vocabulary_word_to_id = voc_word_to_id

        # Layers construction
        self.embedding_layer = self._build_embedding_layer()
        self.fact_encoder = self._build_fact_encoder()
        self.decoder = self._build_decoder()

    def call(self, inputs, training=None, mask=None):
        if inputs.is_first_step:
            base_question_embeddings = self.embedding_layer(inputs.base_question)
        else:
            base_question_embeddings = inputs.base_question_embeddings
        if inputs.is_first_step:
            facts_embeddings = self.embedding_layer(inputs.facts)
            facts_encodings = self.fact_encoder(facts_embeddings, training=training)
        else:
            facts_encodings = inputs.facts_encodings
        logits, decoder_states = self.decoder({
            "base_question_embeddings": base_question_embeddings,
            "facts_encodings": facts_encodings,
            "previous_token_embedding": self.embedding_layer(inputs.observation),
            "decoder_state": inputs.decoder_states
        })
        return logits, base_question_embeddings, facts_encodings, decoder_states

    def get_action(self, network_state, training):
        logits, bqe, facts_encodings, decoder_states = self(network_state, training=training)
        probs = tf.math.softmax(logits, axis=-1, name="probs")
        log_probs = tf.math.log(probs, name="log_prob")
        predicted_tokens = tf.argmax(probs, axis=-1, output_type=tf.int32, name="pred_tokens")
        log_prob = tf.reshape(
            tf.gather(log_probs, predicted_tokens, batch_dims=1), shape=(-1,)
        )
        predicted_tokens = tf.reshape(predicted_tokens, shape=(-1,))
        network_state = RepeatQ.NetworkState(
            base_question=network_state.base_question,
            facts=network_state.facts,
            base_question_embeddings=bqe,
            facts_encodings=facts_encodings,
            decoder_states=decoder_states,
            observation=predicted_tokens,
            is_first_step=False
        )
        return predicted_tokens, log_prob, network_state

    def _build_decoder(self):
        question_attention = self._build_attention_layer("base_question_attention")
        facts_attention = self._build_attention_layer("facts_attention")
        return Decoder(
            embedding_layer=self.embedding_layer,
            question_attention_mechanism=question_attention,
            facts_attention_mechanism=facts_attention,
            units=self.config.decoder_hidden_size,
            recurrent_dropout=self.config.recurrent_dropout,
            vocab_size=len(self.vocabulary_word_to_id),
            readout_size=self.config.decoder_readout_size,
            bos_token=self.vocabulary_word_to_id[PAD_TOKEN],
            name="decoder"
        )

    def _build_fact_encoder(self):
        return FactEncoder(
            encoder_hidden_size=self.config.fact_encoder_hidden_size,
            recurrent_dropout=self.config.recurrent_dropout,
            name="fact_encoder"
        )

    def _build_attention_layer(self, name):
        return Attention(
            attention_style=self.config.question_attention,
            attention_depth=self.config.attention_depth,
            name=name
        )

    def _build_embedding_layer(self):
        return Embedding.new(
            vocabulary=self.vocabulary_word_to_id,
            is_pretrained=self.config.embeddings_pretrained,
            embedding_size=self.config.embedding_size,
            embedding_path=f"{self.config.data_dir}/{REPEAT_Q_EMBEDDINGS_FILENAME}",
            name="embedding_layer"
        )
