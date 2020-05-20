from logging import warning
from typing import List, Generator
import numpy as np
import tensorflow as tf
from stanza import Document

from data_processing.class_defs import SquadExample, Question, Answer
from data_processing.nqg_dataset import NQGDataset
from data_processing.pre_processing import NQGDataPreprocessor, pad_data


class Embedder:
    """
    Converts tokenized sequences to embeddings, suitable for BERT.
    """
    HL_TOKEN = '[unused1]'  # This token is used to indicate where the answer starts and finishes

    def __init__(self, pretrained_model_name, tokenizer):
        super(Embedder, self).__init__()
        self.pretrained_weights_name = pretrained_model_name
        self.tokenizer = tokenizer

        self.padding_token = tf.constant(self.tokenizer.pad_token_id, dtype=tf.int32)
        self.mask_token = tf.Variable(-1, name="mask_token", dtype=tf.int32, trainable=False)
        self.sep_token = tf.Variable(-1, name="mask_token", dtype=tf.int32, trainable=False)
        if self.tokenizer.mask_token_id is not None:
            self.mask_token.assign(self.tokenizer.mask_token_id)
        if self.tokenizer.sep_token_id is not None:
            self.sep_token.assign(self.tokenizer.sep_token_id)

        # Token to indicate where the answer resides in the context
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [Embedder.HL_TOKEN]
        })

    @staticmethod
    def generate_next_input_tokens(prev_tokens: tf.Tensor,
                                   predicted_tokens: tf.Tensor,
                                   padding_token: tf.Tensor):
        """
        Generates token sequences to use for the next predictive iteration.
        :param prev_tokens: The tokens associated with the paragraphs and generated questions at the previous iteration.
        The shape should be (batch_size, sequence_length)
        :param predicted_tokens: Predicted tokens, that is predicted next word for each paragraph. Shape should be
        (batch_size, 1)
        :param padding_token: Tensor containing the ID of the padding token.
        :return: the token sequences for the next iteration, according to this HLSQG architecture from this paper:
        https://www.aclweb.org/anthology/D19-5821.pdf
        """
        non_padding_indices = tf.not_equal(prev_tokens, padding_token)
        sizes = tf.unstack(tf.cast(tf.reduce_sum(tf.cast(non_padding_indices, dtype=tf.int32), axis=1), dtype=tf.int32))
        unstacked_predicted_tokens = tf.unstack(predicted_tokens)
        old_input_tokens = tf.unstack(prev_tokens)
        new_input_tokens = []
        # Creates the new sequences
        for old_input, input_end, predicted_token in zip(old_input_tokens, sizes, unstacked_predicted_tokens):
            new_input_tokens.append(
                tf.concat(
                    (old_input[:input_end - 1],  # this is the tokens we had so far (without mask)
                     tf.expand_dims(predicted_token, axis=0),  # this is our newly introduced token
                     tf.expand_dims(old_input[input_end - 1], axis=0),  # this corresponds to the mask token
                     old_input[input_end:]),  # finally, adds back the paddings to make sure we have a rectangular ds
                    axis=0
                )
            )

        return tf.stack(new_input_tokens, axis=0)

    def generate_bert_hlsqg_input_embedding(self, context: np.ndarray, bio: np.ndarray):
        """
        Generates embeddings according to the hlsqg schema (https://www.aclweb.org/anthology/D19-5821.pdf).
        :param context: A list of words representing the context, that is where the answer shall be found.
        :param answer: An answer object which content shall be part of the context.
        :return: An initial input embedding for BERT.
        """
        start_index = np.where(bio == 'B')[0][0]
        lhs = context[:start_index]
        if start_index >= len(context):
            answer = np.array([])
            rhs = []
        else:
            answer_indices = np.where(np.logical_or(bio == "B", bio == "I"))
            try:
                answer = context[answer_indices]
            except IndexError:
                # TODO this happens even though it should never. Potentially a mismatch between bio and source
                answer = np.array([])
                warning("mismatch between bio and passage")
            rhs = context[start_index + answer.shape[0]:]

        if "gpt" in self.pretrained_weights_name:
            tokens = self.tokenizer.tokenize(" ".join([
                *lhs,
                self.HL_TOKEN,
                *answer,
                self.HL_TOKEN,
                *rhs
            ]))
        else:
            tokens = self.tokenizer.tokenize(" ".join([
                self.tokenizer.cls_token,
                *lhs,
                self.HL_TOKEN,
                *answer,
                self.HL_TOKEN,
                *rhs,
                self.tokenizer.sep_token,
                self.tokenizer.mask_token
            ]))
        embedding = self.tokenizer.encode(tokens)
        return embedding

    def generate_bert_hlsqg_output_embedding(self, question: str):
        tokenized = self.tokenizer.tokenize(question)
        if "bert" in self.pretrained_weights_name:
            tokenized.append(self.tokenizer.sep_token_id)
        try:
            output_embedding = self.tokenizer.encode(tokenized)
        except ValueError:
            output_embedding = ["0"]
            warning(f"Problem with tokens: {tokenized}")  # TODO solve this
        return output_embedding

    def generate_bert_hlsqg_dataset(self,
                                    contexts,
                                    questions,
                                    bios,
                                    max_sequence_length,
                                    max_generated_question_length,
                                    limit=-1):
        """
        Generates a dataset for the HLSQG Bert architecture using the given SQuAD examples.
        :param limit: Number of datapoints to generate (-1 means no limit).
        :param ds:
        :param max_sequence_length: The maximum sequence length supported by the model to be used with this dataset.
        :param max_generated_question_length: The generated questions maximum length.
        :return:
        """
        padding_value = self.tokenizer.pad_token_id

        generated = 0
        x = []
        y = []
        for context, question, bio in zip(contexts, questions, bios):
            # [CLS], c_1, c_2, ..., [HL] a_1, ..., a_|A|m [HL], ..., c_|C|, [SEP], [MASK]
            # Where C is the context, A the answer (within the context, marked by special
            # characters [HL] ... [HL]).
            input_emb = self.generate_bert_hlsqg_input_embedding(np.array(context.split()),
                                                                 np.array(bio.split()))
            label_emb = self.generate_bert_hlsqg_output_embedding(question)
            # Maximum sequence length of this model
            if len(input_emb) < max_sequence_length - max_generated_question_length:
                x.append(np.array(input_emb, dtype=np.int32))
                y.append(np.array(label_emb, dtype=np.int32))
                generated += 1
                if (limit > -1) and (generated >= limit):
                    break
        return x, y

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def vocab_lookup(self, tokens):
        """
        :param tokens: The shape of tokens should be (batch_size, token_sequence_length).
        :return: A string represented by the given tokens.
        """
        return self.tokenizer.decode(tokens)
