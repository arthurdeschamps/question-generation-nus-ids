from typing import List
import numpy as np
import tensorflow as tf
from data_utils.class_defs import SquadExample, Question, Answer


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
        self.mask_token = tf.constant(self.tokenizer.mask_token_id, dtype=tf.int32)
        self.sep_token = tf.constant(self.tokenizer.sep_token_id, dtype=tf.int32)

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
        for old_input, input_end, predicted_token,\
                in zip(old_input_tokens, sizes, unstacked_predicted_tokens):
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

    def generate_bert_hlsqg_input_embedding(self, context, answer: Answer):
        """
        Generates embeddings according to the hlsqg schema (https://www.aclweb.org/anthology/D19-5821.pdf).
        :param context: A list of words representing the context, that is where the answer shall be found.
        :param answer: An answer object which content shall be part of the context.
        :return: An initial input embedding for BERT.
        """
        context_lhs_tokens = self.tokenizer.tokenize(context[:answer.answer_start])
        context_rhs_tokens = self.tokenizer.tokenize(context[answer.answer_start + len(answer.text):])
        answer_tokens = self.tokenizer.tokenize(answer.text)
        hl_token = self.tokenizer.tokenize(self.HL_TOKEN)[0]

        tokens = (
            self.tokenizer.cls_token,
            *context_lhs_tokens,
            hl_token,
            *answer_tokens,
            hl_token,
            *context_rhs_tokens,
            self.tokenizer.sep_token,
            self.tokenizer.mask_token
        )
        return self.tokenizer.encode(tokens, add_special_tokens=False)

    def generate_bert_hlsqg_output_embedding(self, question: Question):
        output_tokens = self.tokenizer.encode(self.tokenizer.tokenize(question.question), add_special_tokens=False)
        output_tokens.append(self.tokenizer.sep_token_id)
        return output_tokens

    def generate_bert_hlsqg_dataset(self,
                                    squad_examples: List[SquadExample],
                                    max_sequence_length,
                                    max_generated_question_length,
                                    limit=-1):
        """
        Generates a dataset for the HLSQG Bert architecture using the given SQuAD examples.
        :param limit: Number of datapoints to generate (-1 means no limit).
        :param squad_examples: A list of squad example objects.
        :param max_sequence_length: The maximum sequence length supported by the model to be used with this dataset.
        :param max_generated_question_length: The generated questions maximum length.
        :return:
        """
        x = []
        y = []
        generated = 0
        for squad_example in squad_examples:
            # [CLS], c_1, c_2, ..., [HL] a_1, ..., a_|A|m [HL], ..., c_|C|, [SEP], [MASK]
            # Where C is the context, A the answer (within the context, marked by special
            # characters [HL] ... [HL]).
            question = squad_example.question
            answer = squad_example.answer
            input_emb = self.generate_bert_hlsqg_input_embedding(squad_example.context, answer)
            label_emb = self.generate_bert_hlsqg_output_embedding(question)
            # Maximum sequence length of this model
            if len(input_emb) < max_sequence_length - max_generated_question_length:
                x.append(np.array(input_emb, dtype=np.int32))
                y.append(np.array(label_emb, dtype=np.int32))
                generated += 1
                if (limit > -1) and (generated >= limit):
                    return x, y
        return x, y

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def vocab_lookup(self, tokens):
        """
        :param tokens: The shape of tokens should be (batch_size, token_sequence_length).
        :return: A string represented by the given tokens.
        """
        return self.tokenizer.decode(tokens)
