from typing import List
import numpy as np
import tensorflow as tf
from tensorflow_core import Variable
from transformers import BertTokenizer, TFBertModel
from data_utils.class_defs import Paragraph, SquadExample, Question


class Embedder:

    HL_TOKEN = '[unused1]'  # This token is used to indicate where the answer starts and finishes

    def __init__(self):
        super(Embedder, self).__init__()
        self.pretrained_weights_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights_name)

        # Token to indicate where the answer resides in the context
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [Embedder.HL_TOKEN]
        })

    @staticmethod
    def generate_next_tokens(prev_tokens: Variable, predicted_token: Variable):
        res = tf.concat((
            tf.expand_dims(prev_tokens[:-1], axis=0),
            tf.reshape(predicted_token, shape=(1, 1)),
            tf.reshape(prev_tokens[-1], shape=(1, 1))
        ), axis=1)
        return tf.squeeze(res)

    def generate_bert_hlsqg_input_embedding(self, context, answer):
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
        return self.tokenizer.encode(self.tokenizer.tokenize(question.question))

    def generate_bert_hlsqg_dataset(self, squad_examples: List[SquadExample]):
        x = []
        y = []
        for squad_example in squad_examples:
            for paragraph in squad_example.paragraphs:
                for qa in paragraph.qas:
                    # [CLS], c_1, c_2, ..., [HL] a_1, ..., a_|A|m [HL], ..., c_|C|, [SEP], [MASK]
                    # Where C is the context, A the answer (within the context, marked by special
                    # characters [HL] ... [HL]).
                    question = qa[0]
                    answers = qa[1]
                    if len(answers) > 0:  # Makes sure the question in answerable
                        for answer in answers:
                            input_emb = self.generate_bert_hlsqg_input_embedding(paragraph.context, answer)
                            label_emb = self.generate_bert_hlsqg_output_embedding(question)
                            # Maximum sequence length of this model
                            if len(input_emb) <= 512:
                                x.append(np.array(input_emb, dtype=np.int))
                                y.append(np.array(label_emb, dtype=np.int))
        return x, y

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def vocab_lookup(self, predicted_tokens):
        return self.tokenizer.decode(predicted_tokens)

