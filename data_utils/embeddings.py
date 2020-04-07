from typing import List

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

    def generate_next_tokens(self, prev_tokens, predicted_word):
        return prev_tokens[:-1] + self.tokenizer.encode(predicted_word, add_special_tokens=False) + [prev_tokens[-1]]

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
        encoded_tokens = self.tokenizer.encode(tokens, add_special_tokens=False)
        print(encoded_tokens)
        return encoded_tokens

    def generate_bert_hlsqg_output_embedding(self, question: Question):
        return self.tokenizer.encode(self.tokenizer.tokenize(question.question))

    def generate_bert_hlsqg_dataset(self, squad_examples: List[SquadExample]):
        ds = []
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
                            ds.append((
                                self.generate_bert_hlsqg_input_embedding(paragraph.context, answer),
                                self.generate_bert_hlsqg_output_embedding(question)
                            ))
        return ds

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def vocab_lookup(self, predicted_tokens):
        return self.tokenizer.decode(predicted_tokens)

