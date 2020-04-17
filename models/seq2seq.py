from typing import List

from data_utils.class_defs import Answer
from data_utils.pre_processing import NQGDataPreprocessor
from models.base_model import BaseModel
from models.pre_trained.nqg.code.NQG.seq2seq_pt.onlinePreprocess import make_vocabulary_from_data
import numpy as np


class NQG(BaseModel):

    def __init__(self, passages: List[str], answers: List[Answer], vocab_size=30000,
                 *args, **kwargs):
        """

        :param passages: A list of passages, not assumed to be pre-processed.
        :param answers: List of corresponding answers for the passages.
        :param vocab_size: Maximum size of the vocabulary to be generated.
        """
        super(NQG, self).__init__(*args, **kwargs)
        self.data_preprocessor = NQGDataPreprocessor(passages)
        self.answers = answers
        self._generate_features(vocab_size)

    def generate_questions(self):
        pass

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError()

    def _generate_features(self, vocab_size: int):
        voc = make_vocabulary_from_data(self.data_preprocessor.passages, voc_size=vocab_size)
        answer_starts = np.array(list(answer.answer_start for answer in self.answers))
        answer_lengths = np.array(list(len(answer.text.split(' ')) for answer in self.answers))
        bio = self.data_preprocessor.create_bio_sequences(answer_starts, answer_lengths)
        case = self.data_preprocessor.create_case_sequences()
        ner = self.data_preprocessor.create_ner_sequences()
        pos = self.data_preprocessor.create_pos_sequences()
        self.passages = self.data_preprocessor.remove_cases()
        return voc, bio, case, ner, pos


passages = [
    "Hey there my name is Arthur Deschamps and I'm Swiss.",
    "The man door used to be a doorman for the Doors, standing by the door to open and close doors."
]
answers = [
    Answer("Arthur Deschamps", 5),
    Answer("for the Doors", 8)
]
NQG(passages, answers, embedder=None, model=None, max_sequence_length=None)
