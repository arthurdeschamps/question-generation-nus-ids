from functools import reduce
from typing import List
import numpy as np
import tensorflow as tf
import stanza
from stanza import Document


def pad_data(data: List[np.ndarray], padding_value) -> List[tf.Tensor]:
    """
    Transforms a variable sized list of arrays to a rectangular array by padding the arrays accordingly with the
    given padding value.
    """
    paddings = np.array([0, np.max(list(datapoint.shape[0] for datapoint in data))]).reshape((1, -1))
    return list(
        tf.pad(datapoint, paddings=paddings - np.array((0, len(datapoint))), mode='CONSTANT',
               constant_values=padding_value) for datapoint in data
    )


def array_to_string(arr: np.ndarray) -> str:
    return reduce(lambda t1, t2: t1 + " " + t2, arr)


class NQGDataPreprocessor:

    def __init__(self, documents: List[Document]):
        """
        :param documents: An array of analyzed documents.
        """
        super(NQGDataPreprocessor, self).__init__()
        self.analyzed = documents
        self.passages = list(list(passage.iter_words()) for passage in self.analyzed)

    def create_bio_sequences(self, answer_starts: np.ndarray, answer_lengths: np.ndarray) -> np.ndarray:
        """
        :param answer_starts: Indices of where the answers start for each passage.
        :param answer_lengths: The lengths (number of words) of each answer.
        :return: The BIO sequence of each passage.
        """
        bio_seqs = []
        for passage, answer_start, answer_length in zip(self.passages, answer_starts, answer_lengths):
            bio = np.full(shape=len(passage), fill_value='O', dtype=np.str)
            bio[answer_start] = 'B'
            if answer_length > 1:
                bio[answer_start+1:(answer_start + answer_length)] = 'I'
            bio_seqs.append(array_to_string(bio))
        return bio_seqs

    def create_case_sequences(self) -> np.ndarray:
        """
        :return: The casing sequence for each passage ('UP' when the word's first letter is capitalized, 'LOW' otherwise).
        """
        case_seqs = []
        for passage in self.passages:
            case_seq = np.array(list("LOW" for _ in range(len(passage))))
            case_indices = np.where(list(str.isupper(word.text[0]) for word in passage))
            case_seq[case_indices] = "UP"
            case_seqs.append(array_to_string(case_seq))
        return case_seqs

    def create_ner_sequences(self):
        ner_sequences = []
        for passage in self.analyzed:
            # Takes care of creating the NER sequence
            ner_sequence = np.full(shape=passage.num_tokens, fill_value='O', dtype=object)
            i = 0
            for word in passage.iter_words():
                ner_sequence[i] = word.parent.ner
                i += 1
            ner_sequences.append(array_to_string(ner_sequence))

        return np.array(ner_sequences)

    def create_pos_sequences(self):
        pos_sequences = []
        for passage in self.analyzed:
            # Creates the POS sequence
            pos_sequence = []
            for word in passage.iter_words():
                pos_sequence.append(word.xpos)
            pos_sequences.append(array_to_string(pos_sequence))
        return np.array(pos_sequences)

    def uncased_sequences(self):
        return list(array_to_string(list(str.lower(word.text) for word in sequence)) for sequence in self.passages)

    def _ner_mapping(self, ne_type):
        if ne_type in ("PERSON", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE",
                       "TIME", "DURATION", "SET"):
            return ne_type
        if ne_type == "ORG":
            return "ORGANIZATION"
        if ne_type == "LOC":
            return "LOCATION"
        if ne_type in ("NORP", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"):
            return "MISC"
        if ne_type in ("FAC", "GPE"):
            return "LOCATION"
        if ne_type in ("QUANTITY", "CARDINAL"):
            return "NUMBER"
        raise NotImplementedError(f"Named Entity type {ne_type} not recognized")
