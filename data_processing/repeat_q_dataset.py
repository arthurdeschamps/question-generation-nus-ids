import json
from typing import List, Dict

import stanza
from tqdm import tqdm
import numpy as np
from data_processing.class_defs import SquadExample, SquadMultiQAExample, RepeatQExample, RepeatQFeature
from data_processing.dataset import Dataset
from defs import UNKNOWN_TOKEN


class RepeatQDataset:

    def __init__(self,
                 ds_json_path,
                 vocabulary: Dict[str, int],
                 feature_vocab: Dict[str, int],
                 unk_token=UNKNOWN_TOKEN,
                 pad_sequences=True,
                 pad_id=0,
                 data_limit=-1,
                 use_pos_features=True,
                 use_ner_features=True):
        """
        Dataset to use in conjunction with the RepeatQ model.
        :param ds_json_path: Path to a JSON file containing facts, base questions and target questions.
        :param vocabulary: A dictionary which maps words to their ids (used to convert words to ids).
        :param feature_vocab: A dictionary mapping feature words to ids.
        :param unk_token: The unknown token/word. Default is the one used by NQG (<unk>).
        :param pad_sequences: If sequences are to be padded to create a rectangular dataset. Default is True.
        :param pad_id: The id of the padding token (default is 0).
        :param data_limit: Number of examples to keep
        :param use_pos_features: Whether to use POS features or not.
        :param use_ner_features: Whether to use NER features or not.
        """
        super(RepeatQDataset, self).__init__()
        self.ds_path = ds_json_path
        self.vocab = vocabulary
        self.feature_vocab = feature_vocab
        self.unk_token = unk_token
        self.pad_sequences = pad_sequences
        self.pad_id = pad_id
        self.use_pos_features = use_pos_features
        self.use_ner_features = use_ner_features
        self.ds = self.read_dataset(data_limit)

    def read_dataset(self, data_limit):
        with open(self.ds_path, mode='r') as f:
            data = RepeatQExample.from_json(json.load(f))
            if data_limit < 0:
                return data
            return data[:data_limit]

    def get_dataset(self):
        base_questions = []
        base_questions_features = []
        facts_list = []
        facts_features = []
        targets = []
        passage_ids = []
        max_fact_length = 0
        max_nb_facts = 0
        for example in tqdm(self.ds):
            if example.rephrased_question == "":
                continue
            passage_ids.append(example.passage_id)
            targets.append(self.words_to_ids(example.rephrased_question.split()))
            base_questions.append(self.words_to_ids(example.base_question.split()))
            base_questions_features.append(self.features_to_ids(example.base_question_features))
            facts = [self.words_to_ids(fact.split(' ')) for fact in example.facts]
            facts_features.append([self.features_to_ids(fact) for fact in example.facts_features])
            max_fact_length = max(max_fact_length, max(len(fact) for fact in facts))
            max_nb_facts = max(max_nb_facts, len(facts))
            facts_list.append(facts)

        if not self.pad_sequences:
            return base_questions, facts_list, targets
        base_questions = self.sequence_padding(base_questions)
        targets = self.sequence_padding(targets)
        facts_list = np.array(self.matrix_padding(facts_list, max_length=128, max_width=max_nb_facts))
        return base_questions, base_questions_features, facts_list, facts_features, targets, passage_ids

    def words_to_ids(self, sentence: List[str]):
        return [self.vocab.get(word.lower(), self.vocab[self.unk_token]) for word in sentence]

    def features_to_ids(self, feature: RepeatQFeature):
        if self.use_pos_features:
            pos_features = [self.feature_vocab[tag] for tag in feature.pos_tags.split()]
        else:
            pos_features = []
        if self.use_ner_features:
            entity_features = [self.feature_vocab[tag] for tag in feature.entity_tags.split()]
        else:
            entity_features = []
        return pos_features, entity_features

    def sequence_padding(self, sequences: List[List[int]], max_length=None):
        """
        Pads liner sequences to the longest sequence length.
        """
        if max_length is None:
            max_length = max([len(seq) for seq in sequences])
        padded_sequences = list(seq[:max_length] + [self.pad_id for _ in range(max_length - len(seq))] for seq in sequences)
        return np.array(padded_sequences)

    def matrix_padding(self, matrices: List[List[List[int]]], max_length, max_width):
        """
        Pads in 2-dimensions, to the max sequence length on the first axis and to the max number of sequences per
        "batch" of sequences on the second axis.
        Ex:
        [[[1 8 39 4 19]
        [93 8 3]],
        [[7 4]]]
        becomes
        [[[1 8 39 4 19]
        [93 8 3 0 0]],
        [[7 4 0 0 0],
        [0 0 0 0 0]]]
        """
        # First dimension
        matrices = [self.sequence_padding(matrix, max_length=max_length) for matrix in matrices]
        # Second dimension
        pad_seq = [self.pad_id for _ in range(max_length)]
        matrices = np.array([matrix if len(matrix) == max_width else np.append(
                matrix,
                [pad_seq for _ in range(max_width - len(matrix))],
                axis=0
        ) for matrix in matrices])
        return matrices

