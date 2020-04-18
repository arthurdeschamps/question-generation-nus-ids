import os
import subprocess
from typing import List

from data_utils.nqg_dataset import NQGDataset
from defs import PROCESSED_DATA_DIR, NQG_PREDS_OUTPUT_PATH, PRETRAINED_MODELS_DIR
from data_utils.class_defs import Answer
from data_utils.pre_processing import NQGDataPreprocessor
import numpy as np


class NQG:

    def __init__(self, nqg_dataset: NQGDataset, *args, **kwargs):
        """
        :param nqg_dataset: A list of (passage, answer), not assumed to be pre-processed.
        """
        super(NQG, self).__init__(*args, **kwargs)
        self.paths = {}
        contexts, self.answers = nqg_dataset.get_dataset()
        self.data_preprocessor = NQGDataPreprocessor(contexts)
        self._generate_features()

    def generate_questions(self):
        subprocess.run([
            "python3",
            f"{PRETRAINED_MODELS_DIR}/nqg/code/NQG/seq2seq_pt/translate.py",
            "-model",
            f"{PRETRAINED_MODELS_DIR}/nqg/data/redistribute/QG/models/NQG_plus/model_e2.pt",
            "-src",
            self.paths["source.txt"],
            "-bio",
            self.paths["bio"],
            "-feats",
            self.paths["pos"],
            self.paths["ner"],
            self.paths["case"],
            "-output",
            NQG_PREDS_OUTPUT_PATH
        ])

    def _generate_features(self):
        answer_starts = np.array(list(answer.start_index for answer in self.answers))
        answer_lengths = np.array(list(answer.nb_words for answer in self.answers))
        bio = self.data_preprocessor.create_bio_sequences(answer_starts, answer_lengths)
        case = self.data_preprocessor.create_case_sequences()
        ner = self.data_preprocessor.create_ner_sequences()
        pos = self.data_preprocessor.create_pos_sequences()
        self.passages = self.data_preprocessor.uncased_sequences()
        data_dir = f"{PROCESSED_DATA_DIR}/temp"
        os.makedirs(data_dir, exist_ok=True)
        for data_name, data in (("source.txt", self.passages), ("bio", bio), ("case", case), ("ner", ner), ("pos", pos)):
            fname = f"{data_dir}/data.txt.{data_name}"
            self.paths[data_name] = fname
            np.savetxt(fname, data, fmt="%s")


ds = NQGDataset(dataset_type="squad_dev", data_limit=1)
NQG(ds).generate_questions()
