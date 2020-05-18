import logging
import random
from typing import List, Tuple

import stanza
from stanza import Document

from data_processing.class_defs import SquadExample, QAExample
from data_processing.parse import read_squad_dataset, read_qa_dataset
from defs import SQUAD_DEV, SQUAD_TRAIN, MEDQUAD_TRAIN, MEDQUAD_DEV, MEDQA_HANDMADE_FILEPATH


class NQGDataset:
    class Answer:

        def __init__(self, start_index, nb_words, text):
            super(NQGDataset.Answer, self).__init__()
            self.start_index = start_index
            self.nb_words = nb_words
            self.text = text

    def __init__(self, dataset_name="squad", mode="train", data_limit=-1):
        super(NQGDataset, self).__init__()
        stanza.download('en')
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,ner')
        self.datatype = QAExample
        if dataset_name == "squad":
            self.datatype = SquadExample
            if mode == "train":
                datapath = SQUAD_TRAIN
            elif mode == "dev":
                datapath = SQUAD_DEV
            else:
                raise ValueError()
            self.ds = read_squad_dataset(
                datapath,
                limit=data_limit
            )
        elif dataset_name == "medquad":
            if mode == "train":
                datapath = MEDQUAD_TRAIN
            elif mode == "dev":
                datapath = MEDQUAD_DEV
            else:
                raise ValueError()
            self.ds = read_qa_dataset(datapath, limit=data_limit)
        elif dataset_name == "medqa_handmade":
            if mode == "test":
                datapath = MEDQA_HANDMADE_FILEPATH
            else:
                raise ValueError()
            self.ds = read_qa_dataset(datapath, limit=data_limit)
        else:
            raise NotImplementedError()

    def get_dataset(self) -> Tuple[List[Document], List[Answer], List[str]]:
        contexts = []
        answers = []
        questions = []
        issues = 0
        for example in self.ds:
            if self.datatype == SquadExample:
                analyzed = self.nlp(example.context)
                answer = example.answer
                start_index = None
                end_index = None
                for i, word in enumerate(analyzed.iter_words()):
                    if start_index is None and word.text == answer.text[:len(word.text)]:
                        start_index = i
                    if start_index is not None and word.text == answer.text[-len(word.text):]:
                        end_index = i
                if (start_index is None) or (end_index is None):
                    issues += 1
                    logging.warning(f"Issue while parsing answer '{answer.text}'")
                    continue
                answers.append(NQGDataset.Answer(
                    start_index,
                    end_index - start_index + 1,
                    text=answer.text
                ))
            else:
                analyzed = self.nlp(example.answer.text)
                answers.append(NQGDataset.Answer(
                    start_index=example.answer.answer_start,
                    nb_words=analyzed.num_tokens,
                    text=example.answer.text
                ))
            contexts.append(analyzed)
            questions.append(example.question.question.lower())
        logging.info(f"Issues: {issues}")
        return contexts, answers, questions

    def get_split(self, first_part_size_ratio: float):
        """
        :param first_part_size_ratio: Size ratio of the first returned dataset from the original one.
        :return: A tuple (ds1, ds2) where ds1 is `first_part_size_ratio` of the original dataset
        and ds2 the rest of it.
        """
        c, a, q = self.get_dataset()
        ds = list(zip(c, a, q))
        random.shuffle(ds)
        c, a, q = zip(*ds)
        ds_size = len(c)
        first_part_size = int(first_part_size_ratio * ds_size)
        return c[:first_part_size], a[:first_part_size], q[:first_part_size], c[first_part_size:], a[first_part_size:],\
               q[first_part_size:]
