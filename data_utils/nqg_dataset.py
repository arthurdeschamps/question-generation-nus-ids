import logging
import random
from typing import List, Tuple

import stanza
from stanza import Document
from stanza.models.common.doc import Word

from data_utils.parse import read_squad_dataset
from defs import SQUAD_DEV, SQUAD_TRAIN


class DocumentWithWords:
    def __init__(self, document: Document, words: List[Word]):
        super(DocumentWithWords, self).__init__()
        self.document = document
        self.words = words


class NQGDataset:
    class Answer:

        def __init__(self, start_index, nb_words, text):
            super(NQGDataset.Answer, self).__init__()
            self.start_index = start_index
            self.nb_words = nb_words
            self.text = text

    def __init__(self, dataset_type="squad_dev", data_limit=-1):
        super(NQGDataset, self).__init__()
        stanza.download('en')
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,ner')
        if "squad" in dataset_type:
            if dataset_type == "squad_train":
                datapath = SQUAD_TRAIN
            elif dataset_type == "squad_dev":
                datapath = SQUAD_DEV
            else:
                raise NotImplementedError()
            self.ds = read_squad_dataset(
                datapath,
                limit=data_limit
            )
        else:
            raise NotImplementedError()

    def get_dataset(self) -> Tuple[List[Document], List[Answer], List[str]]:
        contexts = []
        answers = []
        questions = []
        issues = 0
        for example in self.ds:
            analyzed = self.nlp(example.context)
            answer = example.answer
            start_index = None
            end_index = None
            i = 0
            for word in analyzed.iter_words():
                if start_index is None and word.text == answer.text[:len(word.text)]:
                    start_index = i
                if start_index is not None and word.text == answer.text[-len(word.text):]:
                    end_index = i
                i += 1
            if (start_index is None) or (end_index is None):
                issues += 1
                logging.warning(f"Issue while parsing answer '{answer.text}'")
                continue
            contexts.append(analyzed)
            answers.append(NQGDataset.Answer(
                start_index,
                end_index - start_index + 1,
                text=answer.text
            ))
            questions.append(example.question.question.lower())
        logging.info(f"Issues: {issues}")
        return contexts, answers, questions

    def get_split(self, first_part_size_ratio: float):
        c, a, q = self.get_dataset()
        ds = list(zip(c, a, q))
        random.shuffle(ds)
        c, a, q = zip(*ds)
        ds_size = len(c)
        first_part_size = int(first_part_size_ratio * ds_size)
        return c[:first_part_size], a[:first_part_size], q[:first_part_size], c[first_part_size:], a[first_part_size:],\
               q[first_part_size:]
