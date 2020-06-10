import logging
import random
from typing import List, Tuple

import stanza
from stanza import Document
from tqdm import tqdm

from data_processing.class_defs import SquadExample, QAExample
from data_processing.dataset import Dataset
from data_processing.parse import read_squad_dataset, read_qa_dataset
from data_processing.utils import answer_span


class NQGDataset(Dataset):
    class Answer:

        def __init__(self, start_index, nb_words, text):
            super(NQGDataset.Answer, self).__init__()
            self.start_index = start_index
            self.nb_words = nb_words
            self.text = text

    def __init__(self, break_up_paragraphs=True, *args, mode="train", **kwargs):
        if mode not in ("train", "dev"):
            raise ValueError("Mode should be either 'train' or 'dev'")
        super(NQGDataset, self).__init__(*args, mode=mode, **kwargs)
        stanza.download('en')
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,ner')
        self.datatype = QAExample
        if self.dataset_name == "squad":
            self.datatype = SquadExample
            self.ds = read_squad_dataset(
                self.datapath,
                limit=self.data_limit,
                break_up_paragraphs=break_up_paragraphs
            )
        elif self.dataset_name in ("medquad", "medqa_handmade"):
            self.ds = read_qa_dataset(self.datapath, limit=self.data_limit)
        else:
            raise NotImplementedError()

    def get_dataset(self) -> Tuple[List[Document], List[Answer], List[str]]:
        contexts = []
        answers = []
        questions = []
        issues = 0
        logging.info("Analyzing dataset examples...")
        for example in tqdm(self.ds):
            if self.datatype == SquadExample:
                analyzed = self.nlp(example.context)
                answer = example.answer
                start_index, end_index = answer_span(
                    [word.text for word in analyzed.iter_words()],
                    answer.text.split(' ')
                )
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
