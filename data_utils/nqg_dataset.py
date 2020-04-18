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

    def get_dataset(self) -> Tuple[List[Document], List[Answer]]:
        contexts = []
        answers = []
        for datapoint in self.ds:
            for paragraph in datapoint.paragraphs:
                analyzed = self.nlp(paragraph.context)
                for qa in paragraph.qas:
                    for answer in qa.answers:
                        answer_start_index = None
                        answer_end_index = None
                        answer_end_char_ind = answer.answer_start + len(answer.text)
                        i = 0
                        for word in analyzed.iter_words():
                            if f"start_char={str(answer.answer_start)}" in word.misc:
                                answer_start_index = i
                            if answer_start_index is not None:
                                end_position = int(word.misc[word.misc.index("end_char=")+9:])
                                if answer_end_char_ind <= end_position:
                                    answer_end_index = i
                                    break
                            i += 1
                        if (answer_start_index is None) or (answer_end_index is None):
                            raise AssertionError(f"Issue while parsing answer {answer}")
                        contexts.append(analyzed)
                        answers.append(NQGDataset.Answer(
                            answer_start_index,
                            answer_end_index - answer_start_index + 1,
                            text=answer.text
                        ))
        return contexts, answers
