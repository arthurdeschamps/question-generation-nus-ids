import re
from logging import warning
import nltk
from data_processing.dataset import Dataset
from data_processing.parse import read_squad_dataset
from data_processing.utils import answer_span


class SGDQGDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super(SGDQGDataset, self).__init__(*args, **kwargs)
        if self.dataset_name == "squad":
            self.ds = read_squad_dataset(self.datapath, self.data_limit, break_up_paragraphs=True)
            self.ds = list(self.ds[i] for i in range(self.data_limit))
        elif self.dataset_name == "hotpotqa":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Unrecognized dataset name {self.dataset_name} for SG DQL.")

    def get_dataset(self):
        # Removes double whitespaces
        def clean(s):
            return " ".join(nltk.word_tokenize(re.sub(' +', ' ', s)))

        questions = list(clean(ex.question.question) for ex in self.ds)
        answers = list(clean(ex.answer.text) for ex in self.ds)
        contexts = list(clean(ex.context) for ex in self.ds)
        for i in range(len(contexts)):
            answer = self.ds[i].answer
            context = contexts[i].split(' ')
        assert len(questions) == len(answers) == len(contexts)
        return contexts, answers, questions
