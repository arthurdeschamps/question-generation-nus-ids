from tqdm import tqdm

from data_processing.nqg_dataset import NQGDataset


class MPQGDataset(NQGDataset):
    def get_dataset(self):
        contexts = []
        answers = []
        questions = []
        for example in tqdm(self.ds[:self.data_limit]):
            contexts.append(self.nlp(example.context))
            questions.append(self.nlp(example.question.question))
            answers.append(self.nlp(example.answer.text))
        return contexts, answers, questions
