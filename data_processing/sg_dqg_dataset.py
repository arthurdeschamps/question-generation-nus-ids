import re
from logging import warning

from allennlp.predictors import Predictor

from data_processing.dataset import Dataset
from data_processing.parse import read_squad_dataset
import spacy


class SGDQGDataset(Dataset):

    def __init__(self, spacy_pipeline=spacy.load("en_core_web_sm"), *args, **kwargs):
        super(SGDQGDataset, self).__init__(*args, **kwargs)
        if self.dataset_name == "squad":
            self.ds = read_squad_dataset(self.datapath, self.data_limit, break_up_paragraphs=False)
            if self.data_limit is not None and self.data_limit > 0:
                self.ds = list(self.ds[i] for i in range(self.data_limit))
        elif self.dataset_name == "hotpotqa":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Unrecognized dataset name {self.dataset_name} for SG DQL.")
        self.nlp = spacy_pipeline

    def get_dataset(self):
        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",
            cuda_device=0
        )

        def get_doc(s):
            return self.nlp(re.sub(' +', ' ', s.replace('\n', ' ').strip()))

        def get_sentence(doc, is_context=False):
            tokens = [token.text for token in doc]
            if is_context and len(tokens) <= 1:
                raise ValueError("Sentence contains only 1 or no token.")
            result_sentence = " ".join(tokens).strip()
            if result_sentence[-1] != ".":
                result_sentence += " ."
            return result_sentence

        cleaned_ds = {"contexts": [], "answers": [], "questions": []}
        for ex in self.ds:
            try:
                cont_doc = get_doc(ex.context)
                evidences = [get_sentence(evidence, is_context=True) for evidence in cont_doc.sents
                             if (evidence.end - evidence.start) > 1]
                context = predictor.predict(document=" ".join(evidences))
                question = get_sentence(get_doc(ex.question.question))
                answer = get_sentence(get_doc(ex.answer.text))

                cleaned_ds["contexts"].append(context)
                cleaned_ds["answers"].append(answer)
                cleaned_ds["questions"].append(question)
            except TypeError or ValueError or RuntimeError as e:
                warning(e)
                warning(ex.context)

        return cleaned_ds.values()
