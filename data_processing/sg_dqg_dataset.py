import re
from data_processing.dataset import Dataset
from data_processing.parse import read_squad_dataset
import spacy


class SGDQGDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super(SGDQGDataset, self).__init__(*args, **kwargs)
        if self.dataset_name == "squad":
            self.ds = read_squad_dataset(self.datapath, self.data_limit, break_up_paragraphs=True)
            if self.data_limit is not None and self.data_limit > 0:
                self.ds = list(self.ds[i] for i in range(self.data_limit))
        elif self.dataset_name == "hotpotqa":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Unrecognized dataset name {self.dataset_name} for SG DQL.")
        self.tokenizer = spacy.load("en_core_web_sm")

    def get_dataset(self):
        # Removes double whitespaces
        cleaned_ds = {"contexts": [], "answers": [], "questions": []}
        for ex in self.ds:
            cleaned = []
            for s in (ex.context, ex.answer.text, ex.question.question):
                tokens = self.tokenizer(re.sub(' +', ' ', s.replace('\n', ' ').strip()))
                tokens = [token.text for token in tokens]
                if len(tokens) > 1:
                    cleaned.append(" ".join(tokens))
                else:
                    break
            if len(cleaned) == 3:
                cleaned_ds["contexts"].append(cleaned[0])
                cleaned_ds["answers"].append(cleaned[1])
                cleaned_ds["questions"].append(cleaned[2])

        return cleaned_ds.values()
