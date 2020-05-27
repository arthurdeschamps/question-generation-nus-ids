from evaluating.rouge_score import rouge_l_sentence_level as rouge_l
import nltk.translate.bleu_score as bleu
from nltk.translate.meteor_score import meteor_score as meteor
from nltk.metrics import scores
import pandas as pd
import numpy as np
from defs import NQG_MEDQUAD_DATASET, NQG_MEDQUAD_PREDS_OUTPUT_PATH, NQG_SQUAD_PREDS_OUTPUT_PATH, \
    NQG_SQUAD_GA_PREDS_OUTPUT_PATH, NQG_SQUAD_NA_PREDS_OUTPUT_PATH, NQG_SQUAD_NER_PREDS_OUTPUT_PATH, \
    NQG_SQUAD_DATASET, NQG_SQUAD_NER_DATASET, NQG_SQUAD_TESTGA_PREDS_OUTPUT_PATH


def corpus_f1_score(corpus_candidates, corpus_references):
    def f1_max(candidate, references):
        f1 = 0.0
        for ref in references:
            f1 = max(f1, scores.f_measure(set(ref), set(candidate)))
        return f1

    return np.mean(np.array([f1_max(candidate, references)
                             for (references, candidate) in zip(corpus_references, corpus_candidates)]))


def benchmark(corpus_candidates: np.ndarray, corpus_references: np.ndarray):
    corpus_candidates_split = [candidate.strip().split(' ') for candidate in corpus_candidates]
    corpus_references_split = [[reference.strip().split(' ') for reference in refs] for refs in corpus_references]
    bleu_1 = bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(1.0,))
    print(f"BLEU-1: {bleu_1}")
    bleu_2 = bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(0.5, 0.5))
    print(f"BLEU-2: {bleu_2}")
    bleu_3 = bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(1.0 / 3, 1.0 / 3, 1.0 / 3))
    print(f"BLEU-3: {bleu_3}")
    bleu_4 = bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-4: {bleu_4}")
    # Sentences level ROUGE-L with beta = P_lcs / (R_lcs + 1e-12)
    rouge_l_sentence_level = rouge_l(corpus_candidates_split, corpus_references_split)
    print(f"ROUGE-L: {rouge_l_sentence_level}")
    meteor_score = np.mean(np.array([meteor(references, candidate)
                                     for (references, candidate) in zip(corpus_references, corpus_candidates)]))
    print(f"METEOR macro average: {meteor_score}")
    f1_score = corpus_f1_score(corpus_candidates_split, corpus_references_split)
    print(f"F1 macro average: {f1_score}")


def prepare_for_eval(preds: pd.DataFrame, targets: pd.DataFrame, test_passages: pd.DataFrame,
                     train_passages: pd.DataFrame):
    corpus_candidates = {}
    corpus_references = {}

    for candidate, reference, source in zip(preds.values, targets.values, test_passages.values):
        # Ignores passages that were already contained in the training set
        candidate = candidate[0]
        reference = reference[0]
        source = source[0]
        if not train_passages[train_passages.columns[0]].str.contains(source, regex=False).any():
            if source in corpus_references:
                corpus_references[source].append(reference)
                if len(candidate) > len(corpus_candidates[source]):
                    corpus_candidates[source] = candidate
            else:
                corpus_references[source] = [reference]
                corpus_candidates[source] = candidate

    corpus_references = corpus_references.values()
    corpus_candidates = corpus_candidates.values()
    assert len(corpus_candidates) == len(corpus_references)
    return corpus_candidates, corpus_references


if __name__ == '__main__':

    models = (
        "nqg_squad", "nqg_squad_ga", "nqg_squad_na", "nqg_squad_ner", "nqg_medquad", "nqg_squad_testga",
        "sg_dqg_hotpotqa"
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str,
                        help=f"Name of the model to evaluate.",
                        choices=models)

    args = parser.parse_args()

    model = args.model_name
    path = None
    preds_path = None
    check_trainset = False
    if model == "nqg_squad":
        preds_path = NQG_SQUAD_PREDS_OUTPUT_PATH
        path = NQG_SQUAD_DATASET
    if model == "nqg_squad_ga":
        preds_path = NQG_SQUAD_GA_PREDS_OUTPUT_PATH
        path = NQG_SQUAD_DATASET
    if model == "nqg_squad_na":
        preds_path = NQG_SQUAD_NA_PREDS_OUTPUT_PATH
        path = NQG_SQUAD_DATASET
    if model == "nqg_squad_ner":
        preds_path = NQG_SQUAD_NER_PREDS_OUTPUT_PATH
        path = NQG_SQUAD_NER_DATASET
    if model == "nqg_medquad":
        preds_path = NQG_MEDQUAD_PREDS_OUTPUT_PATH
        path = NQG_MEDQUAD_DATASET
        check_trainset = True
    if model == "nqg_squad_testga":
        preds_path = NQG_SQUAD_TESTGA_PREDS_OUTPUT_PATH
        path = NQG_SQUAD_DATASET
    if model == "sg_dqg_hopotqa":
        raise NotImplementedError()

    path = f"{path}/test/data.txt."
    _targets = pd.read_csv(f"{path}target.txt", header=None, sep='\n')
    _test_passages = pd.DataFrame(np.loadtxt(f"{path}source.txt", delimiter='\n', dtype=str, comments=None))
    _train_passages = pd.read_csv(f"{NQG_MEDQUAD_DATASET}/train/data.txt.source.txt", sep='\n', header=None)
    # loads predictions
    _preds = pd.read_csv(preds_path, header=None, sep='\n')
    # Removes copy characters [[copied word]] -> copied word
    for i in _preds.index:
        _preds.at[i, 0] = _preds.iloc[i, 0].replace("[[", "").replace("]]", "")
    if check_trainset:
        candidates, references = prepare_for_eval(_preds, _targets, _test_passages, _train_passages)
    else:
        candidates = np.array(_preds.values).reshape((-1,))
        references = np.array(_targets.values).reshape((-1, 1))
    benchmark(candidates, references)
