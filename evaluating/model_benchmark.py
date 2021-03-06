import json
import sys

from evaluating.rouge_score import rouge_l_sentence_level as rouge_l
import nltk.translate.bleu_score as bleu
from nltk.translate.meteor_score import meteor_score as meteor
from nltk.metrics import scores
import pandas as pd
import numpy as np
from defs import NQG_MEDQUAD_DATASET, NQG_MEDQUAD_PREDS_OUTPUT_PATH, NQG_SQUAD_PREDS_OUTPUT_PATH, \
    NQG_SQUAD_GA_PREDS_OUTPUT_PATH, NQG_SQUAD_NA_PREDS_OUTPUT_PATH, NQG_SQUAD_NER_PREDS_OUTPUT_PATH, \
    NQG_SQUAD_DATASET, NQG_SQUAD_NER_DATASET, NQG_SQUAD_TESTGA_PREDS_OUTPUT_PATH, SG_DQG_HOTPOT_PREDS_PATH, \
    REPEAT_Q_SQUAD_DATA_DIR, REPEAT_Q_SQUAD_OUTPUT_FILEPATH, ASS2S_PROCESSED_SQUAD_DIR, SG_DQG_SQUAD_PREDS_PATH, \
    REPEAT_Q_PREDS_OUTPUT_DIR, ASS2S_PRED_DIR, ASS2S_DIR, ROOT_DIR

sys.path.append(ASS2S_DIR + '/submodule/')
sys.path.append(f"{ROOT_DIR}/evaluating/coco-caption")
from mytools import remove_adjacent_duplicate_grams
from pycocoevalcap.meteor.meteor import Meteor


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
    bleu_1 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(1.0,))
    print(f"BLEU-1: {bleu_1}")
    bleu_2 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(0.5, 0.5))
    print(f"BLEU-2: {bleu_2}")
    bleu_3 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(1.0 / 3, 1.0 / 3, 1.0 / 3))
    print(f"BLEU-3: {bleu_3}")
    bleu_4 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-4: {bleu_4}")
    # Sentences level ROUGE-L with beta = P_lcs / (R_lcs + 1e-12)
    rouge_l_sentence_level = 100 * rouge_l(corpus_candidates_split, corpus_references_split)
    print(f"ROUGE-L: {rouge_l_sentence_level}")
    meteor_score = 100 * np.mean(np.array([meteor(references, candidate)
                                     for (references, candidate) in zip(corpus_references, corpus_candidates)]))
    # meteor_score = Meteor().compute_score(
    #     gts={i: corpus_references[i] for i in range(len(corpus_references))},
    #     res={i: [corpus_candidates[i]] for i in range(len(corpus_candidates))}
    # )
    print(f"METEOR: {meteor_score}")
    f1_score = 100 * corpus_f1_score(corpus_candidates_split, corpus_references_split)
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


def get_sg_dqg_predictions(pred_path):
    preds = []
    refs = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("<gold>"):
                refs.append([line[len("<gold>\t"):]])
            elif line.startswith("<pred>"):
                preds.append(line[len("<pred>\t"):])
    return preds, refs


if __name__ == '__main__':

    models = (
        "nqg_squad", "nqg_squad_ga", "nqg_squad_na", "nqg_squad_ner", "nqg_medquad", "nqg_squad_testga",
        "sg_dqg_hotpotqa", "ass2s_squad", "sg_dqg_squad", "repeat_q_squad", "repeat_q_squad_mapped_triples",
        "thesis_bad_example", "repeat_q_squad_mturk", "ass2s_squad_repeat_q_mturk", "ass2s_squad_repeat_q"
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str,
                        help=f"Name of the model to evaluate.",
                        choices=models)
    parser.add_argument('--collate_ngrams', action='store_true', help='Removes n-gram duplicates.')

    args = parser.parse_args()

    model = args.model_name
    path = None
    preds_path = None
    check_trainset = False
    candidates, references = None, None

    if "nqg" in model:
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

        path = f"{path}/test/data.txt."
        _targets = pd.read_csv(f"{path}target.txt", header=None, sep='\n')
        _test_passages = pd.DataFrame(np.loadtxt(f"{path}source.txt", delimiter='\n', dtype=str, comments=None))
        _train_passages = pd.read_csv(f"{NQG_MEDQUAD_DATASET}/train/data.txt.source.txt", sep='\n', header=None)
        # loads predictions
        _preds = pd.read_csv(preds_path, header=None, sep='\n')
        for i in _preds.index:
            # Removes copy characters [[copied word]] -> copied word
            _preds.at[i, 0] = _preds.iloc[i, 0].replace("[[", "").replace("]]", "")
            # Collate n-grams
            if args.collate_ngrams:
                _preds.at[i, 0] = remove_adjacent_duplicate_grams(_preds.iloc[i, 0])
        if check_trainset:
            candidates, references = prepare_for_eval(_preds, _targets, _test_passages, _train_passages)
        else:
            candidates = np.array(_preds.values).reshape((-1,))
            references = np.array(_targets.values).reshape((-1, 1))

    elif "sg_dqg" in model:
        if model == "sg_dqg_hotpotqa":
            candidates, references = get_sg_dqg_predictions(SG_DQG_HOTPOT_PREDS_PATH)
        elif model == "sg_dqg_squad":
            candidates, references = get_sg_dqg_predictions(SG_DQG_SQUAD_PREDS_PATH)

    elif "ass2s" in model:
        candidate_file = ASS2S_PRED_DIR
        reference_dir = ASS2S_PROCESSED_SQUAD_DIR
        if model == "ass2s_squad":
            candidate_file += "/squad_original_preds.txt"
        elif model == "ass2s_squad_repeat_q_mturk":
            candidate_file += "/squad_mturk_preds.txt"
            reference_dir = reference_dir.replace("squad", "squad_mturk")
        elif model == "ass2s_squad_repeat_q":
            candidate_file += "/squad_preds.txt"
        else:
            raise ValueError(f"Model {model} not recognized.")

        candidates = np.array(pd.read_csv(candidate_file, header=None, sep='\n')).reshape((-1,))
        references = np.array(
            pd.read_csv(f"{reference_dir}/filtered_txt/test_question_origin.txt", header=None, sep='\n')
        ).reshape((-1, 1))
        assert candidates.shape[0] == references.shape[0]

    elif "repeat_q" in model:
        if model == "repeat_q_squad":
            target_dir = REPEAT_Q_SQUAD_DATA_DIR
            prediction_file = f"{REPEAT_Q_PREDS_OUTPUT_DIR}/full_dataset_sentence_facts_predictions.txt"
        elif model == "repeat_q_squad_mapped_triples":
            target_dir = f"{REPEAT_Q_SQUAD_DATA_DIR}_mapped_triples"
            prediction_file = f"{REPEAT_Q_PREDS_OUTPUT_DIR}/mixed_data_predictions.txt"
        elif model == "repeat_q_squad_mturk":
            target_dir = REPEAT_Q_SQUAD_DATA_DIR
            prediction_file = f"{REPEAT_Q_PREDS_OUTPUT_DIR}/mturk_attempts_predictions.txt"
        else:
            raise NotImplementedError(f"'{model}' not recognized")

        with open(f"{target_dir}/test.data.json", mode='r') as test_file:
            test_data = json.load(test_file)

        references = [[d["target"]] for d in test_data]
        candidates = np.array(pd.read_csv(prediction_file, header=None, sep='\n', comment=None)).reshape((-1,))

    elif model == "thesis_bad_examples":
        ref = "when did martin luther , O.S.A. , who was ordained to the priesthood in 1507 , publish his translation of the new testament ?"
        h1 = "when did publish his translation of the new testament ?"
        h2 = "augustinian monk martin luther published a new testament of his own translation in year ?"

        benchmark([h1], [[ref]])
        benchmark([h2], [[ref]])

        exit()

    benchmark(candidates, references)
