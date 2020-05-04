from functools import reduce

import pandas as pd
from defs import MEDQUAD_FOR_NQG_DATASET, NQG_MEDQUAD_PREDS_OUTPUT_PATH
import nltk


path = f"{MEDQUAD_FOR_NQG_DATASET}/test/data.txt."
targets = pd.read_csv(f"{path}target.txt", header=None, sep='\n')
test_answers = pd.read_csv(f"{path}source.txt", header=None, sep='\n')

train_answers = pd.read_csv(f"{MEDQUAD_FOR_NQG_DATASET}/train/data.txt.source.txt", sep='\n', header=None)

# loads predictions
preds = pd.read_csv(NQG_MEDQUAD_PREDS_OUTPUT_PATH, header=None, sep='\n')

# Groups indices by question
targets = targets.groupby(targets.columns[0])

distribution = {
    "0.5": 0, "0.7": 0, "0.8": 0, "1": 0
}

corpus_macro_bleu = []
total = 0
for _, group in targets:
    indices = group.index
    reference = group.values[0]
    predicted_questions = preds.iloc[indices].values
    answers = test_answers.iloc[indices].values
    exists_in_train_set = False
    for answer in answers:
        answer = answer[0]
        if len(answer) > 0 and train_answers[train_answers[train_answers.columns[0]].str.contains(answer, regex=False)].dropna().size > 0:
            exists_in_train_set = True
            break
    if exists_in_train_set:
        continue
    bleu_avg = [nltk.bleu(reference, predicted_question[0], weights=(0.25, 0.25, 0.25, 0.25)) for
                predicted_question in predicted_questions]
    bleu_avg = sum(bleu_avg) / len(bleu_avg)
    corpus_macro_bleu.append(bleu_avg)
    if bleu_avg >= 0.5:
        distribution["0.5"] += 1
    if bleu_avg >= 0.7:
        distribution["0.7"] += 1
    if bleu_avg >= 0.8:
        distribution["0.8"] += 1
    if bleu_avg == 1:
        distribution["1"] += 1
    total += 1

    if len(predicted_questions) > 1:
        pred_str: str = reduce(lambda p1, p2: p1[0] + ' \n' + p2[0], predicted_questions)
    else:
        pred_str = predicted_questions[0][0]
    if len(answers) > 1:
        answers_str = reduce(lambda a1, a2: a1[0] + '\n' + a2[0], answers)
    else:
        answers_str = answers[0][0]

    if 0.8 < bleu_avg < 1:
        print(f"Gold question: {reference[0]} \nAnswer:\n{answers_str}\n" +
              f"Predictions: \n{pred_str}\nBLEU-4: {bleu_avg}\n")

print(f"BLEU-4 macro avg: {sum(corpus_macro_bleu) / len(corpus_macro_bleu)}")
for quantile, size in distribution.items():
    print(f"BLEU-4 >= {quantile}: {size}")
print(f"Total: {total}")
