import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from defs import NQG_MEDQA_HANDMADE_PREDS_OUTPUT_PATH, MEDQA_HANDMADE_FOR_NQG_DATASET

distribution = {
    "total": 0, "0.1": 0, "0.2": 0, "0.3": 0, "0.5": 0, "1": 0
}

preds = np.loadtxt(NQG_MEDQA_HANDMADE_PREDS_OUTPUT_PATH, delimiter='\n', dtype=str)
targets = np.loadtxt(f"{MEDQA_HANDMADE_FOR_NQG_DATASET}/test/data.txt.target.txt", delimiter='\n', dtype=str)
answers = np.loadtxt(f"{MEDQA_HANDMADE_FOR_NQG_DATASET}/test/data.txt.source.txt", delimiter='\n', dtype=str)

for (pred, target, answer) in zip(preds, targets, answers):
    print(f"Target: {target}")
    print(f"Prediction: {pred}")
    print(f"Answer: {answer}")
    bleu = {}
    for i in range(1, 5):
        bleu[i] = sentence_bleu(target, pred, weights=list(1.0/i for _ in range(i)))
        print(f"BLEU-{i}: {bleu[i]}")
    for threshold in distribution.keys():
        if threshold != "total":
            if bleu[1] >= float(threshold):
                distribution[threshold] += 1
    distribution["total"] += 1
    print()

print(f"Total: {distribution['total']}")
for threshold, size in distribution.items():
    if threshold != "total":
        print(f"BLEU-1 >= {threshold}: {size}")
