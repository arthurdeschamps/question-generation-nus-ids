from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np


def uniform_weights(size):
    return [1.0/size for _ in range(size)]


def bleu(cand, ref, to_print=True):
    cand = cand.split(" ")
    ref = ref.split(" ")
    scores = []
    for n in range(1, min(len(cand) + 1, 5)):
        scores.append(sentence_bleu([ref], cand, weights=uniform_weights(n)))
    if to_print:
        for i in range(len(scores)):
            print(f"BLEU-{i+1}: {scores[i]}")
    return scores

def with_split(_refs, _preds):
    return [[_ref.split(" ")] for _ref in _refs], [_pred.split(" ") for _pred in _preds]


ref_squad = "who was the mvp of super bowl i and ii ?"
short = "who was the mvp ?"

print("Short candidates often achieve high BLEU scores, even though they do not include the information required to"
      " make them understandable.")
print("Let's first take a look at an example from SQuAD.")
print(f'Reference: "{ref_squad}"')
print(f'Candidate: "{short}"')
bleu(short, ref_squad)
print()

print("This effect is even more accentuated for references where the the question formulation is a lot longer"
      " than its unique part. See these 4 questions from MedQuAD:")
refs_medquad = (
    "what are the treatments for meningioma ?",
    "what are the treatments for fucosidosis ?",
    "what are the treatments for diabetes ?",
    "what are the treatments for acromegaly ?"
)
candidate_medquad = "what are the treatments for ?"
for ref in refs_medquad:
    print(f'"{ref}"')
print(f'Now say that our candidate is "{candidate_medquad}". Despite being highly generic and containing no specific'
      f' name of disease or condition, thus being highly uninformative, if predicted for each reference question it'
      f' gets the following corpus BLEU-4 score:')
candidates = [candidate_medquad for _ in range(len(refs_medquad))]

refs_medquad, candidates = with_split(refs_medquad, candidates)
print(str(corpus_bleu(refs_medquad, candidates)) + "\n")

over_gen = (
    "who was the mvp of super bowl i and ii and the first man to walk on the moon ?",
    "what causes bronchiectasis in COVID-19 cases ?"
)
refs = (
    ref_squad,
    "what causes bronchiectasis ?"
)
print("Another issue is over-generation. Here are two textbook examples from SQuAD and MedQuAD:")
for ref, gen in zip(refs, over_gen):
    print(f'Reference: "{ref}"')
    print(f'Candidate: "{gen}')
refs, over_gen = with_split(refs, over_gen)
print(f'BLEU-4: {corpus_bleu(refs, over_gen)}')
print("The candidates have all the questions right, but then add details which change their meaning and turn them"
      " into different or off-topic questions.\n")
print("Finally, but probably most importantly, BLEU has no mechanism to cope with paraphrasing/rephrasing, and this"
      "if a big issue especially when we want or model to generate diversified and complex questions.")
print("Again, let us observe a set of reference/candidate pairs where all the candidates have seemingly the "
      "same meaning as their respective reference, and yet get a next to 0 BLEU score:")
rephrase_refs = (
    "what is ( are ) attention deficit hyperactivity disorder ?",
    "how many people are affected by otospondylomegaepiphyseal dysplasia ?",
    "what nationality is hoesung lee ?",
    "what group was responsible for causing more violence in wittenberg ?"
)
rephrase_candidates = (
    "what could ADHD be defined as ?",
    "what is the proportion of the population that has OSMED ?",
    "where does the south korean economist hoesung lee come from ?",
    "in the german city of wittenberg, who caused a rise in violence ?"
)
for (ref, cand) in zip(rephrase_refs, rephrase_candidates):
    print(f'Reference: "{ref}"')
    print(f'Candidate: "{cand}"')

rephrase_refs, rephrase_candidates = with_split(rephrase_refs, rephrase_candidates)
for i in range(1, 5):
    print(f"BLEU-{i}: {corpus_bleu(rephrase_refs, rephrase_candidates, weights=uniform_weights(i))}")
print()
print("METEOR theoretically alleviates some of BLEU's shortcomings, namely that recall is explicitly taken into "
      "consideration which penalizes too short predictions and synonyms are looked up in case words don't match"
      "directly, which solves the problem of rephrasing.")
print("The BLEU-4 score of the small MedQA Handmade dataset, which includes very diversified medical related questions"
      " with many different question formulations, is close to 0:")

from evaluating.evaluate_medqa_handmade import preds, targets
medqa_targets, medqa_preds = with_split(targets, preds)
print(f"BLEU-4: {corpus_bleu(medqa_targets, medqa_preds)}")
print("Now, using METEOR, we obtain:")
meteor_scores = []
for pred, target in zip(preds, targets):
    meteor_scores.append(meteor_score(target, pred))
print(f"METEOR avg: {np.mean(np.array(meteor_scores))}")
