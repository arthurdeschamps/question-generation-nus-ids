import re
from functools import reduce
from typing import List
import pandas as pd
from defs import MEDQUAD_DEV, MEDQUAD_TRAIN


def array_to_string(arr: List[str]) -> str:
    # Some words actually contain spaces (mostly irrelevant and noisy data)
    arr = list(w.replace(' ', '\\') for w in arr)
    return reduce(lambda t1, t2: t1 + " " + t2, arr)


def answer_span(context_words, answer_words):
    if len(context_words) < len(answer_words) or len(context_words) == 0 or len(answer_words) == 0:
        return None, None
    context_words = [w.lower() for w in context_words]
    answer_words = [w.lower() for w in answer_words]
    start = None
    i = 0
    for j, context_word in enumerate(context_words):
        if start is None:
            if len(answer_words) == 1 and answer_words[0] in context_word:
                return j, j
            elif context_word == answer_words[0]:
                start = j
                i = 1
                if len(answer_words) == 1:
                    return start, start
            elif answer_words[0].startswith(context_word):
                _start, _end = answer_span(
                    context_words[j+1:],
                    [answer_words[0][len(context_word):]] + answer_words[1:]
                )
                if _start is not None and _end is not None:
                    return _start, _end
        elif start is not None:
            if i+1 == len(answer_words) and answer_words[i] in context_word:
                return start, j
            if context_word != answer_words[i]:
                if answer_words[i].startswith(context_word):
                    _start, _end = answer_span(
                        context_words[j+1:],
                        [answer_words[i][len(context_word):]] + answer_words[i+1:]
                    )
                    if _start == 0 and _end is not None:
                        return start, _end
                    start = None
                    i = 0
            else:
                if i + 1 == len(answer_words):
                    return start, start + i
                i += 1

    return start, None


def medquad_question_type_stats():
    dev_set = pd.read_csv(MEDQUAD_DEV, comment=None, delimiter='|')
    train_set = pd.read_csv(MEDQUAD_TRAIN, comment=None, delimiter='|')
    stats = {
        "what is ( are )": 0,
        "what are the symptoms of": 0,
        "what are the treatments for": 0,
        "how many people are affected by": 0,
        "what are the stages of": 0,
        "what causes": 0,
        "is .* inherited": 0,
        "how to diagnose": 0,
        "what research ( or clinical trials ) is being done for": 0,
        "what to do for": 0,
        "how to prevent": 0,
        "what are the genetic changes related to": 0,
        "who is at risk for": 0,
        "do you have information about": 0,
        "what is the outlook for": 0,
        "is there any treatment for": 0,
        "what are the symptoms for": 0,
        "how can patients prevent": 0,
        "what are the complications of": 0,
        "how can .* be prevented": 0,
        "how can .* be treated": 0,
        "what types of infections does": 0,
        "how can .* be diagnosed for": 0,
        "are certain people at risk of": 0,
        "what is the history of": 0,
        "what is the prognosis": 0,
        "how common is": 0,
        "what research is being done for": 0,
        "what are the signs and symptoms of": 0,
        "where to find support for": 0,
        "what are public health agencies doing to prevent or control": 0,
        "what else can be done to prevent": 0,
        "how common are": 0,
        "how is .* diagnosed": 0,
        "what are": 0,
        "what is": 0,
        "what can i do to prevent": 0,
        "are there complications from": 0,
    }
    for ds in (train_set, dev_set):
        for _, ex in ds.iterrows():
            known_formulation = False
            for formulation in stats.keys():
                regex = formulation.replace(')', '\)').replace('(', '\(')
                regex = f"^{regex}.*"
                if re.match(regex, ex["question"]):
                    stats[formulation] += 1
                    known_formulation = True
                    break
            if not known_formulation:
                print(ex["question"])
    total = 0.0
    for formulation, occ in stats.items():
        total += occ
        print(f"{formulation[0].upper()}{formulation[1:]} & {occ} \\\\")
    print(f"Average: {total / len(stats)}")


if __name__ == '__main__':
    medquad_question_type_stats()
