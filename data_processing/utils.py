from functools import reduce
from typing import List


def array_to_string(arr: List[str]) -> str:
    # Some words actually contain spaces (mostly irrelevant and noisy data)
    arr = list(w.replace(' ', '\\') for w in arr)
    return reduce(lambda t1, t2: t1 + " " + t2, arr)


def answer_span(context_words, answer_words):
    start = None
    i = 0
    for j, context_word in enumerate(context_words):
        if start is None and context_word == answer_words[0]:
            start = j
            i = 1
            if len(answer_words) == 1:
                return start, start
        elif start is not None:
            if context_word != answer_words[i]:
                start = None
                i = 0
            else:
                if i + 1 == len(answer_words):
                    return start, start + i
                i += 1

    return start, None
