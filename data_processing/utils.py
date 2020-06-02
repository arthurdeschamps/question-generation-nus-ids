from functools import reduce
from typing import List


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
