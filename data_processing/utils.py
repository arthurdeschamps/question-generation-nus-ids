from functools import reduce
from typing import List


def array_to_string(arr: List[str]) -> str:
    # Some words actually contain spaces (mostly irrelevant and noisy data)
    arr = list(w.replace(' ', '\\') for w in arr)
    return reduce(lambda t1, t2: t1 + " " + t2, arr)
