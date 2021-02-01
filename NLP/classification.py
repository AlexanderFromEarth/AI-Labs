from math import log
from functools import reduce

from db import words


def count(db, idcs, word=None):
    return sum(
        db[idx][word]
        for idx in idcs
        for db_word in db[idx]
        if word is not None and db_word == word
    )


def word_prob(db, word, idcs, alpha=1):
    return (count(db, idcs, word) + alpha) / (count(db, idcs) + alpha * len(words(db)))


def class_prob(db, idcs):
    return len(idcs) / len(db)


def naive_bayes(db, request, classes):
    return (
        reduce(
            lambda p, c: p + log(word_prob(db, c, class_)),
            set(request),
            log(class_prob(db, class_)),
        )
        for class_ in classes
    )
