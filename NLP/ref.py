from functools import reduce

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from gensim.utils import tokenize

from db import words


def koefs(db):
    N = sum(sum(db_.values()) for db_ in db)
    n = len(db)
    K_1 = 9 / (N * n)
    K_2 = (1 + n / 4) ** 2 / (N * n)

    def _inner(word):
        K = (sum(db_[word] for db_ in db) * sum(1 for db_ in db if db_[word])) / (N * n)
        return (K, 1 if K >= K_1 else 2 if K >= K_2 else 0)

    return {word: _inner(word) for word in words(db)}


def keywords(text, db, koefs_=None):
    if koefs_ is None:
        koefs_ = koefs(db)

    request = tokenize(text)
    stemmer = SnowballStemmer("russian", True)

    def _reducer(prev, curr):
        cur = stemmer.stem(curr)
        if koefs_.get(cur) is not None and koefs_[cur][1]:
            if len(prev[1]) == 2:
                return prev[0] + [" ".join([*prev[1], curr])], []
            else:
                return prev[0], [*prev[1], curr]
        else:
            if len(prev[1]) == 2:
                return prev[0] + [" ".join([*prev[1]])], []
            else:
                return prev[0], []

    res = reduce(_reducer, request, ([], []))

    return res[0] + ([" ".join(*res[1])] if len(res[1]) > 1 else [])


def relevant_sents(text, db, koefs_=None):
    if koefs_ is None:
        koefs_ = koefs(db)

    stemmer = SnowballStemmer("russian", True)

    def _inner(tokens):
        return sum(
            1
            for token in tokens
            if koefs_.get(stemmer.stem(token)) and koefs_[stemmer.stem(token)][1]
        ) / len(tokens)

    return {sent: _inner(list(tokenize(sent))) for sent in sent_tokenize(text)}
