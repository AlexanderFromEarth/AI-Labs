from math import log
from functools import reduce

from db import mk_db, count_words, words


def vectorize(vectorizer, preprocessed_texts, *, db=[], cached_words=[], **kwargs):
    db = list(mk_db(*count_words(preprocessed_texts, cached_words), db))
    _f = vectorizer(corpus=db, words=words(db), **kwargs)
    return ([_f(text) for text in db], _f, db)


def make_vector(preprocessed_request, words, vectorizer):
    return vectorizer(
        {
            k: v
            for k, v in reduce(
                lambda p, c: {**p, c: p.get(c, 0) + 1}, preprocessed_request, {}
            ).items()
            if k in words
        },
    )


def binary_vectorizer(*, words=[], **kwargs):
    return lambda text: {
        **{word: 0 for word in words},
        **{word: int(freq != 0) for word, freq in text.items()},
    }


def count_vectorizer(*, words=[], **kwargs):
    return lambda text: {
        **{word: 0 for word in words},
        **{word: freq for word, freq in text.items()},
    }


def tf_vectorizer(*, words=[], **kwargs):
    return lambda text: {
        **{word: 0 for word in words},
        **{word: freq / sum(text.values()) for word, freq in text.items()},
    }


def idf_vectorizer(corpus, *, smooth=True, **kwargs):
    corpus = list(corpus)
    return {
        word: (1 if smooth else 0)
        + log(
            len(corpus)
            / ((1 if smooth else 0) + sum(int(word in text) for text in corpus)),
        )
        for word in set(word for text in corpus for word in text)
    }


def tfidf_vectorizer(corpus, *, idfs=None, **kwargs):
    if idfs is None:
        idfs = idf_vectorizer(corpus, **kwargs)
    return lambda text: {
        word: freq * idfs[word] for word, freq in tf_vectorizer(**kwargs)(text).items()
    }


if __name__ == "__main__":
    from argparse import ArgumentParser
    from read import read
    from preprocess import preprocess
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument("-i", dest="path", type=str, help="path to text")
    args = parser.parse_args()
    model, vectorizer, db = vectorize(tfidf_vectorizer, preprocess(read(args.path)))
    model = list(model)[:10]
    model = list(
        {
            item: [text.get(item, 0) for text in model]
            for item in list(model[0].keys())[:30]
        }.items()
    )
    model.sort(key=lambda x: x[1][0], reverse=True)
    pprint(model[:10])
