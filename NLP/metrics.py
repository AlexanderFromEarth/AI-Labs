from math import sqrt
from functools import reduce

from utils import soft_division
from vectorize import make_vector


def calc_metrics(metric, preprocessed_request, model, vectorizer, words):
    request_vector = make_vector(preprocessed_request, words, vectorizer)
    return [metric(request_vector.values(), vector.values()) for vector in model]


def cossine(vector_1, vector_2):
    return soft_division(
        reduce(lambda p, c: p + c[0] * c[1], zip(vector_1, vector_2), 0),
        (
            sqrt(reduce(lambda p, c: p + c * c, vector_1, 0))
            * sqrt(reduce(lambda p, c: p + c * c, vector_2, 0))
        ),
    )


def euclide(vector_1, vector_2):
    return sqrt(reduce(lambda p, c: p + (c[0] - c[1]) ** 2, zip(vector_1, vector_2), 0))


def jaccard(vector_1, vector_2):
    return reduce(
        lambda p, c: p + min(c[0], c[1]), zip(vector_1, vector_2), 0
    ) / reduce(lambda p, c: p + max(c[0], c[1]), zip(vector_1, vector_2), 0)


def dice(vector_1, vector_2):
    return (
        2
        * reduce(lambda p, c: p + min(c[0], c[1]), zip(vector_1, vector_2), 0)
        / reduce(lambda p, c: p + c[0] + c[1], zip(vector_1, vector_2), 0)
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from read import read
    from preprocess import preprocess
    from vectorize import vectorize, tfidf_vectorizer
    from db import words
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument("-i", dest="path", type=str, help="path to text")
    parser.add_argument("-r", dest="request", type=str, help="request")
    args = parser.parse_args()
    model, vectorizer, db = vectorize(tfidf_vectorizer, preprocess(read(args.path)))
    pprint(
        [
            calc_metrics(
                metric,
                list(preprocess([args.request]))[0],
                model,
                vectorizer,
                words(db),
            )
            for metric in (cossine, jaccard, dice)
        ]
    )
