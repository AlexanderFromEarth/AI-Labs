from functools import reduce

from db import words
from vectorize import make_vector


def calc_centroids(model, words, idcs):
    return {
        word: reduce(lambda p, c: p + model[c][word], idcs, 0) / len(idcs)
        for word in words
    }


def rocchio(preprocessed_request, model, words, vectorizer, classes, metric):
    return (
        metric(
            calc_centroids(model, words, class_).values(),
            make_vector(preprocessed_request, words, vectorizer).values(),
        )
        for class_ in classes
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from read import read
    import preprocess
    from vectorize import vectorize, tfidf_vectorizer
    from metrics import euclide
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument("-i1", dest="path1", type=str, help="path to text")
    parser.add_argument("-i2", dest="path2", type=str, help="path to text")
    parser.add_argument("-r", dest="request", type=str, help="request")
    args = parser.parse_args()

    first_class_texts = list(read(args.path1))
    second_class_texts = list(read(args.path2))
    count = len(first_class_texts) + len(second_class_texts)
    classes = (
        range(0, len(first_class_texts)),
        range(len(first_class_texts), count),
    )
    preprocess.preprocessors.append(preprocess.rm_stop_words)
    model, vectorizer, db = vectorize(
        tfidf_vectorizer,
        preprocess.preprocess(first_class_texts + second_class_texts),
    )
    res = list(
        rocchio(
            list(preprocess.preprocess([args.request]))[0],
            model,
            words(db),
            vectorizer,
            classes,
            euclide,
        )
    )
    res.sort()
    pprint(res)
