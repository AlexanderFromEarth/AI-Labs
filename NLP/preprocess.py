from gensim.utils import tokenize
from gensim.parsing.porter import PorterStemmer

from utils import compose


preprocessors = [tokenize, PorterStemmer().stem_documents]
stop_words = []


def rm_stop_words(text):
    return [token for token in text if token not in stop_words]


def preprocess(texts):
    return (compose(*preprocessors)(text) for text in texts)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from read import read
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument("-i", dest="path", type=str, help="path to text")
    args = parser.parse_args()
    pprint(list(preprocess(read(args.path))))
