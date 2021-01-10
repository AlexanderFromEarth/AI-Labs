from functools import reduce

from utils import merge


def count_words(preprocessed_texts, cached_words=[]):
    words = list(*cached_words)

    def _reducer(prev_state, current_token):
        if current_token not in words:
            words.append(current_token)
        return {
            **prev_state,
            current_token: prev_state.get(current_token, 0) + 1,
        }

    return (
        [
            reduce(
                _reducer,
                text,
                {},
            )
            for text in preprocessed_texts
        ],
        words,
    )


def mk_db(corpus, words, db=[]):
    return ({**dict.fromkeys(words, 0), **text} for text in merge(db, corpus))


def words(db):
    return db[0].keys()
