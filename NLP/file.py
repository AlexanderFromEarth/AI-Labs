import json
from functools import reduce

import pandas as pd

import db as data


def save(db, path):
    pd.DataFrame.from_dict(db).transpose().to_csv(path, header=True)


def load(path):
    return pd.read_csv(path).transpose().to_dict("records")


def append(db, path):
    _db = load(path)
    save(
        data.mk_db(
            db,
            reduce(
                lambda p, c: [*p, c] if c not in p else p,
                data.merge(data.words(_db), data.words(db)),
                [],
            ),
            _db,
        )
    )


def save_idfs(idfs, path):
    with open(path, "w") as f:
        json.dump(idfs, f)


def load_idfs(path):
    with open(path, "r") as f:
        return json.load(f)
