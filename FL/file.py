import pandas as pd


def save(db, path):
    db.to_csv(path, header=True)


def load(path):
    return pd.read_csv(path, index_col=0)


def append(record, path):
    save(load(path).append(record, ignore_index=True), path)
