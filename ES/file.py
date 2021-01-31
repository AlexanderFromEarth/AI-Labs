import pandas as pd


def save(db, path):
    db.to_csv(path, header=True)


def load(path):
    return pd.read_csv(path, index_col=0)
