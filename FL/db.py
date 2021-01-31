def process(db, key_name, indicator_name):
    def fun(name):
        def _inner(x):
            return len(list(e for e in x if e == name))

        _inner.__name__ = name
        return _inner

    return (
        db.groupby(key_name)[indicator_name]
        .aggregate([fun(g) for g in db[indicator_name].unique()])
        .transform(lambda x: x / sum(x), axis=1)
        .reset_index()
    )


if __name__ == "__main__":
    from file import load

    print(process(load("data/db.csv"), "intensity", "mark"))
