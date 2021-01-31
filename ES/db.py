def process(db, hypo_cols, evid_cols):
    grouped = db.groupby(hypo_cols)
    return {
        g: {v[0]: v[1:] for v in grouped.get_group(g)[evid_cols].values.tolist()}
        for g in grouped.groups
    }


def hypothesis_names(data):
    return [k[0] for k in data]


def hypothesis_values(data):
    return [k[1] for k in data]


def evidence_names(data):
    return list(dict.fromkeys([v for k in data for v in data[k]]))


def evidence_values(data):
    return [[v for v in data[k].values()] for k in data]


if __name__ == "__main__":
    from file import load

    data = process(load("data/db.csv"), ["desease", "ph"], ["symptom", "peh", "penh"])
    print(data)
    print(hypothesis_names(data))
    print(hypothesis_values(data))
    print(evidence_names(data))
    print(evidence_values(data))
