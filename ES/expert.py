from functools import reduce


def expertize(hypothesises, evidences, request):
    return (
        reduce(
            lambda p, c: ((c[1][0] if request[c[0]] else (1 - c[1][0])) * p)
            / (
                (c[1][0] if request[c[0]] else (1 - c[1][0])) * p
                + (c[1][1] if request[c[0]] else (1 - c[1][1])) * (1 - p)
            ),
            enumerate(evidences[idx]),
            hypothesis,
        )
        for idx, hypothesis in enumerate(hypothesises)
    )


if __name__ == "__main__":
    from file import load
    from db import (
        process,
        evidence_values,
        hypothesis_names,
        hypothesis_values,
    )

    data = process(load("data/db.csv"), ["desease", "ph"], ["symptom", "peh", "penh"])
    res = expertize(
        hypothesis_values(data),
        evidence_values(data),
        (True, False, True, False, True),
    )
    print(dict(zip(hypothesis_names(data), res)))
