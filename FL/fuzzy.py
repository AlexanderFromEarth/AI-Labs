from scipy.optimize import curve_fit


def train(db, key, funcs, categories):
    return {
        cat: curve_fit(fun, db[key].to_numpy(), db[cat].to_numpy(), maxfev=2400)[0]
        for cat, fun in zip(categories, funcs)
    }


def setup(func, params_to_setup, param_names):
    return lambda *args: func(*args, **dict(zip(param_names, params_to_setup)))
