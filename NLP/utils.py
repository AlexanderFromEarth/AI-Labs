from functools import reduce


def identity(x, *args, **kwargs):
    return (x,) + args + tuple(kwargs.values()) if args or kwargs else x


def merge(*iterators):
    for values in iterators:
        for value in values:
            yield value


def compose(*funcs):
    return reduce(
        lambda p, c: (lambda *args, **kwargs: c(p(*args, **kwargs))), funcs, identity
    )


def soft_division(f, s, *, ret=0.0):
    try:
        return f / s
    except ZeroDivisionError:
        return ret
