from __future__ import print_function

import os
import sys


class Cleaner(object):
    def __init__(self, func):
        self.func = func

    def __del__(self):
        self.func()


def as_list(t):
    return t if isinstance(t, (tuple, list)) else [t]


def getenv_as(name, type, defval=None):
    env = os.environ.get(name, None)
    return defval if env is None else type(env)


def shape(inputs):
    cshape = []
    if isinstance(inputs, (list, tuple)):
        lshape = None
        for input in inputs:
            ishape = shape(input)
            if lshape is None:
                lshape = ishape
            else:
                assert lshape == ishape
        cshape.extend([len(inputs)] + (lshape or []))
    return cshape


def flatten_nested_tuple(inputs):
    flat = []
    if isinstance(inputs, (list, tuple)):
        for input in inputs:
            flat.extend(flatten_nested_tuple(input))
    else:
        flat.append(inputs)
    return tuple(flat)


def list_copy_append(ilist, item):
    ilist_copy = list(ilist)
    ilist_copy.append(item)
    return ilist_copy


def null_print(*args, **kwargs):
    return


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_print_fn(debug):
    return eprint if debug else null_print
