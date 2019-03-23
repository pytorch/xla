from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import time


class Cleaner(object):

  def __init__(self, func):
    self.func = func

  def __del__(self):
    self.func()


class TmpFolder(object):

  def __init__(self):
    self.name = tempfile.mkdtemp()
    self.cleaner = Cleaner(lambda: shutil.rmtree(self.name))


class SampleGenerator(object):

  def __init__(self, data, target, sample_count):
    self._data = data
    self._target = target
    self._sample_count = sample_count
    self._count = 0

  def __iter__(self):
    return SampleGenerator(self._data, self._target, self._sample_count)

  def __len__(self):
    return self._sample_count

  def __next__(self):
    return self.next()

  def next(self):
    if self._count >= self._sample_count:
      raise StopIteration
    self._count += 1
    return self._data, self._target


class FnDataGenerator(object):

  def __init__(self, func, batch_size, gen_tensor, dims=None, count=1):
    self._func = func
    self._batch_size = batch_size
    self._gen_tensor = gen_tensor
    self._dims = list(dims) if dims else [1]
    self._count = count
    self._emitted = 0

  def __len__(self):
    return self._count

  def __iter__(self):
    return FnDataGenerator(
        self._func,
        self._batch_size,
        self._gen_tensor,
        dims=self._dims,
        count=self._count)

  def __next__(self):
    return self.next()

  def next(self):
    if self._emitted >= self._count:
      raise StopIteration
    data = self._gen_tensor(self._batch_size, *self._dims)
    target = self._func(data)
    self._emitted += 1
    return data, target


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


def get_print_fn(debug=None):
  if debug is None:
    debug = int(os.environ.get('DEBUG', '0'))
  return eprint if debug else null_print


def timed(fn, msg='', printfn=eprint):
  if printfn is None:
    printfn = get_print_fn()
  s = time.time()
  result = fn()
  printfn('{}{:.3f}ms'.format(msg, 1000.0 * (time.time() - s)))
  return result


class TimedScope(object):

  def __init__(self, msg='', printfn=eprint):
    if printfn is None:
      printfn = get_print_fn()
    self._msg = msg
    self._printfn = printfn

  def __enter__(self):
    self._start = time.time()
    return self

  def __exit__(self, type, value, traceback):
    self._printfn('{}{:.3f}ms'.format(self._msg,
                                      1000.0 * (time.time() - self._start)))
