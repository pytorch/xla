import functools
import logging
from typing import TypeVar

FN = TypeVar('FN')


def deprecated(module, new: FN) -> FN:
  already_warned = [False]

  @functools.wraps(new)
  def wrapped(*args, **kwargs):
    if not already_warned[0]:
      logging.warning(
          f'{module.__name__}.{new.__name__} is deprecated. Use {new.__module__}.{new.__name__} instead.'
      )
      already_warned[0] = True

    return new(*args, **kwargs)

  return wrapped


def register_deprecated(module, new: FN):
  setattr(module, new.__name__, deprecated(module, new))
