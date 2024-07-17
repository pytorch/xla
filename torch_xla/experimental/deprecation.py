import functools
import logging
from typing import TypeVar

FN = TypeVar('FN')


def deprecated(module, new: FN, old_name=None) -> FN:
  already_warned = [False]
  old_name = old_name or new.__name__

  @functools.wraps(new)
  def wrapped(*args, **kwargs):
    if not already_warned[0]:
      logging.warning(
          f'{module.__name__}.{old_name} is deprecated. Use {new.__module__}.{new.__name__} instead.'
      )
      already_warned[0] = True

    return new(*args, **kwargs)

  return wrapped


def mark_deprecated(module, new: FN) -> FN:
  """Decorator to mark a function as deprecated and map to new function.

  Args:
    module: current module of the deprecated function that is in. Assume current module name is X, you can use `from . import X` and pass X here.
    new: new function that we map to. Need to include the path the new function that is in.

  Returns:
    Wrapper of the new function.
  """
  already_warned = [False]

  def decorator(func):

    @functools.wraps(new)
    def wrapped(*args, **kwargs):
      if not already_warned[0]:
        logging.warning(
            f'{module.__name__}.{func.__name__} is deprecated. Use {new.__module__}.{new.__name__} instead.'
        )
        already_warned[0] = True

      return new(*args, **kwargs)

    return wrapped

  return decorator


def register_deprecated(module, new: FN):
  setattr(module, new.__name__, deprecated(module, new))
