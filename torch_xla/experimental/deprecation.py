import functools
import logging
import importlib
from typing import Optional, TypeVar

FN = TypeVar('FN')


def deprecated(module,
               new: FN,
               old_name: Optional[str] = None,
               extra_msg: Optional[str] = None) -> FN:
  already_warned = [False]
  old_name = old_name or new.__name__

  @functools.wraps(new)
  def wrapped(*args, **kwargs):
    if not already_warned[0]:
      warning_msg = f'{module.__name__}.{old_name} is deprecated. Use {new.__module__}.{new.__name__} instead.'
      if extra_msg:
        warning_msg += f' {extra_msg}'
      logging.warning(warning_msg)
      already_warned[0] = True

    return new(*args, **kwargs)

  return wrapped


def mark_deprecated(new: FN, extra_msg: Optional[str] = None) -> FN:
  """Decorator to mark a function as deprecated and map to new function.

  Args:
    new: new function that we map to. Need to include the path the new function that is in.
    extra_msg: extra message to include in the warning. For example, which release version will completely deprecate the function.

  Returns:
    Wrapper of the new function.
  """

  def decorator(func):
    return deprecated(
        importlib.import_module(func.__module__), new, old_name=func.__name__)

  return decorator


def register_deprecated(module, new: FN):
  setattr(module, new.__name__, deprecated(module, new), extra_msg)
