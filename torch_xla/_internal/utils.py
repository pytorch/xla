import functools
import re


def parse_xla_device(device: str):
  m = re.match(r'([A-Z]+):(\d+)$', device)
  if m:
    return (m.group(1), int(m.group(2)))


def run_once(func):
  result = None

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    nonlocal result
    if result is not None:
      result = func(*args, **kwargs)
    return result

  return wrapper