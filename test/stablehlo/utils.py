import functools


@functools.lru_cache
def has_tf_package() -> bool:
  try:
    import tensorflow
    return tensorflow is not None
  except ImportError:
    return False
