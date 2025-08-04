import torch


def _is_in_bad_fork():
  """Returns `False` as forking is not applicable in the same way as CUDA."""
  return False


def manual_seed_all(seed):
  """A placeholder for API compatibility; does not affect JAX's PRNG."""
  pass


def device_count():
  """Returns `1` as JAX manages devices as a single logical device."""
  return 1


def get_rng_state():
  """Returns an empty list for API compatibility."""
  return []


def set_rng_state(new_state, device):
  """A placeholder for API compatibility; does not affect JAX's PRNG."""
  pass


def is_available():
  """Returns `True` if JAX is available."""
  return True


def current_device():
  """Returns `0` as JAX manages devices as a single logical device."""
  return 0


def get_amp_supported_dtype():
  """Returns the data types supported by AMP (Automatic Mixed Precision)."""
  return [torch.float16, torch.bfloat16]
