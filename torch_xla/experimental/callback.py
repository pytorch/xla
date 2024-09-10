from typing import Callable
import torch
import torch_xla
import threading


def on_ready_callback(tensor, callback: Callable[[torch.Tensor], None]):
  """Installs callback on `tensor` to be called when underlying buffer is ready.

  Note: Since `callback` will need to re-acquire the GIL since it is a Python
  callable. If the main thread is blocking on `callback` and holding the GIL,
  this will result in a deadlock.
  """

  def _callback_wrapper():
    callback(tensor)

  torch_xla._XLAC._on_ready_callback(tensor, _callback_wrapper)


def on_ready_event(tensor: torch.Tensor) -> threading.Event:
  """Return a python threading.event that will be set once underlying
  tensor buffer is ready.

  Args:
    tensor: tensor that the event will be blocked on
  """
  ready_event = threading.Event()

  def _callback_wrapper():
    ready_event.set()

  torch_xla._XLAC._on_ready_callback(tensor, _callback_wrapper)
  return ready_event
