from typing import Callable
import torch
import torch_xla


def on_ready_callback(tensor, callback: Callable[[torch.Tensor], None]):
  """Installs callback on `tensor` to be called when underlying buffer is ready.

  Note: Since `callback` will need to re-acquire the GIL since it is a Python
  callable. If the main thread is blocking on `callback` and holding the GIL,
  this will result in a deadlock.
  """

  def _callback_wrapper():
    callback(tensor)

  torch_xla._XLAC._on_ready_callback(tensor, _callback_wrapper)
