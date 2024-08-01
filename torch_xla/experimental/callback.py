from typing import Callable
import torch
import torch_xla


def on_ready_callback(tensor, callback: Callable[[torch.Tensor], None]):

  def _callback_wrapper():
    callback(tensor)

  torch_xla._XLAC._on_ready_callback(tensor, _callback_wrapper)
