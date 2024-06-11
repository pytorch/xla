import torch_xla


def eager_mode(enable: bool):
  """Configure torch_xla's default executation mode.
  Under eager mode only functions that was `torch_xla.compile`d will be
  traced and compiled. Other torch ops will be executed eagerly.
  """
  torch_xla._XLAC._set_use_eager_mode(enable)
