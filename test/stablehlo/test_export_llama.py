import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental.stablehlo_saved_model import save_as_stablehlo
import torch
import torch._export
import torchvision

import tempfile
import unittest

from . import llama_model


class ExportTest(unittest.TestCase):

  def test_llama_export(self):
    options = llama_model.ModelArgs()
    model = llama_model.Transformer(options)
    arg = (torch.randint(0, 1000, (1, 100)), 0)
    exported = torch._export.export(model, arg)

    with tempfile.TemporaryDirectory() as tempdir:
      save_as_stablehlo(exported, arg, tempdir)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
