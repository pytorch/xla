import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import save_as_stablehlo, StableHLOExportOptions
import torch
import torch._export
import torchvision

import tempfile
import unittest

from . import llama_model, llama_model2


class ExportTest(unittest.TestCase):

  def test_llama_export(self):
    options = llama_model.ModelArgs()
    model = llama_model.Transformer(options)
    arg = (torch.randint(0, 1000, (1, 100)), 0)
    """
    exported = torch._export.export(model, arg)

    with tempfile.TemporaryDirectory() as tempdir:
      save_as_stablehlo(exported, arg, tempdir)
      """

    gen = llama_model.GenLoop(model).eval()
    arg = (torch.randint(0, 1000, (1, 10)),)
    arg[0].requires_grad = False
    options = StableHLOExportOptions()
    options.override_tracing_arguments = arg
    with torch.no_grad():
      exported2 = torch._export.export(gen, arg)
    with tempfile.TemporaryDirectory() as tempdir:
      save_as_stablehlo(exported2, tempdir, options)

  def test_llama_export(self):
    options = llama_model.ModelArgs()
    model = llama_model2.Transformer(options)
    arg = (torch.randint(0, 1000, (8, 100)), torch.arange(0, 100), None)
    options = StableHLOExportOptions()
    options.override_tracing_arguments = arg
    exported = torch._export.export(model, arg)
    with tempfile.TemporaryDirectory() as tempdir:
      save_as_stablehlo(exported, tempdir, options)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
