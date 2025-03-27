import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest
import torch.nn as nn
from torch_xla.debug.frame_parser_util import process_frames


class FrameParserUtilTest(unittest.TestCase):

  def test_process_frames(self):
    dev = xm.xla_device()
    x1 = torch.rand((3, 3)).to(dev)
    x2 = torch.rand((3, 8)).to(dev)
    y1 = torch.einsum('bs,st->bt', x1, x2)
    y1 = y1 + x2
    xm.mark_step()
    # met.metric_data('CompileTime')[:1]
    y1 = y1.view(3, 1, 2, 4)
    unfold = nn.Unfold(kernel_size=(2, 3))
    y2 = unfold(y1)
    y4 = y2 * 2
    # met.metric_data('CompileTime')[:1]
    debug_file = torch_xla._tmp_fname
    assert (len(process_frames(debug_file)) > 0)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)

