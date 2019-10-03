from __future__ import print_function

import torch_xla


def metrics_report():
  return torch_xla._XLAC._xla_metrics_report()
