from __future__ import print_function

import os
import threading
import torch_xla

_STEP_METRICS_FILE = os.environ.get('XLA_METRICS_FILE', None)
_STEP_METRICS_FILE_LOCK = threading.Lock()


def save_metrics(metrics_file=None):
  if metrics_file is None:
    metrics_file = _STEP_METRICS_FILE
  if metrics_file is not None:
    metrics_data = torch_xla._XLAC._xla_metrics_report()
    if metrics_file == 'STDERR':
      print(metrics_data, file=sys.stderr)
    elif metrics_file == 'STDOUT':
      print(metrics_data)
    else:
      with _STEP_METRICS_FILE_LOCK:
        with open(metrics_file, 'a') as fd:
          fd.write(metrics_data)
