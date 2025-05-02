import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm

from extract_debug_helper import (check_env_flag, extract_execution_cause,
                                  extract_python_frames)


def _mp_fn(index):
  if not check_env_flag('PT_XLA_DEBUG'):
    assert False, "This test should be run with PT_XLA_DEBUG"
  debug_file_name = os.getenv('PT_XLA_DEBUG_FILE')
  if not debug_file_name:
    assert False, "This test should be run with PT_XLA_DEBUG_FILE"
  if index == 0:
    open(debug_file_name, 'w').close()
  device = xm.xla_device()
  t1 = torch.randn(10, 10, device=device)
  t2 = t1 * 100
  torch_xla.sync()
  xm.wait_device_ops()

  if index == 0:
    # All of the process will write to the same PT_XLA_DEBUG_FILE, but the
    # no need to check this on all processes.
    with open(debug_file_name, 'rb') as f:
      lines = f.readlines()
      causes = extract_execution_cause(lines)
      frames = extract_python_frames(lines)
    # only the local master process should dump the execution analysis
    assert (len(causes) == 1)
    assert (len(frames) == 3)
    max_frame = os.getenv('PT_XLA_DEBUG_MAX_FRAME', 8)
    # Additonal lines are
    # 1. Python Frame Triggered Execution:
    # 2. ....
    # 3. empty line
    assert ('sync' in frames[0])
    assert (len(frames[0].split('\n')) == max_frame + 3)
    assert (len(frames[2].split('\n')) == max_frame + 3)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
