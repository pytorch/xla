import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def extract_execution_cause(lines):
  causes = []
  for i in range(len(lines)):
    if 'Execution Cause' in lines[i].decode():
      causes.append(lines[i + 1].decode())
  return causes


def extract_python_frames(lines):
  frames = []
  current_frame = ''
  record_frame = False
  for i in range(len(lines)):
    if 'Python Frame Triggered Execution' in lines[i].decode():
      record_frame = True
    elif 'Execution Analysis: ----------------' in lines[i].decode():
      record_frame = False
      frames.append(current_frame)
      current_frame = ''
    if record_frame:
      current_frame += lines[i].decode()
  return frames


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
  xm.mark_step()
  xm.wait_device_ops()

  if index == 0:
    # All of the process will write to the same PT_XLA_DEBUG_FILE, but the
    # no need to check this on all processes.
    with open(debug_file_name, 'rb') as f:
      lines = f.readlines()
      causes = extract_execution_cause(lines)
      frames = extract_python_frames(lines)
    # only the local master process should dump the executation analysis
    assert (len(causes) == 1)
    assert ('user mark_step' in causes[0])
    # make sure that frame that spawn up process is skipped
    assert (len(frames) == 1)
    assert ('....' in frames[0])
    assert ('_internal/pjrt.py' not in frames[0])


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
