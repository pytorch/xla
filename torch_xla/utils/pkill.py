import argparse
import os
import signal

PSUTIL_NOT_INSTALLED = False
try:
  import psutil
except ImportError:
  PSUTIL_NOT_INSTALLED = True


def _strip_lr_quotes(cmdline, quote='"'):
  # For example:
  # Input: ['bash', '-c', '"/tmp/18129-remote/dist_training_ptxla_1.sh"']
  # Output: ['bash', '-c', '/tmp/18129-remote/dist_training_ptxla_1.sh']
  return [cmd.lstrip(quote).rstrip(quote) for cmd in cmdline]

def _find_script_process(script_name):
  proc = None
  for p in psutil.process_iter():
    # Also check 'bash' is in the cmdline since pkill.py also has --script_name.
    cmdline = _strip_lr_quotes(p.cmdline())
    if 'bash' in cmdline and script_name in cmdline:
      # xla_dist.py calls script as: ['bash', '-c', '"/tmp/18129-remote/dist_training_ptxla_1.sh"']
      proc = p
  return proc

def _kill_script(script_name):
  proc = _find_script_process(script_name)
  if not proc:
    print(('WARNING: process for {} not found.'
           ' Processes not cleaned up.').format(script_name))
    return
  os.killpg(proc.pid, signal.SIGTERM)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Script process group janitor.',
    epilog=('Usage exampe: python -m'
            ' torch_xla.utils.pkill --script_name /tmp/runner.sh'))
  parser.add_argument(
    '--script_name', type=str, required=True,
    help='Name of the script to kill process and all children.')
  args = parser.parse_args()

  if PSUTIL_NOT_INSTALLED:
    print(('WARNING: `psutil` package is not installed. This means we wont be'
           ' able to cleanup processes and processes may be leaked. Please'
           ' install `psutil` by running `conda install psutil`.'))
  else:
    _kill_script(args.script_name)