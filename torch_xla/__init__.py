import logging
import os
import re
import socket
import tempfile
import subprocess
import time

from .version import __version__

XRT_RUN_SERVER_PROCESS = 'torch_xla.core._xrt_run_server'
XRT_SERVER_REGEX = '^python3 -m {} [0-9]+$'.format(XRT_RUN_SERVER_PROCESS)


def _maybe_select_tpu_version():
  # Setup correct TPU runtime version for Colab and Kaggle.

  def _is_open(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if s.connect_ex((ip, int(port))) == 0:
      return True
    return False

  def _wait_for_open(version, timeout=100, interval=10, log=True):
    tpu_addr = os.environ['TPU_NAME'].split('grpc://')[1]
    deadline = time.time() + timeout

    while not _is_open(*tpu_addr.split(':')):
      if log:
        logging.warning(
            f'Waiting for TPU to be start up with version pytorch-{version}...')
      if time.time() > deadline:
        raise RuntimeError('Timed out waiting for TPU to start up')
      time.sleep(interval)

    if log:
      logging.warning(
          f'TPU has started up successfully with version pytorch-{version}')

  try:
    tpu_name = os.environ.get('TPU_NAME', '')
    if not tpu_name.startswith('grpc://'):
      # Not colab/kaggle
      return

    import cloud_tpu_client
    client = cloud_tpu_client.Client(tpu_name)
    client.configure_tpu_version(
        f'pytorch-{__version__}', restart_type='ifNeeded')
    # client.wait_for_healthy() API doesn't work as we dont have TPU API access
    _wait_for_open(__version__)
  except ImportError:
    logging.warning((
        'Not selecting corresponding TPU runtime since cloud_tpu_client is not '
        'installed. Ignore if not running on Colab/Kaggle TPU.'))
  except Exception:
    # This path is hit, when we get throttled by the verison changer
    # when we import torch_xla from xmp.spawn-ed processes.
    _wait_for_open(__version__, log=False)


def server_is_alive():
  # pgrep returns 0 when at least one running process matches the requested name.
  # Otherwise, the exit code is 1. If pgrep is not availiable in the system, it
  # will return an exit code 127.
  return subprocess.getstatusoutput(
      'pgrep -f "{}"'.format(XRT_SERVER_REGEX))[0] == 0


def _setup_grpc():
  # Setup GRPC options to correctly talk to TPU backends.
  options = [
      'grpc.keepalive_time_ms=60000',  # 1 min
      'grpc.keepalive_timeout_ms=14400000',  # 4 hrs
      'grpc.http2.max_pings_without_data=0',  # unlimited
      'grpc.http2.min_ping_interval_without_data_ms=300000',  # 5 min
  ]
  os.environ['TF_GRPC_DEFAULT_OPTIONS'] = ','.join(options)


def _set_missing_flags(flags, sets):
  for name, defval in sets:
    insert = True
    for fval in flags:
      m = re.match(r'(--)?([^=]+)', fval)
      if m and m.group(2) == name:
        insert = False
        break
    if insert:
      flags.append('--{}={}'.format(name, defval))
  return flags


def _setup_xla_flags():
  flags = os.environ.get('XLA_FLAGS', '').split(' ')
  flags = _set_missing_flags(flags, (('xla_cpu_enable_fast_math', 'false'),))
  os.environ['XLA_FLAGS'] = ' '.join(flags)


def _set_missing_env(name, value):
  if name not in os.environ:
    os.environ[name] = value


def _setup_default_env():
  _set_missing_env('TF_CPP_MIN_LOG_LEVEL', '1')
  _set_missing_env('TPU_HOST_BOUNDS', '1,1,1')
  _set_missing_env('GRPC_VERBOSITY', 'ERROR')
  _set_missing_env('ALLOW_MULTIPLE_LIBTPU_LOAD', '1')
  if server_is_alive():
    _set_missing_env('XRT_START_LOCAL_SERVER', '0')


_fd, _tmp_fname = -1, ''


def _setup_debug_env():
  fd, tmp_fname = tempfile.mkstemp('.ptxla', text=True)
  _set_missing_env('XLA_FNTRACKER_FILE', tmp_fname)
  return fd, tmp_fname


def _summarize_fn_tracker():
  if not _tmp_fname:
    return
  from .debug.frame_parser_util import process_frames
  process_frames(_tmp_fname)
  os.close(_fd)
  os.remove(_tmp_fname)


def _tpu_vm_init():
  try:
    import libtpu
    libtpu.configure_library_path()
  except ImportError:
    return


# These needs to be called before the _XLAC module is loaded.
_maybe_select_tpu_version()
_setup_default_env()
_setup_grpc()
_setup_xla_flags()
if int(os.environ.get('PT_XLA_DEBUG', '0')):
  _fd, _tmp_fname = _setup_debug_env()

# Setup libtpu library for TPU VM.
_tpu_vm_init()

import atexit
import torch
from ._patched_functions import _apply_patches
from .version import __version__
import _XLAC


def _prepare_to_exit():
  _XLAC._prepare_to_exit()
  if int(os.environ.get('PT_XLA_DEBUG', '0')):
    _summarize_fn_tracker()


atexit.register(_prepare_to_exit)
_apply_patches()
