import logging
import os
import re
import tempfile
import subprocess

logging.basicConfig()
logger = logging.getLogger(__name__)


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
  flags = _set_missing_flags(
      flags, (('xla_gpu_simplify_all_fp_conversions', 'false'),))
  flags = _set_missing_flags(flags,
                             (('xla_gpu_force_compilation_parallelism', '8'),))
  os.environ['XLA_FLAGS'] = ' '.join(flags)


def _set_missing_env(name, value):
  if name not in os.environ:
    os.environ[name] = value


def _setup_default_env():
  _set_missing_env('TF_CPP_MIN_LOG_LEVEL', '1')
  _set_missing_env('GRPC_VERBOSITY', 'ERROR')
  _set_missing_env('ALLOW_MULTIPLE_LIBTPU_LOAD', '1')
  _set_missing_env('TPU_ML_PLATFORM', 'PyTorch/XLA')


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


def _setup_tpu_vm_library_path() -> bool:
  """Returns true if $TPU_LIBRARY is set or can be inferred.

  We load libtpu.so in the following order of precedence:

  1. User-set $TPU_LIBRARY_PATH
  2. libtpu.so included in torch_xla/lib
  3. libtpu-nightly pip package
  """
  if 'TPU_LIBRARY_PATH' in os.environ:
    return True

  module_path = os.path.dirname(__file__)
  bundled_libtpu_path = os.path.join(module_path, 'lib/libtpu.so')
  if os.path.isfile(bundled_libtpu_path) and not os.getenv('TPU_LIBRARY_PATH'):
    logger.info('Using bundled libtpu.so (%s)', bundled_libtpu_path)
    os.environ['TPU_LIBRARY_PATH'] = bundled_libtpu_path
    return True

  try:
    import libtpu
    libtpu.configure_library_path()
    return True
  except ImportError:
    return False


# These needs to be called before the _XLAC module is loaded.
_setup_default_env()
_setup_grpc()
_setup_xla_flags()
if int(os.environ.get('PT_XLA_DEBUG', '0')):
  _fd, _tmp_fname = _setup_debug_env()

if os.environ.get('TF_CPP_MIN_LOG_LEVEL') == '0':
  logger.setLevel(logging.INFO)

import atexit
import torch
from ._patched_functions import _apply_patches
from .version import __version__

logger.info(
    'Letting libtpu.so load fail during _XLAC import. libtpu.so will be loaded '
    'from `libtpu` Python package when the ComputationClient is created.')
os.environ['TPU_LOAD_LIBRARY'] = '0'
import _XLAC
del os.environ['TPU_LOAD_LIBRARY']

_found_libtpu = _setup_tpu_vm_library_path()


def _prepare_to_exit():
  _XLAC._prepare_to_exit()
  if int(os.environ.get('PT_XLA_DEBUG', '0')):
    _summarize_fn_tracker()


def _init_xla_lazy_backend():
  _XLAC._init_xla_lazy_backend()


atexit.register(_prepare_to_exit)
_apply_patches()
_init_xla_lazy_backend()
