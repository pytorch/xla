import logging
import os
import re
import tempfile

from ._internal import tpu

logging.basicConfig()
logger = logging.getLogger(__name__)


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


def _setup_libtpu_flags():
  flags = os.environ.get('LIBTPU_INIT_ARGS', '').split(' ')
  # This flag will rerun the latency hidding scheduler if the default
  # shared memory limit 95% leads to OOM. Each rerun will choose a value
  # 0.9x of the previous run, and the number of rerun is set to 1 now.
  # Shared memory limit refers to --xla_tpu_scheduler_percent_shared_memory_limit.
  # Lower shared memory limit means less communiation and computation overlapping,
  # and thus worse performance.
  flags = _set_missing_flags(flags,
                             (('xla_latency_hiding_scheduler_rerun', '1'),))

  if tpu.version() == 5:
    default_v5_flags = {
        # Enable async collectives
        'xla_enable_async_all_gather': 'true',
        'xla_enable_async_collective_permute': 'true',
    }
    flags = _set_missing_flags(flags, default_v5_flags.items())

  os.environ['LIBTPU_INIT_ARGS'] = ' '.join(flags)


def _setup_default_env():
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')

  if tpu.num_available_chips() > 0:
    _setup_libtpu_flags()

    os.environ.setdefault('ALLOW_MULTIPLE_LIBTPU_LOAD', '1')
    os.environ.setdefault('TPU_ML_PLATFORM', 'PyTorch/XLA')

    if tpu.version() == 4:
      os.environ.setdefault('TPU_MEGACORE', 'megacore_dense')


_fd, _tmp_fname = -1, ''


def _setup_debug_env():
  fd, tmp_fname = tempfile.mkstemp('.ptxla', text=True)
  os.environ.setdefault('XLA_FNTRACKER_FILE', tmp_fname)
  return fd, tmp_fname


def _summarize_fn_tracker():
  if not _tmp_fname:
    return
  from .debug.frame_parser_util import process_frames
  process_frames(_tmp_fname)
  os.close(_fd)
  os.remove(_tmp_fname)


def _aws_ec2_inf_trn_init():
  try:
    from torch_neuronx import xla
  except ImportError:
    return
  else:
    xla.init()


def _setup_tpu_vm_library_path() -> bool:
  """Returns true if $TPU_LIBRARY_PATH is set or can be inferred.

  We load libtpu.so in the following order of precedence:

  1. User-set $TPU_LIBRARY_PATH
  2. libtpu.so included in torch_xla/lib
  3. libtpu-nightly pip package

  Sets $PTXLA_TPU_LIBRARY_PATH if path is inferred by us to prevent conflicts
  with other frameworks. This env var will be removed in a future version.
  """
  if 'TPU_LIBRARY_PATH' in os.environ:
    return True

  module_path = os.path.dirname(__file__)
  bundled_libtpu_path = os.path.join(module_path, 'lib/libtpu.so')
  if os.path.isfile(bundled_libtpu_path) and not os.getenv('TPU_LIBRARY_PATH'):
    logger.info('Using bundled libtpu.so (%s)', bundled_libtpu_path)
    os.environ['PTXLA_TPU_LIBRARY_PATH'] = bundled_libtpu_path
    return True

  try:
    import libtpu
    os.environ['PTXLA_TPU_LIBRARY_PATH'] = libtpu.get_library_path()
    return True
  except ImportError:
    return False


# These needs to be called before the _XLAC module is loaded.
_setup_default_env()
_setup_xla_flags()
if int(os.environ.get('PT_XLA_DEBUG', '0')):
  _fd, _tmp_fname = _setup_debug_env()

if os.environ.get('TF_CPP_MIN_LOG_LEVEL') == '0':
  logger.setLevel(logging.INFO)

import atexit
import torch
from ._patched_functions import _apply_patches
from .version import __version__

import _XLAC

_found_libtpu = _setup_tpu_vm_library_path()

# Setup Neuron library for AWS EC2 inf/trn instances.
_aws_ec2_inf_trn_init()


def _prepare_to_exit():
  _XLAC._prepare_to_exit()
  if int(os.environ.get('PT_XLA_DEBUG', '0')):
    _summarize_fn_tracker()


def _init_xla_lazy_backend():
  _XLAC._init_xla_lazy_backend()


atexit.register(_prepare_to_exit)
_apply_patches()
_init_xla_lazy_backend()

# This is to temporarily disable the automtic dynamic shape in PyTorch Dynamo,
# which was enabled by https://github.com/pytorch/pytorch/pull/103623.
# While we come up with a long term fix, we'll set this flag to False to
# keep PyTorch/XLA CI healthy.
# TODO @wonjoo come up with a long term fix in Dynamo.
torch._dynamo.config.automatic_dynamic_shapes = False

from .stablehlo import save_as_stablehlo, save_torch_model_as_stablehlo
