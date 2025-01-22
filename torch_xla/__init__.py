import logging
import os
import re
import sys
import tempfile
import warnings

import torch

if not torch.cuda.is_available():
  # Load _XLAC_cuda_functions to RTLD_GLOBAL, so that it can be used by _XLAC.
  flags = sys.getdlopenflags()
  sys.setdlopenflags(flags | os.RTLD_NOW | os.RTLD_GLOBAL)

  import _XLAC_cuda_functions

  # Then, restore the original flags.
  sys.setdlopenflags(flags)

import _XLAC
from ._internal import tpu
from .version import __version__

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

  # This flag will prevent AllGather decomposition into AllReduce by the
  # compiler when async AllGather is enabled. Decomposed AllGathers are
  # persisted in-memory and shared between the forward and backward passes,
  # which can result in the entire model's parameters being in device memory.
  # However, regular AllGathers are instead rematerialized in the backward pass,
  # and when they are async this incurs little overhead but significantly
  # improves device memory usage.
  flags = _set_missing_flags(
      flags, (('xla_tpu_prefer_async_allgather_to_allreduce', 'true'),))

  # This flag enables FlashAttention HLO pass that pattern matches attention
  # and rewrites it as flash attention. This pattern matching is causing
  # issues for our standard dot product attention. Turning it off till
  # we fix the issue with pattern matching.
  flags = _set_missing_flags(flags,
                             (('xla_tpu_enable_flash_attention', 'false'),))

  if tpu.version() == 5:
    default_v5_flags = {
        # TODO(jonbolin): Tune these flags for async collective fusion - v5
        # requires continuation fusion to run async collectives.
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
    # This is used for ML Framework Telemetry.
    os.environ.setdefault('TPU_ML_PLATFORM_VERSION', __version__)
    os.environ.setdefault('ENABLE_RUNTIME_UPTIME_TELEMETRY', '1')

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
    from libneuronxla.libneuronpjrt_path import libneuronpjrt_path
  except ImportError:
    # Did not find libneuronxla
    return False

  # Need to set NEURON_LIBRARY_PATH here for proper Neuron Cache behavior
  os.environ.setdefault('NEURON_LIBRARY_PATH', libneuronpjrt_path())
  # Enable addition features and overrides
  try:
    from torch_neuronx import xla
  except ImportError:
    # Basic initializations if torch-neuronx is not available
    from ._internal import neuron
    if os.path.basename(sys.argv[0]) != 'neuron_parallel_compile':
      import libneuronxla
      libneuronxla.configure_environment()
      neuron.set_envvar_defaults()
      neuron.configure_pjrt_environment()
  else:
    xla.init()
  # Found libneuronxla
  return True


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


def _check_deprecated_env_var():
  deprecated_env_vars = ['XLA_USE_FP16', 'XLA_DOWNCAST_FP16']
  for env_var in deprecated_env_vars:
    if os.environ.get(env_var):
      warnings.warn(f"The environment variable '{env_var}' is deprecated "
                    "Please update your code to avoid using it.")


# These needs to be called before the _XLAC module is loaded.
_setup_default_env()
_setup_xla_flags()
_check_deprecated_env_var()
if int(os.environ.get('PT_XLA_DEBUG', '0')):
  _fd, _tmp_fname = _setup_debug_env()

if os.environ.get('TF_CPP_MIN_LOG_LEVEL') == '0':
  logger.setLevel(logging.INFO)

import atexit
from ._patched_functions import _apply_patches

_found_libtpu = _setup_tpu_vm_library_path()

# Setup Neuron library for AWS EC2 inf/trn instances.
_found_libneuronxla = _aws_ec2_inf_trn_init()


def _prepare_to_exit():
  try:
    _XLAC._prepare_to_exit()
    if int(os.environ.get('PT_XLA_DEBUG', '0')):
      _summarize_fn_tracker()
  except Exception as e:
    logging.error(
        "Caught an exception when exiting the process. Exception: ", exc_info=e)
    # Due to https://bugs.python.org/issue27035, simply raising an exception in the atexit callback does not set the exit code correctly. That is why we need to set the exit code explicitly.
    # Using `exit(1)` does not set a correct exit code because it is useful for the interactive interpreter shell and should not be used in programs and it works by raising an exception. (https://docs.python.org/3/library/constants.html#exit)
    # sys.exit(1) does not set a correct exit code because it also raises an exception.
    os._exit(1)


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
# Unspecialized float is not friendly to XLA, set flag to False until XLA
# can better compile F64 scalar tensors
torch._dynamo.config.specialize_float = True

# Activate view-replay on AOTAutograd.
# See: https://github.com/pytorch/pytorch/pull/124488
import torch._functorch.config

torch._functorch.config.view_replay_for_aliased_outputs = True

import importlib.metadata
import warnings

try:
  # TensorFlow TPU distribution has the same package name as GPU, but not CPU
  dist = importlib.metadata.distribution('tensorflow')
  warnings.warn(
      "`tensorflow` can conflict with `torch-xla`. Prefer `tensorflow-cpu` when"
      " using PyTorch/XLA. To silence this warning, `pip uninstall -y "
      "tensorflow && pip install tensorflow-cpu`. If you are in a notebook "
      "environment such as Colab or Kaggle, restart your notebook runtime "
      "afterwards.")
except importlib.metadata.PackageNotFoundError:
  pass

from .stablehlo import save_as_stablehlo, save_torch_model_as_stablehlo

from .experimental import plugins
from ._internal import neuron, xpu  # Additional built-in plugins

if os.getenv('XLA_REGISTER_INSTALLED_PLUGINS',
             '0' if _XLAC._has_cuda_support() else '1') == '1':
  plugins.use_dynamic_plugins()
  plugins.register_installed_plugins()

if os.getenv('XLA_USE_EAGER_DEBUG_MODE', '0') == '1':
  from .experimental import eager_mode
  eager_mode(True)

from .torch_xla import *

# register all custom kenels and decomp by default
from ._internal import custom_kernel, decomp_registration, c10d_registration

# select default PJRT_DEVICE before any execution
runtime._maybe_select_default_device()
