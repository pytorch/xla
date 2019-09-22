import warnings
warnings.warn('torch_xla_py has been restructured to torch_xla and it will be removed soon, '
              'please call torch_xla directly.')
from torch_xla.core import xla_model, xla_env_vars
from torch_xla.distributed import data_parallel, parallel_loader, xla_multiprocessing, xla_dist
from torch_xla.debug import graph_saver, metrics_saver, model_comparator
from torch_xla.utils import utils, keyd_queue
