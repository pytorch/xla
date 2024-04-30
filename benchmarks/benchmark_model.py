from collections import OrderedDict
import contextlib
import logging
import re
import torch
import torch.nn as nn
from torch._dynamo.testing import collect_results
import torch_xla.utils.utils as xu
from util import cast_to_dtype, move_to_device

logger = logging.getLogger(__name__)


class ModelLoader:

  def __init__(self, args):
    self._args = args
    self.suite_name = self._args.suite_name
    self.benchmark_model_class = BenchmarkModel
    self._dynamo_compile_opts = dict()
    if self._args.filter_by_single_graph:
      self._dynamo_compile_opts['fullgraph'] = True

  def list_model_configs(self):
    model_configs = [
        {
            "model_name": "dummy"
        },
    ]

    return model_configs

  def is_compatible(self, dummy_benchmark_model, benchmark_experiment):
    return True

  def get_benchmark_indices(self, length):
    start = self._args.partition_id * (length // self._args.total_partitions)
    end = ((self._args.partition_id + 1) *
           (length // self._args.total_partitions)
           if self._args.partition_id < self._args.total_partitions - 1 else
           length)
    return start, end

  def skip_model(self, model_name):
    return (not re.search("|".join(self._args.filter), model_name, re.I) or
            re.search("|".join(self._args.exclude), model_name, re.I))

  def load_model(self, model_config, benchmark_experiment, dummy=False):
    suite_name = self.suite_name
    model_name = model_config["model_name"]
    benchmark_model = self.benchmark_model_class(
        suite_name=suite_name,
        model_name=model_name,
        benchmark_experiment=benchmark_experiment,
    )

    if not dummy:
      benchmark_model.set_up()
      benchmark_model.prepare_for_experiment(
          dynamo_compilation_opts=self._dynamo_compile_opts)

    return benchmark_model


class BenchmarkModel:

  def __init__(self, suite_name, model_name, benchmark_experiment):
    self.suite_name = suite_name
    self.model_name = model_name
    self.benchmark_experiment = benchmark_experiment
    self.autocast = contextlib.nullcontext
    self.autocast_kwargs = {}

  def set_up(self):
    """Set up module, actual batch_size, example_inputs, and optimizer_class

    This is model suite specific.
    """
    if self.model_name != "dummy":
      raise NotImplementedError

    self.module = nn.Sequential(
        nn.Linear(32, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 32),
        nn.Softmax(dim=1),
    )

    self.benchmark_experiment.batch_size = 16
    self.example_inputs = (torch.rand(self.benchmark_experiment.batch_size,
                                      32),)
    self.optimizer_class = torch.optim.Adam

  def _prepare_for_eval(self):
    self.module.eval()
    self.model_iter_fn = self.eval

  def _prepare_for_train(self):
    self.module.train()
    self.model_iter_fn = self.train
    if not hasattr(self, "optimizer"):
      # For some special models, self.set_up() may have initialized an
      # optimizer to use. So only initialize it when there is none existing.
      self.optimizer = self.optimizer_class(self.module.parameters(), lr=0.01)

  def conversion_dtype(self):
    return None

  def prepare_for_experiment(self, dynamo_compilation_opts):
    self.device = self.benchmark_experiment.get_device()
    self.dtype = self.conversion_dtype()

    if self.dtype is not None:
      self.module = self.module.to(self.dtype)
      self.example_inputs = cast_to_dtype(self.example_inputs, self.dtype)

    xla_fallback_cuda_enabled = xu.getenv_as('XLA_FALLBACK_CUDA', bool, defval=False)
    if not (self.benchmark_experiment.xla and self.benchmark_experiment.accelerator == "cuda" and xla_fallback_cuda_enabled):
      self.module = self.module.to(self.device)
      self.example_inputs = move_to_device(self.example_inputs, self.device)

    if self.benchmark_experiment.test == "eval":
      self._prepare_for_eval()
    elif self.benchmark_experiment.test == "train":
      self._prepare_for_train()
    else:
      raise NotImplementedError

    if self.benchmark_experiment.dynamo:
      compilation_opts = dynamo_compilation_opts.copy()
      compilation_opts['backend'] = self.benchmark_experiment.dynamo

      logger.info(f"Running torch.compile with opts {compilation_opts}")
      self.model_iter_fn = torch.compile(self.model_iter_fn, **compilation_opts)

  def pick_grad(self):
    if self.benchmark_experiment.test == "eval":
      return torch.no_grad()
    elif self.benchmark_experiment.test == "train":
      return torch.enable_grad()
    raise NotImplementedError

  def _optimizer_zero_grad(self):
    if self.optimizer is not None:
      self.optimizer.zero_grad(True)
    else:
      self.module.zero_grad(True)

  def _optimizer_step(self):
    if self.optimizer is not None:
      self.optimizer.step()

  def compute_loss(self, pred):
    raise NotImplementedError

  def train(self, inputs, collect_full_output=False):
    self._optimizer_zero_grad()
    with self.autocast(**self.autocast_kwargs):
      pred = self.module(*inputs)
      loss = self.compute_loss(pred)
    loss.backward()
    self._optimizer_step()
    if collect_full_output:
      return collect_results(self.module, pred, loss, inputs)
    # return loss.detach()
    # TODO: dynamo inductor would fail if .detach() is used
    return None

  def eval(self, inputs, collect_full_output=False):
    with self.autocast(**self.autocast_kwargs):
      pred = self.module(*inputs)
    return pred

  @property
  def filename_str(self):
    return "-".join(self.to_dict().values())

  def to_dict(self):
    d = OrderedDict()
    d["suite_name"] = self.suite_name
    d["model_name"] = self.model_name
    return d

  @property
  def default_precision_flag(self):
    return None

  def update_process_env(self, process_env):
    pass
