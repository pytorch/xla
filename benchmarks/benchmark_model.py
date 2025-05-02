import argparse
from collections import OrderedDict
import contextlib
import logging
import re
import torch
import torch.nn as nn
from torch._dynamo.testing import collect_results
from torch.utils import _pytree as pytree
from util import cast_to_dtype, move_to_device
from benchmark_experiment import BenchmarkExperiment
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


class BenchmarkModel:

  def __init__(self, suite_name: str, model_name: str,
               benchmark_experiment: BenchmarkExperiment):
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

  def skip_verifier(self):
    """Returns whether the verifier should be skipped for this model.
    """
    return False

  def tolerance(self):
    """Tolerance to be used by the verifier.
    """
    # Default value taken from: PyTorch
    # Source: benchmarks/dynamo/torchbench.py
    return 1e-4

  def use_cosine_similarity(self):
    """Whether the verifier should use cosine similarity for checking the result's accuracy.
    """
    # Default value taken from: PyTorch
    # Source: benchmarks/dynamo/torchbench.py
    return False

  def conversion_dtype(self):
    return None

  def prepare_for_experiment(
      self,
      dynamo_compilation_opts: Dict[str, str],
      force_dtype: Optional[torch.dtype] = None,
  ):
    self.device = self.benchmark_experiment.get_device()

    if force_dtype is None:
      self.dtype = self.conversion_dtype()
    else:
      self.dtype = force_dtype

    if self.dtype is not None:
      self.module = self.module.to(self.dtype)
      self.example_inputs = cast_to_dtype(self.example_inputs, self.dtype)

    if self.benchmark_experiment.test == "eval":
      self._prepare_for_eval()
    elif self.benchmark_experiment.test == "train":
      self._prepare_for_train()
    else:
      raise NotImplementedError

    keep_model_data_on_cuda = self.benchmark_experiment.keep_model_data_on_cuda
    if self.benchmark_experiment.torch_xla2:
      import torch_xla2.export
      import torch_xla2
      import jax
      import jax.numpy as jnp
      device = jax.devices()[0]
      if self.benchmark_experiment.torch_xla2 == 'torch_export':
        # for torch_xla2, we export model to FX graph and move weights to JAX device
        exported = torch.export.export(self.module, self.example_inputs)
        weights, jax_func = torch_xla2.export.exported_program_to_jax(exported)
      elif self.benchmark_experiment.torch_xla2 == 'extract_jax':
        weights, jax_func = torch_xla2.extract_jax(self.module)
      else:
        raise ValueError("torch_xla2 option unavailable")
      weights = pytree.tree_map_only(jnp.ndarray,
                                     lambda x: jax.device_put(x, device),
                                     weights)
      jax_func = jax.jit(jax_func)
      self.module = lambda *x: jax_func(weights, x)
      self.example_inputs = move_to_device(
          self.example_inputs, device, torch_xla2=True)
    elif not keep_model_data_on_cuda:
      self.module = self.module.to(self.device)
      self.example_inputs = move_to_device(
          self.example_inputs, self.device, torch_xla2=False)

    if self.benchmark_experiment.dynamo:
      compilation_opts = dynamo_compilation_opts.copy()
      compilation_opts['backend'] = self.benchmark_experiment.dynamo

      logger.info(f"Running torch.compile with opts {compilation_opts}")
      self.model_iter_fn = torch.compile(self.model_iter_fn, **compilation_opts)

    if keep_model_data_on_cuda:

      def assert_func(t):
        assert t.device.type.lower(
        ) == 'cuda', 'When keep_model_data_on_cuda is set, the input data should remain on the CUDA device.'

      pytree.tree_map_only(torch.Tensor, assert_func, self.example_inputs)

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

  def train(self, inputs: Sequence[Any], collect_full_output: bool = False):
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

  def eval(self, inputs: Sequence[Any], collect_full_output: bool = False):
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

  def update_process_env(self, process_env: Dict[str, str]):
    pass


class ModelLoader:

  def __init__(self, args: argparse.Namespace):
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

  def is_compatible(self, dummy_benchmark_model: BenchmarkModel,
                    benchmark_experiment: BenchmarkExperiment):
    return True

  def get_benchmark_indices(self, length: int):
    start = self._args.partition_id * (length // self._args.total_partitions)
    end = ((self._args.partition_id + 1) *
           (length // self._args.total_partitions)
           if self._args.partition_id < self._args.total_partitions - 1 else
           length)
    return start, end

  def skip_model(self, model_name: str):
    return (not re.search("|".join(self._args.filter), model_name, re.I) or
            re.search("|".join(self._args.exclude), model_name, re.I))

  def load_model(
      self,
      model_config: Dict[str, Any],
      benchmark_experiment: BenchmarkExperiment,
      dummy: bool = False,
      force_dtype: Optional[torch.dtype] = None,
  ) -> BenchmarkModel:
    """Loads the model.

    Using both model and experiment configuration, this function will return an
    instance of BenchmarkModel.

    If specified, `force_dtype` will force the underlying model to be cast to
    that data type. This is useful when running the verifier, where we force
    float64 data-type for checking the accuracy.
    """
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
          dynamo_compilation_opts=self._dynamo_compile_opts,
          force_dtype=force_dtype,
      )

    return benchmark_model
