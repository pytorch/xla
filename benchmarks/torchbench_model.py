import gc
import importlib
import logging
import os
from os.path import abspath, exists
import sys
import torch
import torch.nn as nn
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs
import types

try:
  from .util import move_to_device, set_cwd
  from .benchmark_model import ModelLoader, BenchmarkModel
except ImportError:
  from util import move_to_device, set_cwd
  from benchmark_model import ModelLoader, BenchmarkModel

logger = logging.getLogger(__name__)


class TorchBenchModelLoader(ModelLoader):

  def __init__(self, args):
    super().__init__(args)
    self.benchmark_model_class = TorchBenchModel
    self.torchbench_dir = self.add_torchbench_dir()

  def add_torchbench_dir(self):
    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    for torchbench_dir in (
        "./torchbenchmark",
        "./torchbench",
        "./benchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ):
      if exists(torchbench_dir):
        break

    if exists(torchbench_dir):
      torchbench_dir = abspath(torchbench_dir)
      if torchbench_dir not in sys.path:
        sys.path.append(torchbench_dir)
    else:
      raise Exception("Torch Benchmark folder not found.")

    return torchbench_dir

  def list_model_configs(self):
    model_configs = []

    from torchbenchmark import _list_model_paths

    models = _list_model_paths()

    start, end = self.get_benchmark_indices(len(models))
    models = models[start:end]
    for model_path in models:
      model_name = os.path.basename(model_path)

      if self.skip_model(model_name):
        continue

      model_configs.append({"model_name": model_name})

    return model_configs


class TorchBenchModel(BenchmarkModel):

  def __init__(self, suite_name, model_name, benchmark_experiment):
    super().__init__(suite_name, model_name, benchmark_experiment)

  def set_up(self):
    """Set up module, actual batch_size, example_inputs, and optimizer_class

    This is model suite specific.
    """
    self.optimizer_class = torch.optim.Adam

    try:
      module = importlib.import_module(
          f"torchbenchmark.models.{self.model_name}")
    except ModuleNotFoundError:
      module = importlib.import_module(
          f"torchbenchmark.models.fb.{self.model_name}")
    benchmark_cls = getattr(module, "Model", None)

    cant_change_batch_size = (not getattr(benchmark_cls,
                                          "ALLOW_CUSTOMIZE_BSIZE", True))
    if cant_change_batch_size:
      self.benchmark_experiment.batch_size = None

    # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
    # torch.backends.__allow_nonbracketed_mutation_flag = True

    if self.benchmark_experiment.accelerator == "cpu":
      device = "cpu"
    elif self.benchmark_experiment.accelerator == "gpu" and not self.benchmark_experiment.xla:
      device = "cuda"
    else:
      device = str(self.benchmark_experiment.get_device())

    benchmark = benchmark_cls(
        test=self.benchmark_experiment.test,
        device=device,
        jit=False,
        batch_size=self.benchmark_experiment.batch_size,
    )

    self.module, self.example_inputs = benchmark.get_module()

    self.benchmark_experiment.batch_size = benchmark.batch_size

    # Torchbench has quite different setup for yolov3, so directly passing
    # the right example_inputs
    if self.model_name == "yolov3":
      self.example_inputs = (torch.rand(self.benchmark_experiment.batch_size, 3,
                                        384, 512),)

    del benchmark
    gc.collect()

  def compute_loss(self, pred):
    """Reduce the output of a model to get scalar loss"""
    if isinstance(pred, torch.Tensor):
      # Mean does not work on integer tensors
      return pred.sum() / pred.numel()
    elif isinstance(pred, (list, tuple)):
      return sum([reduce_to_scalar_loss(x) for x in pred]) / len(pred)
    elif type(pred).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
    ):
      return reduce_to_scalar_loss(pred.logits)
    elif type(pred).__name__ == "SquashedNormal":
      return pred.mean.sum()
    elif isinstance(pred, dict):
      return sum([reduce_to_scalar_loss(value) for value in pred.values()
                 ]) / len(pred.keys())
    raise NotImplementedError("Don't know how to reduce", type(pred))
