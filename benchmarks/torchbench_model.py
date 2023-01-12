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

    self.torchbench_dir = self.add_torchbench_dir()

  def add_torchbench_dir():
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
    models = models[start: end]
    for model_path in models:
      model_name = os.path.basename(model_path)

      if self.skip_model(model_name):
        continue

      model_configs.append({"model_name": model_name})

    return model_configs

  def is_compatible(self, model_config, experiment_config):
    return True

  def load_model(self, model_config, benchmark_experiment):
    suite_name = self.suite_name
    model_name = model_config["model_name"]
    batch_size = self._args.batch_size
    benchmark_model = TorchBenchModel(
        suite_name=suite_name,
        model_name=model_name,
        batch_size=batch_size,
        benchmark_experiment=benchmark_experiment,
    )

    benchmark_model.set_up()
    benchmark_model.prepare_for_experiment()

    return benchmark_model


class TorchBenchModel(BenchmarkModel):

  def __init__(self, suite_name, model_name, batch_size, benchmark_experiment):
    super().__init__(suite_name, model_name, batch_size, benchmark_experiment)

  def set_up(self):
    """Set up module, actual batch_size, example_inputs, and optimizer_class

    This is model suite specific.
    """
    self.optimizer_class = torch.optim.Adam

    
