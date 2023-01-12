import logging
import re
import torch
import torch.nn as nn
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs
import types

try:
  from .util import move_to_device
except ImportError:
  from util import move_to_device


logger = logging.getLogger(__name__)


class ModelLoader:

  def __init__(self, args):
    self._args = args
    self.suite_name = self._args.suite_name

  def list_model_configs(self):
    model_configs = [
        {"model_name": "dummy"},
    ]

    return model_configs

  def is_compatible(self, model_config, experiment_config):
    return True

  def get_benchmark_indices(self, length):
    start = self._args.partition_id * (length // self._args.total_partitions)
    end = (
        (self._args.partition_id + 1) * (length // self._args.total_partitions)
        if self._args.partition_id < self._args.total_partitions - 1
        else length
    )
    return start, end

  def skip_model(self, model_name):
    return (not re.search("|".join(self._args.filter), model_name, re.I) or
            re.search("|".join(self._args.exclude), model_name, re.I))

  def load_model(self, model_config, benchmark_experiment):
    suite_name = self.suite_name
    model_name = model_config["model_name"]
    optimizer_name = model_config.get("optimizer_name", "Adam")
    batch_size = self._args.batch_size
    benchmark_model = BenchmarkModel(
        suite_name=suite_name,
        model_name=model_name,
        optimizer_name=optimizer_name,
        batch_size=batch_size,
        benchmark_experiment=benchmark_experiment,
    )

    benchmark_model.set_up()
    benchmark_model.prepare_for_experiment()

    return benchmark_model


class BenchmarkModel:

  def __init__(self, suite_name, model_name, optimizer_name, batch_size,
               benchmark_experiment):
    self.suite_name = suite_name
    self.model_name = model_name
    self.optimizer_name = optimizer_name
    self.batch_size = batch_size
    self.benchmark_experiment = benchmark_experiment

  def set_up(self):
    """Set up module, actual batch_size, example_inputs, and optimizer_class

    This is model suite specific.
    """
    if self.model_name != "dummy":
      raise NotImplementedError

    self.module = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
        nn.Softmax(dim=1),
    )

    self.batch_size = 10
    self.example_inputs = (torch.rand(self.batch_size, 3),)
    self.optimizer_class = torch.optim.Adam

  def prepare_for_experiment(self):
    self.device = self.benchmark_experiment.get_device()
    self.module = self.module.to(self.device)
    self.example_inputs = move_to_device(self.example_inputs, self.device)

    if self.benchmark_experiment.test == "eval":
      self.module.eval()
      self.model_iter_fn = self.eval
      self.optimizer = None
    elif self.benchmark_experiment.test == "train":
      self.module.train()
      self.model_iter_fn = self.train
      self.optimizer = self.optimizer_class(self.module.parameters(), lr=0.01)
    else:
      raise NotImplementedError

  def optimizer_zero_grad(self):
    if self.optimizer is not None:
      self.optimizer.zero_grad(True)
    else:
      self.module.zero_grad(True)

  def optimizer_step(self):
    if self.optimizer is not None:
      self.optimizer.step()

  def compute_loss(self, pred):
    return reduce_to_scalar_loss(pred)

  def train(self, collect_outputs=True):
    cloned_inputs = clone_inputs(self.example_inputs)
    self.optimizer_zero_grad()
    pred = self.module(*cloned_inputs)
    loss = self.compute_loss(pred)
    loss.backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(self.module, pred, loss, cloned_inputs)
    return None

  def eval(self, collect_outputs=True):
    return self.module(*self.example_inputs)

  @property
  def filename_str(self):
    return f"{self.suite_name}-{self.model_name}-{self.optimizer_name}-{self.batch_size}"