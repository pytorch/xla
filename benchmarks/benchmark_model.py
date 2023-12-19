from collections import OrderedDict
import logging
import re
import torch
import torch.nn as nn
from torch._dynamo.testing import collect_results
from util import move_to_device

logger = logging.getLogger(__name__)


class ModelLoader:

  def __init__(self, args):
    self._args = args
    self.suite_name = self._args.suite_name
    self.benchmark_model_class = BenchmarkModel

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
      benchmark_model.prepare_for_experiment()

    return benchmark_model


class BenchmarkModel:

  def __init__(self, suite_name, model_name, benchmark_experiment):
    self.suite_name = suite_name
    self.model_name = model_name
    self.benchmark_experiment = benchmark_experiment

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

  def prepare_for_experiment(self):
    self.device = self.benchmark_experiment.get_device()
    self.module = self.module.to(self.device)
    self.example_inputs = move_to_device(self.example_inputs, self.device)

    if self.benchmark_experiment.test == "eval":
      self._prepare_for_eval()
    elif self.benchmark_experiment.test == "train":
      self._prepare_for_train()
    else:
      raise NotImplementedError

    if self.benchmark_experiment.dynamo:
      self.model_iter_fn = torch.compile(
          self.model_iter_fn, backend=self.benchmark_experiment.dynamo)

  def pick_grad(self):
    if self.benchmark_experiment.test == "eval":
      return torch.no_grad()
    elif self.benchmark_experiment.test == "train":
      return torch.enable_grad()

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
