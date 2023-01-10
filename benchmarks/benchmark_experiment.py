import logging
import os
import torch
import torch_xla.core.xla_model as xm

try:
  from .utils import is_tpu_available
except ImportError:
  from utils import is_tpu_available

logger = logging.getLogger(__name__)


class ExperimentLoader:

  def __init__(self, args):
    self._args = args

  def list_experiment_configs(self):
    experiment_configs = []

    accelerators = ["cpu"]

    if torch.cuda.is_available():
      accelerators.append("gpu")

    if is_tpu_available():
      accelerators.append("tpu")

    xla_options = [None, "PJRT"]
    tests = ["eval", "train"]
    for accelerator in accelerators:
      for xla in xla_options:
        for test in tests:
          experiment_config = {"accelerator": accelerator, "xla": xla, "test": test}
          if self._is_valid(experiment_config):
            self._add_experiment_env(experiment_config)
            experiment_configs.append(experiment_config)
    return experiment_configs

  def _is_valid(self, experiment_config):
    if experiment_config["accelerator"] == "tpu" and not experiment_config["xla"]:
      return False
    return True

  def _add_experiment_env(self, experiment_config):
    process_env = None
    if experiment_config["xla"] == "PJRT":
      process_env = os.environ.copy()
      process_env["PJRT_DEVICE"] = experiment_config["accelerator"].upper()
    experiment_config["process_env"] = process_env

  def load_experiment(self, experiment_config):
    accelerator = experiment_config.get("accelerator", "cpu")
    xla = experiment_config.get("xla", None)
    test = experiment_config.get("test", "eval")
    benchmark_experiment = BenchmarkExperiment(accelerator=accelerator, xla=xla, test=test)

    return benchmark_experiment


class BenchmarkExperiment:

  def __init__(self, accelerator, xla, test):
    self.accelerator = accelerator
    self.xla = xla
    self.test = test

  def get_device(self):
    if self.xla:
      device = xm.xla_device(devkind=self.accelerator.upper())
    elif self.accelerator == "cpu":
      device = torch.device("cpu")
    elif self.accelerator == "gpu":
      device = torch.device("cuda")
    else:
      raise NotImplementedError

    return device