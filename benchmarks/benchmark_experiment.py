import logging
import os
import torch

try:
  from .util import is_xla_device_available
except ImportError:
  from util import is_xla_device_available

logger = logging.getLogger(__name__)


class ExperimentLoader:

  def __init__(self, args):
    self._args = args

  def list_experiment_configs(self):
    experiment_configs = []

    accelerators = ["cpu", "gpu", "tpu"]
    xla_options = [None, "PJRT", "XRT"]
    tests = ["eval", "train"]
    for accelerator in accelerators:
      for xla in xla_options:
        for test in tests:
          experiment_config = {"accelerator": accelerator, "xla": xla, "test": test}

          if not self.is_available(experiment_config):
            continue

          self._add_experiment_env(experiment_config)
          experiment_configs.append(experiment_config)
    return experiment_configs

  def is_available(self, experiment_config):
    if (experiment_config["xla"] and
        not is_xla_device_available(experiment_config["accelerator"].upper())):
      return False
    if (experiment_config["accelerator"] == "tpu" and
        not experiment_config["xla"]):
      return False
    if (experiment_config["accelerator"] == "gpu" and
        experiment_config["xla"] == "PJRT"):
      return False
    if (experiment_config["accelerator"] == "gpu" and
        not experiment_config["xla"] and
        not torch.cuda.is_available()):
      return False
    return True

  def _add_experiment_env(self, experiment_config):
    process_env = None
    if experiment_config["xla"] == "PJRT":
      process_env = os.environ.copy()
      process_env["PJRT_DEVICE"] = experiment_config["accelerator"].upper()
    elif experiment_config["xla"] == "XRT":
      process_env = os.environ.copy()
      if is_xla_device_available("TPU"):
        process_env["TPU_NUM_DEVICES"] = "1"
        process_env["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
      elif is_xla_device_available("GPU"):
        process_env["GPU_NUM_DEVICES"] = "1"

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
      import torch_xla.core.xla_model as xm
      device = xm.xla_device(devkind=self.accelerator.upper())
    elif self.accelerator == "cpu":
      device = torch.device("cpu")
    elif self.accelerator == "gpu":
      device = torch.device("cuda")
    else:
      raise NotImplementedError

    return device

  @property
  def filename_str(self):
    return f"{self.accelerator}-{self.xla}-{self.test}"