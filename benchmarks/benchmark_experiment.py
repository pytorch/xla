from collections import OrderedDict
import logging
import os
import torch
import torch._dynamo as dynamo

try:
  from .util import is_xla_device_available, get_accelerator_model
except ImportError:
  from util import is_xla_device_available, get_accelerator_model

try:
  import torch_xla.core.xla_model as xm
except ImportError:
  # ignore the error if torch_xla is not installed
  pass

logger = logging.getLogger(__name__)


class ExperimentLoader:

  def __init__(self, args):
    self._args = args
    self.experiment_name = self._args.experiment_name

  def expand_config_choices(self, config_choices):
    configs = [{}]

    for key, choices in config_choices.items():
      tmp_configs = []
      for config in configs:
        for choice in choices:
          tmp_config = config.copy()
          tmp_config[key] = choice
          tmp_configs.append(tmp_config)
      configs = tmp_configs

    return configs

  def list_experiment_configs(self):
    if self.experiment_name == "run_all":
      config_choices = {
          "accelerator": ["cpu", "cuda", "tpu"],
          "xla": [None, "PJRT", "XRT"],
          "xla_flags": [None],
          "dynamo": [None, "inductor", "openxla_eval", "openxla"],
          "test": ["eval", "train"],
      }

      if self._args.accelerator:
        config_choices["accelerator"] = list(set(self._args.accelerator))
      if self._args.xla:
        config_choices["xla"] = [
            x if x != "None" else None for x in list(set(self._args.xla))
        ]
      if self._args.dynamo:
        config_choices["dynamo"] = [
            x if x != "None" else None for x in list(set(self._args.dynamo))
        ]
      if self._args.test:
        config_choices["test"] = list(set(self._args.test))
      if self._args.xla_flags:
        config_choices["xla_flags"] = [
            x if x != "None" else None for x in list(set(self._args.xla_flags))
        ]

    else:
      raise NotImplementedError

    experiment_configs = []
    for experiment_config in self.expand_config_choices(config_choices):
      if not self.is_available(experiment_config):
        continue

      self._add_experiment_env(experiment_config)
      experiment_configs.append(experiment_config)
    return experiment_configs

  def is_available(self, experiment_config):
    if experiment_config["dynamo"] and experiment_config[
        "dynamo"] not in dynamo.list_backends(exclude_tags=()):
      return False
    if experiment_config["dynamo"] == "inductor" and not (
        experiment_config["accelerator"] == "cuda" and
        not experiment_config["xla"]):
      return False
    if experiment_config["dynamo"] == "openxla_eval" and not (
        experiment_config["xla"] and experiment_config["test"] == "eval"):
      return False
    if experiment_config["dynamo"] == "openxla" and not (
        experiment_config["xla"] and experiment_config["test"] == "train"):
      return False
    if (experiment_config["xla"] and
        not is_xla_device_available(experiment_config["accelerator"].upper())):
      return False
    if (experiment_config["accelerator"] == "tpu" and
        not experiment_config["xla"]):
      return False
    if (experiment_config["accelerator"] == "cuda" and
        not experiment_config["xla"] and not torch.cuda.is_available()):
      return False
    return True

  def _add_experiment_env(self, experiment_config):
    process_env = None

    if experiment_config["xla"]:
      # remove env vars that would interfere with subprocess settings
      os.environ.pop("PJRT_DEVICE", None)
      os.environ.pop("XRT_TPU_CONFIG", None)
      os.environ.pop("XLA_FLAGS", None)

    process_env = os.environ.copy()
    if experiment_config["xla"] == "PJRT":
      process_env["PJRT_DEVICE"] = experiment_config["accelerator"].upper()
    elif experiment_config["xla"] == "XRT":
      if is_xla_device_available("TPU"):
        process_env["TPU_NUM_DEVICES"] = "1"
        process_env["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
      elif is_xla_device_available("CUDA"):
        process_env["GPU_NUM_DEVICES"] = "1"
    elif not experiment_config["xla"] and is_xla_device_available(
        experiment_config["accelerator"].upper()):
      # In non-xla CPU training experiments, an env var is still needed if an
      # xla device exists, or there will be "Missing XLA configuration" error.
      process_env["PJRT_DEVICE"] = experiment_config["accelerator"].upper()

    if experiment_config["xla_flags"]:
      process_env["XLA_FLAGS"] = experiment_config["xla_flags"]

    experiment_config["process_env"] = process_env

  def load_experiment(self, experiment_config, dummy=False):
    experiment_name = self.experiment_name
    accelerator = experiment_config["accelerator"]
    xla = experiment_config["xla"]
    xla_flags = experiment_config["xla_flags"]
    dynamo = experiment_config["dynamo"]
    test = experiment_config["test"]
    batch_size = experiment_config.get("batch_size", self._args.batch_size)
    benchmark_experiment = BenchmarkExperiment(
        experiment_name=experiment_name,
        accelerator=accelerator,
        xla=xla,
        xla_flags=xla_flags,
        dynamo=dynamo,
        test=test,
        batch_size=batch_size)

    return benchmark_experiment


class BenchmarkExperiment:

  def __init__(self, experiment_name, accelerator, xla, xla_flags, dynamo, test,
               batch_size):
    self.experiment_name = experiment_name
    self.accelerator = accelerator
    self.xla = xla
    self.xla_flags = xla_flags
    self.dynamo = dynamo
    self.test = test
    self.batch_size = batch_size
    self.accelerator_model = get_accelerator_model(self.accelerator)

  def get_device(self):
    if self.xla:
      device = xm.xla_device(devkind=self.accelerator.upper())
    elif self.accelerator == "cpu":
      device = torch.device("cpu")
    elif self.accelerator == "cuda":
      device = torch.device("cuda")
    else:
      raise NotImplementedError

    return device

  @property
  def filename_str(self):
    return "-".join(self.to_dict().values())

  def to_dict(self):
    d = OrderedDict()
    d["experiment_name"] = self.experiment_name
    d["accelerator"] = self.accelerator
    d["accelerator_model"] = self.accelerator_model
    d["xla"] = self.xla
    d["xla_flags"] = self.xla_flags
    d["dynamo"] = self.dynamo
    d["test"] = self.test
    d["batch_size"] = self.batch_size
    return d
