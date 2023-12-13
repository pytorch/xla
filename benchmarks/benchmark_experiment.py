from collections import OrderedDict
import logging
import os
import torch
import torch._dynamo as dynamo
import torch_xla.core.xla_model as xm
from util import is_xla_device_available, get_accelerator_model

logger = logging.getLogger(__name__)


class ExperimentLoader:

  def __init__(self, args):
    self._args = args

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

    experiment_configs = []
    for experiment_config in self.expand_config_choices(config_choices):
      if not self._is_available(experiment_config):
        continue

      self._add_experiment_env(experiment_config)
      experiment_configs.append(experiment_config)
    return experiment_configs

  def _expand_config_choices(self, config_choices):
    configs = [{}]
    for k, choices in config_choices.items():
      new_configs = []
      for base_cfg in configs:
        for c in choices:
          new_cfg = base_cfg.copy()
          new_cfg[k] = c
          new_configs.append(new_cfg)
      configs = new_configs
    return configs

  def _is_available(self, experiment_config):
    cfg_dynamo = experiment_config["dynamo"]
    cfg_accelerator = experiment_config["accelerator"]
    cfg_xla = experiment_config["xla"]
    cfg_test = experiment_config["test"]

    # Check that dynamo refers to an existing backend.
    if cfg_dynamo is not None and cfg_dynamo not in dynamo.list_backends(
        exclude_tags=()):
      return False

    # Check dynamo backend-specifics constraints.
    if cfg_dynamo == "inductor":
      if cfg_accelerator != "cuda" or cfg_xla is not None:
        return False
    elif cfg_dynamo == "openxla_eval":
      if cfg_xla is None or cfg_test != "eval":
        return False
    elif cfg_dynamo == "openxla":
      if cfg_xla is None:
        return False
    else:
      raise NotImplementedError

    # Check XLA device available if requested.
    if cfg_xla is not None and not is_xla_device_available(
        cfg_accelerator.upper()):
      return False

    # Check accelerator contraints.
    if cfg_accelerator == "tpu":
      if cfg_xla is None:
        return False
    elif cfg_accelerator == "cuda":
      if cfg_xla is None and not torch.cuda.is_available():
        return False
    elif cfg_accelerator == "cpu":
      pass
    else:
      raise NotImplementedError

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
    accelerator = experiment_config["accelerator"]
    xla = experiment_config["xla"]
    xla_flags = experiment_config["xla_flags"]
    dynamo = experiment_config["dynamo"]
    test = experiment_config["test"]
    batch_size = experiment_config.get("batch_size", self._args.batch_size)
    benchmark_experiment = BenchmarkExperiment(
        accelerator=accelerator,
        xla=xla,
        xla_flags=xla_flags,
        dynamo=dynamo,
        test=test,
        batch_size=batch_size)

    return benchmark_experiment


class BenchmarkExperiment:

  def __init__(self, accelerator, xla, xla_flags, dynamo, test, batch_size):
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
    d = self.to_dict()

    # Remove these 2 components that may end up making the filename too big.
    d.pop("accelerator_model", None)
    d.pop("xla_flags", None)

    return "-".join(str(v) for v in self.to_dict().values()).replace(" ", "")

  def to_dict(self):
    d = OrderedDict()
    d["accelerator"] = self.accelerator
    d["accelerator_model"] = self.accelerator_model
    d["xla"] = self.xla
    d["xla_flags"] = self.xla_flags
    d["dynamo"] = self.dynamo
    d["test"] = self.test
    d["batch_size"] = self.batch_size
    return d
