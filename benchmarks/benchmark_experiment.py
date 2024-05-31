from collections import OrderedDict
import logging
import os
import torch
import torch._dynamo as dynamo
import torch_xla.core.xla_model as xm
from util import parse_none_str, is_xla_device_available, get_accelerator_model

logger = logging.getLogger(__name__)


class ExperimentLoader:

  def __init__(self, args):
    self._args = args

  def list_experiment_configs(self):

    # Start with default config.
    config_choices = {
        "accelerator": ["cpu", "cuda", "tpu"],
        "xla": [None, "PJRT", "XRT"],
        "xla_flags": [None],
        "dynamo": [None, "inductor", "openxla_eval", "openxla"],
        "torch_xla2": [None],  # options only apply to torch_xla2
        "test": ["eval", "train"],
        "keep_model_data_on_cuda": [False],
    }

    # Apply command line choices.
    if self._args.accelerator:
      config_choices["accelerator"] = list(set(self._args.accelerator))
    if self._args.xla:
      config_choices["xla"] = list(map(parse_none_str, set(self._args.xla)))
    if self._args.torch_xla2:
      config_choices["torch_xla2"] = list(
          map(parse_none_str, set(self._args.torch_xla2)))
    if self._args.dynamo:
      config_choices["dynamo"] = list(
          map(parse_none_str, set(self._args.dynamo)))
    if self._args.test:
      config_choices["test"] = list(set(self._args.test))
    if self._args.xla_flags:
      config_choices["xla_flags"] = list(
          map(parse_none_str, set(self._args.xla_flags)))
    if self._args.keep_model_data_on_cuda:
      config_choices["keep_model_data_on_cuda"] = [
          self._args.keep_model_data_on_cuda
      ]

    # Expand experiment configs and add env vars.
    logger.debug(f"Expand experiment configs")
    experiment_configs = []
    for cfg in self._expand_config_choices(config_choices):
      if not self._is_available(cfg):
        continue
      logger.debug(f"Experiment config: {cfg}")
      experiment_configs.append(cfg)
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
    cfg_torch_xla2 = experiment_config["torch_xla2"]
    cfg_keep_model_data_on_cuda = experiment_config["keep_model_data_on_cuda"]

    # Check that dynamo refers to an existing backend.
    if cfg_dynamo is not None and cfg_dynamo not in dynamo.list_backends(
        exclude_tags=()):
      return False

    use_xla2 = True if cfg_torch_xla2 else False
    # torch_xla2 doesn't support dynamo at this time.
    if cfg_dynamo is not None and use_xla2:
      return False

    # Check dynamo backend-specifics constraints.
    if cfg_dynamo == "inductor":
      if cfg_accelerator == "tpu" or cfg_xla is not None:
        return False
    elif cfg_dynamo == "openxla_eval":
      if cfg_xla is None or cfg_test != "eval":
        return False
    elif cfg_dynamo == "openxla":
      if cfg_xla is None:
        return False
    elif cfg_dynamo is None:
      pass
    else:
      raise NotImplementedError

    # Check XLA device available if requested.
    if not use_xla2 and cfg_xla is not None and not is_xla_device_available(
        cfg_accelerator.upper(), use_xla2):
      return False

    # Check accelerator constraints.
    if cfg_accelerator == "tpu":
      if cfg_xla is None:
        return False
    elif cfg_accelerator in ("cpu", "cuda"):
      pass
    else:
      raise NotImplementedError

    # cfg_keep_model_data_on_cuda is only avaible when using dynamo
    if cfg_keep_model_data_on_cuda and cfg_dynamo != "openxla":
      return False

    return True

  def load_experiment(self, experiment_config):
    accelerator = experiment_config["accelerator"].lower()
    xla = experiment_config["xla"]
    xla_flags = experiment_config["xla_flags"]
    dynamo = experiment_config["dynamo"]
    test = experiment_config["test"]
    batch_size = experiment_config.get("batch_size", self._args.batch_size)
    torch_xla2 = experiment_config["torch_xla2"]
    keep_model_data_on_cuda = experiment_config["keep_model_data_on_cuda"]
    return BenchmarkExperiment(
        accelerator=accelerator,
        xla=xla,
        xla_flags=xla_flags,
        dynamo=dynamo,
        torch_xla2=torch_xla2,
        keep_model_data_on_cuda=keep_model_data_on_cuda,
        test=test,
        batch_size=batch_size)


class BenchmarkExperiment:

  def __init__(self, accelerator, xla, xla_flags, dynamo, torch_xla2,
               keep_model_data_on_cuda: bool, test, batch_size):
    self.accelerator = accelerator
    self.xla = xla
    self.xla_flags = xla_flags
    self.dynamo = dynamo
    self.torch_xla2 = torch_xla2
    self.keep_model_data_on_cuda = keep_model_data_on_cuda
    self.test = test
    self.batch_size = batch_size
    self.accelerator_model = get_accelerator_model(self.accelerator)

  def update_process_env(self, process_env):

    # Remove env vars that would interfere with the subprocess.
    if self.xla is not None:
      process_env.pop("PJRT_DEVICE", None)
      process_env.pop("XRT_TPU_CONFIG", None)
      process_env.pop("XLA_FLAGS", None)

    use_xla2 = False
    if self.torch_xla2:
      process_env["JAX_PLATFORMS"] = self.accelerator.lower()
      use_xla2 = True

    if self.xla == "PJRT":
      process_env["PJRT_DEVICE"] = self.accelerator.upper()
    elif self.xla == "XRT":
      if is_xla_device_available("TPU"):
        process_env["TPU_NUM_DEVICES"] = "1"
        process_env["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
      elif is_xla_device_available("CUDA"):
        process_env["GPU_NUM_DEVICES"] = "1"
    elif self.xla is None:
      # In non-xla CPU training experiments, an env var is still needed if an
      # xla device exists, or there will be "Missing XLA configuration" error.
      if is_xla_device_available(self.accelerator.upper(), use_xla2):
        process_env["PJRT_DEVICE"] = self.accelerator.upper()

    if self.xla_flags:
      process_env["XLA_FLAGS"] = self.xla_flags

  def get_device(self):
    if self.torch_xla2:
      # Initiate the model in CPU first for xla2. We will move the model to jax device later.
      # This is because we don't have xm.xla_device() function in torch_xla2.
      return torch.device("cpu")
    if self.xla:
      return xm.xla_device(devkind=self.accelerator.upper())
    elif self.accelerator == "cpu":
      return torch.device("cpu")
    elif self.accelerator == "cuda":
      return torch.device("cuda")
    else:
      raise NotImplementedError

  @property
  def filename_str(self):

    def to_short_string(x):
      max_len = 32
      s = str(x)
      if len(s) > max_len:
        s = str(hex(hash(s)))
      return s

    short_strs = map(to_short_string, self.to_dict().values())
    return "-".join(short_strs).replace(" ", "")

  def to_dict(self):
    d = OrderedDict()
    d["accelerator"] = self.accelerator
    d["accelerator_model"] = self.accelerator_model
    d["xla"] = self.xla
    d["xla_flags"] = self.xla_flags
    d["dynamo"] = self.dynamo
    d["torch_xla2"] = self.torch_xla2
    d["keep_model_data_on_cuda"] = self.keep_model_data_on_cuda
    d["test"] = self.test
    d["batch_size"] = self.batch_size
    return d
