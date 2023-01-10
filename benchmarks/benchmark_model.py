import logging
import types


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

  def load_model(self, model_config, benchmark_experiment):
    if model_config["model_name"] != "dummy":
      raise NotImplementedError

    device = benchmark_experiment.get_device()

    benchmark_model = BenchmarkModel(module=None, example_input=None, optimizer=None, device=device, benchmark_experiment=benchmark_experiment)

    def train(self):
      print(self.benchmark_experiment.accelerator, self.benchmark_experiment.xla, self.benchmark_experiment.test)
      print(self.device)
      if self.benchmark_experiment.xla:
        import torch_xla.core.xla_model as xm
        print(xm.xla_real_devices([self.device]))
      return None

    def eval(self):
      print(self.benchmark_experiment.accelerator, self.benchmark_experiment.xla, self.benchmark_experiment.test)
      print(self.device)
      if self.benchmark_experiment.xla:
        import torch_xla.core.xla_model as xm
        print(xm.xla_real_devices([self.device]))
      return None

    benchmark_model.train = types.MethodType(train, benchmark_model)
    benchmark_model.eval = types.MethodType(eval, benchmark_model)

    return benchmark_model

class TorchBenchModelLoader(ModelLoader):

  def __init__(self, args):
    super(TorchBenchModelLoader, self).__init__(args)


class BenchmarkModel:

  def __init__(self, module, example_input, optimizer, device, benchmark_experiment):
    self.module = module
    self.example_input = example_input
    self.optimizer = optimizer
    self.device = device
    self.benchmark_experiment = benchmark_experiment

  def train(self):
    return

  def eval(self):
    return
