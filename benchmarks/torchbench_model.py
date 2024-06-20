import functools
import gc
import contextlib
import importlib
import logging
import os
import sys
import torch
import torch.amp
import torch.nn as nn
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs
import torch_xla
import torch_xla.core.xla_model as xm
import types
import yaml
from util import move_to_device, set_cwd, get_torchbench_test_name, find_near_file
from benchmark_model import ModelLoader, BenchmarkModel

logger = logging.getLogger(__name__)

# torchbench models that might OOM using Adam.
# This list was extracted from PyTorch's repository: benchmarks/dynamo/common.py
TRAIN_WITH_SGD = {
    "BERT_pytorch",
    "LearningToPaint",
    "alexnet",
    "dcgan",
    "demucs",
    "densenet121",
    "dlrm",
    "fastNLP_Bert",
    "mobilenet_v2",
    "phlippe_densenet",
    "phlippe_resnet",
    "pytorch_stargan",
    "resnet18",
    "shufflenet_v2_x1_0",
    "speech_transformer",
    "squeezenet1_1",
    "stable_diffusion_text_encoder",
    "timm_efficientdet",
    "timm_nfnet",
    "timm_regnet",
    "timm_vision_transformer",
    "timm_vovnet",
    "vgg16",
    "hf_T5",
    # PyTorch/benchmark sets its optimizer as SGD.
    # Otherwise, OOMs.
    "llama_v2_7b_16h",
}

# Skip the experiment of a model if any of the experiment configs in the list is fully matched
DENY_LIST = {
    "cm3leon_generate": [
        {
            "test": "eval",
            "xla": "PJRT",
            "dynamo": None,
        },  # TIMEOUT
    ],
    "hf_T5_generate": [
        {
            "test": "eval",
            "xla": "PJRT",
            "dynamo": None,
        },  # TIMEOUT
    ],
    "mobilenet_v2_quantized_qat": [
        {
            "test": "eval",
            "accelerator": "cuda"
        },  # The eval test only supports CPU
        {
            "test": "eval",
            "accelerator": "tpu"
        },  # The eval test only supports CPU
    ],
    "resnet50_quantized_qat": [
        {
            "test": "eval",
            "accelerator": "cuda"
        },  # The eval test only supports CPU
        {
            "test": "eval",
            "accelerator": "tpu"
        },  # The eval test only supports CPU
    ],
}

# Models that had more graphs to be compiled than the actual size of
# the cache.
NEED_LARGER_CACHE = {
    "cm3leon_generate",
    "hf_T5_generate",
}


@functools.lru_cache(maxsize=1)
def config_data():
  """Retrieve the skip data in the PyTorch YAML file.

  Reads the YAML file in PyTorch's dynamo benchmarks directory, and transform
  its lists of models into sets of models.
  """

  benchmarks_dynamo_dir = find_near_file(
      ("pytorch/benchmarks/dynamo", "benchmarks/dynamo"))
  assert benchmarks_dynamo_dir is not None, "PyTorch benchmarks folder not found."

  skip_file = os.path.join(benchmarks_dynamo_dir, "torchbench.yaml")
  with open(skip_file) as f:
    data = yaml.safe_load(f)

  def flatten(lst):
    for item in lst:
      if isinstance(item, list):
        yield from flatten(item)
      else:
        yield item

  def maybe_list_to_set(obj):
    if isinstance(obj, dict):
      return {k: maybe_list_to_set(v) for k, v in obj.items()}
    if isinstance(obj, list):
      return set(flatten(obj))
    return obj

  return maybe_list_to_set(data)


# List of models retrieved from the YAML configuration data.
# File (inside PyTorch main repository): benchmarks/dynamo/torchbench.yaml
DETECTRON2_MODELS = config_data()["detectron2_models"]

FORCE_AMP_FOR_FP16_BF16_MODELS = config_data(
)["dtype"]["force_amp_for_fp16_bf16_models"]

FORCE_FP16_FOR_BF16_MODELS = config_data(
)["dtype"]["force_fp16_for_bf16_models"]


class TorchBenchModelLoader(ModelLoader):

  def __init__(self, args):
    super().__init__(args)
    self.benchmark_model_class = TorchBenchModel
    self.torchbench_dir = self.add_torchbench_dir()

  def add_torchbench_dir(self):
    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam

    torchbench_dir = find_near_file(
        ("torchbenchmark", "torchbench", "benchmark"))
    assert torchbench_dir is not None, "Torch Benchmark folder not found."

    if torchbench_dir is not None:
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

  @property
  def skip(self):
    return config_data()["skip"]

  def is_compatible(self, dummy_benchmark_model, benchmark_experiment):
    name = dummy_benchmark_model.model_name
    test = get_torchbench_test_name(benchmark_experiment.test)

    if name in self.skip["all"]:
      return False

    if name in self.skip["test"].get(test, {}):
      return False

    if name in self.skip["device"].get(benchmark_experiment.accelerator, {}):
      return False

    if name in self.skip["multiprocess"]:
      # No support for multiprocess, yet. So, skip all benchmarks that
      # only work with it.
      return False

    def is_attr_eq(k, v):
      return getattr(benchmark_experiment, k) == v

    for deny_experiment_config in DENY_LIST.get(name, []):
      if all(is_attr_eq(k, v) for k, v in deny_experiment_config.items()):
        return False

    return True


class TorchBenchModel(BenchmarkModel):

  def __init__(self, suite_name, model_name, benchmark_experiment):
    super().__init__(suite_name, model_name, benchmark_experiment)

  def _cleanup(self):
    # Garbage-collect right now.
    gc.collect()

    # If we are using CUDA, clean-up its cache left-over.
    if self.is_accelerator_cuda():
      torch.cuda.empty_cache()

  def set_up(self):
    """Set up module, actual batch_size, example_inputs, and optimizer_class

    This is model suite specific.
    """
    # Set the optimizer class.
    # Check if we should use SGD instead of Adam for memory reasons.
    if self.benchmark_experiment.test == "train" and self.model_name in TRAIN_WITH_SGD:
      self.optimizer_class = torch.optim.SGD
    else:
      self.optimizer_class = torch.optim.Adam

    # Setup the autocast environment if we are running on AMP precision.
    self.autocast, self.autocast_kwargs = self._get_autocast_with_kwargs()

    # Load the actual benchmark instance.
    benchmark = self.load_benchmark()

    self.module, self.example_inputs = benchmark.get_module()
    if isinstance(self.example_inputs,
                  dict) and "input_ids" in self.example_inputs:
      self.example_inputs = (self.example_inputs['input_ids'],)
    self.benchmark_experiment.batch_size = benchmark.batch_size

    # Move the initialized model to XLA device if it's not there already.
    if self.benchmark_experiment.xla and not self.should_initialize_on_xla():
      # First, move the model and the inputs to CPU.
      # This avoids having dupplicated data on CUDA.
      keep_model_data_on_cuda = self.benchmark_experiment.keep_model_data_on_cuda
      if self.is_accelerator_cuda() and not keep_model_data_on_cuda:
        self.module = self.module.to("cpu")
        self.example_inputs = move_to_device(self.example_inputs, "cpu")
        self._cleanup()

    # Torchbench has quite different setup for yolov3, so directly passing
    # the right example_inputs
    if self.model_name == "yolov3":
      self.example_inputs = (torch.rand(self.benchmark_experiment.batch_size, 3,
                                        384, 512),)

    del benchmark
    self._cleanup()

  @functools.lru_cache(maxsize=1)
  def benchmark_cls(self):
    for module_src in [
        f"torchbenchmark.models.{self.model_name}",
        f"torchbenchmark.models.fb.{self.model_name}"
    ]:
      try:
        module = importlib.import_module(module_src)
        return getattr(module, "Model", None)
      except ModuleNotFoundError:
        logger.warning(f"Unable to import {module_src}.")
    return None

  @property
  def batch_size(self):
    return config_data()["batch_size"]

  def load_benchmark(self):
    cant_change_batch_size = (
        not getattr(self.benchmark_cls(), "ALLOW_CUSTOMIZE_BSIZE", True) or
        self.model_name in config_data()["dont_change_batch_size"])

    if cant_change_batch_size:
      self.benchmark_experiment.batch_size = None

    batch_size = self.benchmark_experiment.batch_size

    if batch_size is None:
      if self.is_training() and self.model_name in self.batch_size["training"]:
        batch_size = self.batch_size["training"][self.model_name]
      elif self.is_inference(
      ) and self.model_name in self.batch_size["inference"]:
        batch_size = self.batch_size["inference"][self.model_name]

    # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
    # torch.backends.__allow_nonbracketed_mutation_flag = True

    if self.should_initialize_on_xla():
      device = "xla"
    else:
      # Initialize the model in the given accelerator first. If we are supposed
      # to run on XLA device, move it later. We do this for a couple of reasons:
      #
      #   1. To make sure we are comparing the same model on CUDA and XLA
      #   2. There are some models that only expect either CUDA or CPU
      device = self.benchmark_experiment.accelerator

    return self.benchmark_cls()(
        test=self.benchmark_experiment.test,
        device=device,
        batch_size=batch_size,
    )

  def update_process_env(self, process_env):
    if self.model_name in NEED_LARGER_CACHE:
      process_env["XLA_COMPILATION_CACHE_SIZE"] = "2048"

  def pick_grad(self):
    # special case
    if self.model_name in ("maml",):
      return torch.enable_grad()
    return super().pick_grad()

  def should_initialize_on_xla(self):
    # Reasons why we need to initialize the benchmark on XLA directly:
    #
    #   1. Models don't expect 'tpu' as their device.
    #   2. 'moco' initializes a ProcessGroup, i.e. the backend depends on
    #      the given device
    return self.is_accelerator_tpu() or self.model_name == "moco"

  def is_inference(self):
    return self.benchmark_experiment.test == "eval"

  def is_training(self):
    return self.benchmark_experiment.test == "train"

  def is_accelerator_cuda(self):
    return self.benchmark_experiment.accelerator == "cuda"

  def is_accelerator_tpu(self):
    return self.benchmark_experiment.accelerator == "tpu"

  def use_amp(self):
    return self.is_training(
    ) or self.model_name in FORCE_AMP_FOR_FP16_BF16_MODELS

  def use_fp16(self):
    return self.is_inference() and self.model_name in FORCE_FP16_FOR_BF16_MODELS

  def conversion_dtype(self):
    if self.is_training() or self.use_amp():
      return super().conversion_dtype()

    # From here, we are running inference without AMP, for sure.
    # Do we have to use float16, instead of bfloat16?
    if self.use_fp16():
      return torch.float16

    return torch.bfloat16

  def _get_autocast_with_kwargs(self):
    kwargs = {}
    if self.use_amp():
      # TODO: Should call device specific autocast implementations.
      # Specifically, we should be using:
      #   - torch.cuda.amp.autocast for inductor
      #   - torch_xla.amp.autocast for PyTorch/XLA experiments.
      # PyTorch/XLA autocast does not run with dynamo, though:
      # https://github.com/pytorch/xla/issues/6511
      if self.is_accelerator_cuda():
        # For inductor and XLA:CUDA, we use CUDA autocast.
        autocast = torch.cuda.amp.autocast
        kwargs["dtype"] = torch.float16
      elif self.is_accelerator_tpu():
        autocast = torch.amp.autocast
        kwargs["device_type"] = "xla"
        kwargs["dtype"] = torch.bfloat16
      else:
        # Error: AMP is only supported on XLA:CUDA and XLA:TPU.
        name = self.model_name
        accelerator = self.benchmark_experiment.accelerator
        raise RuntimeError(f"Tried to run {name} with AMP on {accelerator}. "
                           "However, AMP is only supported on cuda and tpu.")
    else:
      autocast = contextlib.nullcontext
    return (autocast, kwargs)

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

  def train(self, inputs, collect_full_output=False):
    if self.model_name in DETECTRON2_MODELS:
      from detectron2.utils.events import EventStorage
      with EventStorage():
        super().train(inputs, collect_full_output=collect_full_output)
    else:
      super().train(inputs, collect_full_output=collect_full_output)
