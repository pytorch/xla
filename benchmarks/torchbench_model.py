import functools
import gc
import contextlib
import importlib
import logging
import os
from os.path import abspath, exists
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
from util import move_to_device, set_cwd
from benchmark_model import ModelLoader, BenchmarkModel

logger = logging.getLogger(__name__)

DETECTRON2_MODELS = {
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    "detectron2_maskrcnn",
    "detectron2_fcos_r_50_fpn",
}

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
            "test": "train",
        },  # Model's DEFAULT_TRAIN_BSIZE is not implemented
        {
            "test": "eval",
            "xla": "PJRT",
            "dynamo": None,
        },  # TIMEOUT
    ],
    "hf_T5_generate": [
        {
            "test": "train",
        },  # Model's DEFAULT_TRAIN_BSIZE is not implemented
        {
            "test": "eval",
            "xla": "PJRT",
            "dynamo": None,
        },  # TIMEOUT
    ],
    "doctr_det_predictor": [{
        "test": "train"
    },],  # Model's DEFAULT_TRAIN_BSIZE is not implemented
    "doctr_reco_predictor": [{
        "test": "train"
    },],  # Model's DEFAULT_TRAIN_BSIZE is not implemented
    "detectron2_fcos_r_50_fpn": [{
        "test": "train"
    },],  # FCOS train is not supported by upstream detectron2
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
    # self.load_benchmark() exits the main process. See issue #6207.
    "pytorch_CycleGAN_and_pix2pix": [{}],
    "pyhpc_equation_of_state": [{
        "test": "train"
    },],  # Model's DEFAULT_TRAIN_BSIZE is not implemented
    "pyhpc_isoneutral_mixing": [{
        "test": "train"
    },],  # Model's DEFAULT_TRAIN_BSIZE is not implemented
    "pyhpc_turbulent_kinetic_energy": [{
        "test": "train"
    },],  # Model's DEFAULT_TRAIN_BSIZE is not implemented
    "pytorch_unet": [
        {
            # self.load_benchmark() exits the main process. See issue #6207.
            "xla": "PJRT",
        },
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

FORCE_AMP_FOR_FP16_BF16_MODELS = {
    "DALLE2_pytorch",
    "doctr_det_predictor",
    "doctr_reco_predictor",
    "Super_SloMo",
    "tts_angular",
    "pyhpc_turbulent_kinetic_energy",
    "detectron2_fcos_r_50_fpn",
}

FORCE_FP16_FOR_BF16_MODELS = {"vision_maskrcnn"}


class TorchBenchModelLoader(ModelLoader):

  def __init__(self, args):
    super().__init__(args)
    self.benchmark_model_class = TorchBenchModel
    self.torchbench_dir = self.add_torchbench_dir()
    self.skip = self.get_skip_data()

  def _find_near_file(self, names):
    """Find a file near the current directory.

    Looks for `names` in the current directory, up to its two direct parents.
    """
    for dir in ("./", "../", "../../", "../../../"):
      for name in names:
        path = os.path.join(dir, name)
        if exists(path):
          return abspath(path)
    return None

  def add_torchbench_dir(self):
    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam

    torchbench_dir = self._find_near_file(
        ("torchbenchmark", "torchbench", "benchmark"))
    assert torchbench_dir is not None, "Torch Benchmark folder not found."

    if torchbench_dir is not None:
      if torchbench_dir not in sys.path:
        sys.path.append(torchbench_dir)
    else:
      raise Exception("Torch Benchmark folder not found.")

    return torchbench_dir

  def get_skip_data(self):
    """Retrieve the skip data in the PyTorch YAML file.

    Reads the YAML file in PyTorch's dynamo benchmarks directory, and transform
    its lists of models into sets of models.
    """

    benchmarks_dynamo_dir = self._find_near_file(
        ("pytorch/benchmarks/dynamo", "benchmarks/dynamo"))
    assert benchmarks_dynamo_dir is not None, "PyTorch benchmarks folder not found."

    skip_file = os.path.join(benchmarks_dynamo_dir,
                             "torchbench_skip_models.yaml")
    with open(skip_file) as f:
      data = yaml.safe_load(f)

    def maybe_list_to_set(obj):
      if isinstance(obj, dict):
        return {k: maybe_list_to_set(v) for k, v in obj.items()}
      if isinstance(obj, list):
        return set(obj)
      return obj

    return maybe_list_to_set(data)

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

  def is_compatible(self, dummy_benchmark_model, benchmark_experiment):
    name = dummy_benchmark_model.model_name

    if name in self.skip["skip"]:
      return False

    if name in self.skip["test"].get(benchmark_experiment.test, {}):
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
    if self.benchmark_experiment.accelerator == "cuda":
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

    self.benchmark_experiment.batch_size = benchmark.batch_size

    # Move the initialized model to XLA device.
    if self.benchmark_experiment.xla:
      # First, move the model and the inputs to CPU.
      # This avoids having dupplicated data on CUDA.
      if self.benchmark_experiment.accelerator == "cuda":
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

  def load_benchmark(self):
    cant_change_batch_size = (not getattr(self.benchmark_cls(),
                                          "ALLOW_CUSTOMIZE_BSIZE", True))
    if cant_change_batch_size:
      self.benchmark_experiment.batch_size = None

    # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
    # torch.backends.__allow_nonbracketed_mutation_flag = True

    # torchbench uses `xla` as device instead of `tpu`
    if (device := self.benchmark_experiment.accelerator) == 'tpu':
      device = str(self.benchmark_experiment.get_device())

    return self.benchmark_cls()(
        test=self.benchmark_experiment.test,
        device=device,
        batch_size=self.benchmark_experiment.batch_size,
    )

  def update_process_env(self, process_env):
    if self.model_name in NEED_LARGER_CACHE:
      process_env["XLA_COMPILATION_CACHE_SIZE"] = "2048"

  def pick_grad(self):
    # special case
    if self.model_name in ("maml",):
      return torch.enable_grad()
    return super().pick_grad()

  def is_inference(self):
    return self.benchmark_experiment.test == "eval"

  def is_training(self):
    return self.benchmark_experiment.test == "train"

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

    # Set the default data-type based on the accelerator.
    if self.benchmark_experiment.accelerator == "cuda":
      kwargs["dtype"] = torch.float16
    else:
      # Both CPU and TPU autocast mode defaults to bfloat16.
      kwargs["dtype"] = torch.bfloat16

    if self.use_amp():
      if self.benchmark_experiment.xla:
        # Should call device specific autocast implementations.
        # PyTorch/XLA autocast does not run with dynamo, though:
        # https://github.com/pytorch/xla/issues/6511
        autocast = torch.amp.autocast
        kwargs["device_type"] = "xla"
      else:
        autocast = torch.cuda.amp.autocast
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
