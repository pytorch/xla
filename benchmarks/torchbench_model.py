import gc
import importlib
import logging
import os
from os.path import abspath, exists
import sys
import torch
import torch.nn as nn
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs
import types
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
}

# Skip the experiment of a model if any of the experiment configs in the list is fully matched
DENY_LIST = {
    "cm3leon_generate": [
        {
            "test": "train",
        },
        {
            "test": "eval",
            "xla": "PJRT",
        },
    ],  # no install.py
    "hf_T5_generate": [
        {
            "test": "train",
        },
        {
            "test": "eval",
            "xla": "PJRT",
        },
    ],  # no install.py
    "doctr_det_predictor": [{
        "test": "train"
    },],  # not implemented
    "doctr_reco_predictor": [{
        "test": "train"
    },],  # not implemented
    "detectron2_fcos_r_50_fpn": [{
        "test": "train"
    },],  # not implemented
    # https://github.com/pytorch/torchdynamo/issues/145
    "fambench_xlmr": [{}],
    "llama": [{
        "test": "train"
    },],  # not implemented
    "mobilenet_v2_quantized_qat": [
        {
            "test": "eval",
            "accelerator": "cuda"
        },  # not implemented
        {
            "test": "eval",
            "accelerator": "tpu"
        },
    ],  # not implemented
    # self.load_benchmark() exits the main process. See issue #6207.
    "pytorch_CycleGAN_and_pix2pix": [{}],
    "pyhpc_equation_of_state": [{
        "test": "train"
    },],  # not implemented
    "pyhpc_isoneutral_mixing": [{
        "test": "train"
    },],  # not implemented
    "pyhpc_turbulent_kinetic_energy": [{
        "test": "train"
    },],  # not implemented
    "pytorch_struct": [{
        "test": "eval"
    },],  # not implemented
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
        },  # not implemented
        {
            "test": "eval",
            "accelerator": "tpu"
        },
    ],  # not implemented
    "tacotron2": [
        {
            # self.load_benchmark() exits the main process. See issue #6207.
            "xla": "PJRT",
        },
    ],
    # https://github.com/pytorch/pytorch/issues/99438
    "vision_maskrcnn": [{}],
}

# This strict deny list denies tests that hold for too long and timeoout.
STRICT_DENY_LIST = {
    **{
        "opacus_cifar10": [{
            "accelerator": "tpu",
        },],  # stackdump issue in TPU
        "pytorch_stargan": [{
            "accelerator": "tpu",
        },],  # stackdump issue in TPU
        "soft_actor_critic": [{
            "accelerator": "tpu",
        },],  # stackdump issue in TPU
        "speech_transformer": [{
            "accelerator": "tpu",
        },],  # stackdump issue in TPU
    },
    **DENY_LIST
}


class TorchBenchModelLoader(ModelLoader):

  def __init__(self, args):
    super().__init__(args)
    self.benchmark_model_class = TorchBenchModel
    self.torchbench_dir = self.add_torchbench_dir()

  def add_torchbench_dir(self):
    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    for torchbench_dir in (
        "./torchbenchmark",
        "./torchbench",
        "./benchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ):
      if exists(torchbench_dir):
        break

    if exists(torchbench_dir):
      torchbench_dir = abspath(torchbench_dir)
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

  def is_compatible(self,
                    dummy_benchmark_model,
                    benchmark_experiment,
                    use_strict_deny=False):
    deny_list = STRICT_DENY_LIST if use_strict_deny else DENY_LIST
    if dummy_benchmark_model.model_name in deny_list:
      for deny_experiment_config in deny_list[dummy_benchmark_model.model_name]:
        matched = True
        for k, v in deny_experiment_config.items():
          if getattr(benchmark_experiment, k) != v:
            matched = False
            break
        if matched:
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
    if self.benchmark_experiment.test == "train" and self.model_name in TRAIN_WITH_SGD:
      self.optimizer_class = torch.optim.SGD
    else:
      self.optimizer_class = torch.optim.Adam

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

      device = self.benchmark_experiment.get_device()
      self.module = self.module.to(device)
      self.example_inputs = move_to_device(self.example_inputs, device)

    # Torchbench has quite different setup for yolov3, so directly passing
    # the right example_inputs
    if self.model_name == "yolov3":
      self.example_inputs = (torch.rand(self.benchmark_experiment.batch_size, 3,
                                        384, 512),)

    del benchmark
    self._cleanup()

  def load_benchmark(self):
    try:
      module = importlib.import_module(
          f"torchbenchmark.models.{self.model_name}")
    except ModuleNotFoundError:
      module = importlib.import_module(
          f"torchbenchmark.models.fb.{self.model_name}")
    benchmark_cls = getattr(module, "Model", None)

    cant_change_batch_size = (not getattr(benchmark_cls,
                                          "ALLOW_CUSTOMIZE_BSIZE", True))
    if cant_change_batch_size:
      self.benchmark_experiment.batch_size = None

    # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
    # torch.backends.__allow_nonbracketed_mutation_flag = True

    # torchbench uses `xla` as device instead of `tpu`
    if device := self.benchmark_experiment.accelerator == 'tpu':
      device = str(self.benchmark_experiment.get_device())
    return benchmark_cls(
        test=self.benchmark_experiment.test,
        device=device,
        batch_size=self.benchmark_experiment.batch_size,
    )

  @property
  def default_precision_flag(self):
    """
    Get the default precision config to XLA, if present.

    Whenever a model has a default precision for cuda set
    we need to set proper environment flags so XLA catches
    the requird precision.

    This function is a workaround. Proper solution requires
    changes to the PT/XLA bridge so that the input shape
    is properly inferred after issuing converts to `torch.nn.Module`.
    """
    test = self.benchmark_experiment.test
    try:
      benchmark = self.load_benchmark()
    except Exception:
      logger.exception("Cannot load benchmark model")
      return None

    if test == "eval" and hasattr(benchmark, 'DEFAULT_EVAL_CUDA_PRECISION'):
      precision = benchmark.DEFAULT_EVAL_CUDA_PRECISION
    elif test == "train" and hasattr(benchmark, 'DEFAULT_TRAIN_CUDA_PRECISION'):
      precision = benchmark.DEFAULT_TRAIN_CUDA_PRECISION
    else:
      precision = None
      logger.warning("No default precision set. No patching needed.")

    del benchmark
    self._cleanup()

    precision_flag = None
    if precision is None:
      return None
    if precision == "fp16":
      precision_flag = 'XLA_USE_FP16'
    elif precision == "amp":
      raise ValueError(
          f"AMP for PT/XLA:GPU is not implemented yet for torchbench models")
    elif precision == "bf16":
      precision_flag = 'XLA_USE_BF16'
    elif precision == "fp32":
      logger.warning("Sticking with the default fp32 precision.")
    else:
      raise ValueError(f"Unknown precision: {precision}")
    return precision_flag

  def update_process_env(self, process_env):
    precision_flag = self.default_precision_flag
    if precision_flag is not None:
      process_env[precision_flag] = '1'

  def pick_grad(self):
    # special case
    if self.model_name in ("maml",):
      return torch.enable_grad()

    if self.benchmark_experiment.test == "eval":
      return torch.no_grad()
    elif self.benchmark_experiment.test == "train":
      return torch.enable_grad()

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
