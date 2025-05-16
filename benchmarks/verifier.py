import copy
import logging
import torch
import traceback

from benchmark_experiment import ExperimentLoader
from benchmark_model import ModelLoader
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from util import cleanup, move_to_device, reset_rng_state, StrOrBool

logger = logging.getLogger(__name__)


class VerificationCode(str, Enum):
  # Verification passed accuracy check.
  PASS = 'PASS'
  # Verification failed accuracy check.
  FAIL = 'FAIL',
  # Eager execution failed.
  EAGER_FAILED = 'EAGER_FAILED'
  # An exception was raised when running the verifier.
  EXCEPTION_RAISED = 'EXCEPTION_RAISED'
  # Eager runs do not agree.
  NONDETERMINISTIC_EAGER_RUN = 'NONDETERMINISTIC_EAGER_RUN'
  # Verifier skipped.
  VERIFIER_SKIPPED = 'VERIFIER_SKIPPED'
  # Verifier did not run. It was skipped, but due to an unexpected reason.
  # Either an exception was raised or the process timeout.
  VERIFIER_DIDNT_RUN = 'VERIFIER_DIDNT_RUN'


class VerificationException(Exception):

  def __init__(self, code: VerificationCode) -> None:
    super().__init__(f"verifier failed with code: {code}")
    self.code = code


def verify(
    runner: "ExperimentRunner",
    experiment_config: Dict[str, Optional[StrOrBool]],
    model_config: Dict[str, Optional[StrOrBool]],
    tolerance: float,
    use_cosine_similarity: bool,
) -> VerificationCode:
  """Verify the accuracy of the given experiment and model configurations.

  Both `tolerance` and `use_cosine_similarity` will be used when checking whether the
  accuracy of the actual experiment is close to that of eager.
  """

  try:
    # 1. Run eager twice, so as to make sure the model actually outputs deterministic results.
    try:
      eager_output = _run(runner, experiment_config, model_config, eager=True)
      additional_eager_output = _run(
          runner, experiment_config, model_config, eager=True)
    except Exception as e:
      raise VerificationException(VerificationCode.EAGER_FAILED) from e

    # If the results are not close, it might mean that this model is not deterministic.
    # Therefore, we give up the verification process, entirely.
    if not _same(eager_output, additional_eager_output):
      return VerificationCode.NONDETERMINISTIC_EAGER_RUN

    # 2. Compute the output using float64 precision for increased precision. This should
    #    help deciding whether the outputs of the actual experiment have acceptable accuracy.
    try:
      eager_fp64_output = _run(
          runner, experiment_config, model_config, force_fp64=True, eager=True)
    except:
      logger.warning(
          "failed running fp64 golden ref. Setting accuracy to cosine.")
      eager_fp64_output = None
      use_cosine_similarity = True

    # 3. Compute the output of the actual experiment.
    output = _run(runner, experiment_config, model_config)

    # Check whether the results computed by (3) are close to that of eager (1) by using
    # the higher-precision output (2) and the given tolerance and cosine similarity.
    if not _same(
        eager_output,
        output,
        fp64_ref=eager_fp64_output,
        cos_similarity=use_cosine_similarity,
        tol=tolerance,
    ):
      return VerificationCode.FAIL
  except VerificationException:
    raise
  except Exception as e:
    # If anything went wrong (other than an explicit VerificationException), raise
    # a VerificationException with EXCEPTION_RAISED code, while chaining the cause.
    raise VerificationException(VerificationCode.EXCEPTION_RAISED) from e

  return VerificationCode.PASS


def _run(
    runner: "ExperimentRunner",
    experiment_config: Dict[str, Optional[StrOrBool]],
    model_config: Dict[str, Optional[StrOrBool]],
    force_fp64: bool = False,
    eager: bool = False,
) -> Any:
  """Runs a possibly modified experiment, returning the output of the model.

  `force_fp64` loads the model by forcing float64 data-type.

  `eager` modifies the experiment configuration, running it with PyTorch eager.
  """
  model = None

  try:
    if eager:
      experiment_config = _apply_eager_config(experiment_config)

    experiment = runner.experiment_loader.load_experiment(experiment_config)
    reset_rng_state(experiment)

    force_dtype = torch.float64 if force_fp64 else None
    model = runner.model_loader.load_model(
        model_config, experiment, force_dtype=force_dtype)

    iterations = runner._args.verify_iterations
    inputs = copy.deepcopy(model.example_inputs)

    def maybe_sync():
      runner._sync(experiment)

    def maybe_synchronize():
      runner._synchronize(experiment)

    maybe_sync()
    maybe_synchronize()

    with model.pick_grad():
      for i in range(iterations):
        # Collect grad results only in the last run.
        collect_full_output = i == iterations - 1
        output = model.model_iter_fn(
            inputs, collect_full_output=collect_full_output)
        maybe_sync()

    maybe_synchronize()
    return output
  finally:
    if model is not None:
      # Delete the model for saving up memory.
      del model
      # Clean-up CUDA as well.
      cleanup(cuda=True)


def _apply_eager_config(experiment):
  experiment = experiment.copy()
  experiment['dynamo'] = None
  experiment['xla'] = None
  return experiment


def _collect(out: Any) -> List[Any]:
  """Collect leaf objects into a nested list.

  This function uses the conditions used by `torch._dynamo.utils.same` for
  collecting recursively the objects that will be compared by that function.

  This is needed because, otherwise, we can't move ALL XLA tensors to be on
  the same device as the eager runs. Without that move, `torch.allclose`
  calls the XLA kernel with a CPU/CUDA tensor, failing due to an assertion
  error.
  """

  def collect_impl(out: Any) -> List[Any]:
    if isinstance(out, (list, tuple, torch.nn.ParameterList, torch.Size)):
      return [_collect(x) for x in out]
    elif type(out).__name__ == "QuestionAnsweringModelOutput":
      return _collect(out.loss)
    elif isinstance(out, dict):
      return [_collect(out[k]) for k in sorted(out.keys())]
    elif type(out).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
      return [_collect(getattr(out, k)) for k in sorted(out.__dict__.keys())]
    else:
      return [out]

  # Have the first element of ever list in this nested list be the type
  # name of this composite class.
  typename = type(out).__name__
  return [typename] + collect_impl(out)


def _maybe_get_device(r: Any) -> Optional[torch.device]:
  """Get the device from a tensor inside `r`, if one exists.

  Recursively go through `r`, looking for a tensor. Once found,
  return its device. Otherwise, return None.

  This is used so that we can move the XLA result into the same
  device as the eager output.
  """
  if isinstance(r, torch.Tensor):
    return r.device
  if isinstance(r, list):
    for x in r:
      maybe_device = _maybe_get_device(x)
      if maybe_device is not None:
        return maybe_device
  return None


def _same(
    ref: Any,
    res: Any,
    fp64_ref: Any = None,
    cos_similarity: bool = False,
    tol: float = 0,
    equal_nan: bool = True,
) -> bool:
  ref = _collect(ref)
  res = _collect(res)
  fp64_ref = _collect(fp64_ref)

  # If there's a tensor in `ref`, then retrieve its device.
  # Do the the same for `res`. They should agree on whether there should
  # be a tensor in the output (i.e. both None or both not None).
  ref_device = _maybe_get_device(ref)
  res_device = _maybe_get_device(res)
  assert not ((res_device is not None) ^ (ref_device is not None)), (
      "the device found of (i) the result and (ii) the reference should be either: both None or both not None. "
      f"Found: {res_device} (result) vs. {ref_device} (reference).")

  if ref_device is not None:
    # Then, move `res` to the found device, so that we have no errors when
    # trying to call `allclose`.
    res = move_to_device(res, ref_device)

  return torch._dynamo.utils.same(
      ref,
      res,
      fp64_ref=fp64_ref,
      cos_similarity=cos_similarity,
      tol=tol,
      equal_nan=equal_nan,
  )
