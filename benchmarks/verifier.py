import logging
import torch
import traceback

from benchmark_experiment import ExperimentLoader
from benchmark_model import ModelLoader
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional
from util import cleanup, reset_rng_state, StrOrBool

logger = logging.getLogger(__name__)


class VerificationCode(str, Enum):
  # Verification passed accuracy check.
  PASS = 'PASS'
  # Verification failed accuracy check.
  FAIL = 'FAIL',
  # Eager execution failed.
  EAGER_FAILED = 'EAGER_FAILED'
  # Verifier failed, raising an exception.
  VERIFIER_FAILED = 'VERIFIER_FAILED'
  # Eager runs do not agree.
  NONDETERMINISTIC_EAGER_RUN = 'NONDETERMINISTIC_EAGER_RUN'
  # Verifier skipped.
  VERIFIER_SKIPPED = 'VERIFIER_SKIPPED'
  # Verifier did not run. It was skipped, but due to an unexpected reason.
  # Either an exception was raised or the process timeout.
  VERIFIER_SKIPPED_UNEXPECTEDLY = 'VERIFIER_SKIPPED_UNEXPECTEDLY'


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
    except:
      traceback.print_exc()
      return VerificationCode.EAGER_FAILED

    # If the results are not close, it might mean that this model is not deterministic.
    # Therefore, we give up the verification process, entirely.
    if not _same(eager_output, additional_eager_output):
      return VerificationCode.NONDETERMINISTIC_EAGER_RUN

    # 2. Compute the output using float64 precision for increased precision. This should
    #    help deciding whether the outputs of the actual experiment have acceptable accuracy.
    eager_fp64_output = _run(
        runner, experiment_config, model_config, force_fp64=True, eager=True)

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
  except:
    traceback.print_exc()
    return VerificationCode.VERIFIER_FAILED

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
  try:
    reset_rng_state(experiment)

    if eager:
      experiment_config = _apply_eager_config(experiment_config)

    experiment = runner.experiment_loader.load_experiment(experiment_config)
    model = runner.model_loader.load_model(
        model_config,
        eager_benchmark_experiment,
        force_dtype=torch.float64 if force_fp64 else None,
    )

    experiment, model = get_benchmark_model(
        runner,
        experiment_config,
        model_config,
        force_fp64=force_fp64,
        eager=eager,
    )

    n = runner._args.iterations_per_run
    inputs = copy.deepcopy(model.example_inputs)

    def maybe_mark_step():
      runner._mark_step(experiment)

    def maybe_synchronize():
      runner._synchronize(experiment)

    maybe_mark_step()
    maybe_synchronize()

    with model.pick_grad():
      for i in range(runner._args.iterations_per_run):
        # Collect grad results only in the last run.
        collect_full_output = i == n - 1
        output = model.model_iter_fn(
            inputs, collect_full_output=collect_full_output)
        maybe_mark_step()

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


def _same(
    out1: Any,
    out2: Any,
    fp64_ref: Any = None,
    cos_similarity: bool = False,
    tol: float = 0,
    equal_nan: bool = True,
) -> bool:
  return torch._dynamo.utils.same(
      out1,
      out2,
      fp64_ref=fp64_ref,
      cos_similarity=cos_similarity,
      tol=tol,
      equal_nan=equal_nan,
  )
