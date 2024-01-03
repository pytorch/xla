import logging
import torch

from benchmark_experiment import ExperimentLoader
from benchmark_model import ModelLoader
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class VerificationCode(str, Enum):
  PASS = 'PASS'
  FAIL = 'FAIL',
  NONDETERMINISTIC_EAGER_RUN = 'INDETERMINISTIC_EAGER_RUN'
  CANNOT_PROCEED_WITH_VERIFICATION = 'CANNOT_PROCEED_WITH_VERIFICATION'
  NO_OUTPUT_PROVIDED = 'NO_OUTPUT_PROVIDED'
  SKIP_VERIFICATION = 'SKIP_VERIFICATION'
  VERIFICATION_FAILED = 'VERIFICATION_FAILED'


@dataclass
class VerificationResult:
  result_code: VerificationCode
  mean_rel_error: Optional[float] = None


def verify(output: torch.Tensor,
           experiment: dict,
           model_config: dict,
           experiment_loader: ExperimentLoader,
           model_loader: ModelLoader,
           mean_rel_error_tolerance: float,
           noop: bool = False):
  """
    Verify the mean relative error for `output` against eager runtime of the model.
    If `mean_rel_error_tolerance` is greater than the calculated mean relative error of
    the output vs. eager run then the function returns `VerificationCode.FAIL`.
    Otherwise, if the run is successful, `VerificationCode.PASS` is returned, along with calculated
    mean relative error.

    :param output: The tensor to compare eager runtime against.
    :type fn: torch.Tensor
    :param experiment: Experiment config of the `output` run.
    :type experiment: dict
    :param model_config: Model config of the `output` run.
    :type model_config: dict
    :param experiment_loader: Experiment loader module.
    :type experiment_loader: ExperimentLoader
    :param model_loader: Model loader module.
    :type model_loader: ModelLoader
    :param mean_rel_error_tolerance: Mean rel error tolerance to check against.
    :type mean_rel_error_tolerance: float
    :param noop: If set to true, function returns `SKIP_VERIFICATION` code.
    :type noop: bool
  """
  if noop:
    return VerificationResult(VerificationCode.SKIP_VERIFICATION)

  if output is None:
    return VerificationResult(VerificationCode.NO_OUTPUT_PROVIDED)

  try:
    experiment = apply_eager_config_(experiment)
    benchmark_experiment = experiment_loader.load_experiment(experiment)
    model = model_loader.load_model(model_config, benchmark_experiment)

    eager_output = run_(model)
    additional_eager_output = run_(model)

    rel_err, close_res = close_results_(eager_output, additional_eager_output,
                                        mean_rel_error_tolerance)
    if not close_res:
      return VerificationResult(VerificationCode.NONDETERMINISTIC_EAGER_RUN,
                                rel_err)

    eager_output = eager_output.to(device=output.device)
    rel_err, close_res = close_results_(eager_output, output,
                                        mean_rel_error_tolerance)
    if close_res:
      return VerificationResult(VerificationCode.PASS, rel_err)
    else:
      log_(eager_output, output)
      return VerificationResult(VerificationCode.FAIL, rel_err)
  except Exception as e:
    logger.exception(e)
    return VerificationResult(VerificationCode.VERIFICATION_FAILED)


def log_(lhs, rhs):
  logger.error('Output differs significantly.')
  lhs = lhs.clone().to(device='cpu')
  rhs = rhs.clone().to(device='cpu')

  logger.error(f"lhs: {lhs}")
  logger.error(f"rhs: {rhs}")


def run_(model):
  inputs = model.example_inputs
  with model.pick_grad():
    output = model.model_iter_fn(inputs)
  return output


def apply_eager_config_(experiment):
  experiment = experiment.copy()
  experiment['dynamo'] = experiment['xla'] = None
  return experiment


def close_results_(lhs, rhs, delta):
  rel_error_tensor = torch.abs(rhs - lhs) / torch.abs(lhs)
  mean_rel_error = (rel_error_tensor.sum() / lhs.numel()).item()
  return mean_rel_error, (mean_rel_error < delta)
