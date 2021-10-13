from torch import Tensor
import torch_xla
from typing import List


def adam_step_cpp(params: List[Tensor], grads: List[Tensor],
                  exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor],
                  max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor],
                  amsgrad: bool, beta1: float, beta2: float, lr: float,
                  weight_decay: float, eps: float, found_inf: Tensor,
                  use_adamw: bool):
  r"""Functional API that performs Adam/AdamW algorithm computation
   """

  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]
    max_exp_avg_sq = max_exp_avg_sqs[i]

    torch_xla._XLAC._xla_adam_optimizer_step(found_inf, step, param, grad,
                                             exp_avg, exp_avg_sq,
                                             max_exp_avg_sq, amsgrad, beta1,
                                             beta2, lr, weight_decay, eps,
                                             use_adamw)
