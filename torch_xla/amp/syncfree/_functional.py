import torch
from torch import Tensor
import torch_xla
from typing import List, Optional


def adam_step(found_inf: Tensor, state_steps: List[Tensor],
              params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor],
              exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], *,
              amsgrad: bool, beta1: float, beta2: float, lr: float,
              weight_decay: float, eps: float, maximize: bool, use_adamw: bool):
  r"""Functional API that performs PT-XLA sync-free Adam/AdamW algorithm computation
   """

  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]
    max_exp_avg_sq = max_exp_avg_sqs[i]
    torch_xla._XLAC._xla_adam_optimizer_step_(found_inf, step, param, grad,
                                              exp_avg, exp_avg_sq,
                                              max_exp_avg_sq, beta1, beta2, lr,
                                              weight_decay, eps, amsgrad,
                                              maximize, use_adamw)


def sgd_step(found_inf: Tensor, state_steps: List[Tensor], params: List[Tensor],
             d_p_list: List[Tensor],
             momentum_buffer_list: List[Optional[Tensor]], *,
             weight_decay: float, momentum: float, lr: float, dampening: float,
             nesterov: bool, maximize: bool):
  r"""Functional API that performs PT-XLA sync-free SGD algorithm computation.
        """

  for i, param in enumerate(params):
    d_p = d_p_list[i]
    buf = momentum_buffer_list[i]
    step = state_steps[i]
    if buf is None:
      buf = torch.clone(d_p).detach()
      momentum_buffer_list[i] = buf
    torch_xla._XLAC._xla_sgd_optimizer_step_(found_inf, step, param, buf, d_p,
                                             weight_decay, momentum, lr,
                                             dampening, nesterov, maximize)
