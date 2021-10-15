import torch
import torch_xla
from torch import Tensor
from typing import List, Optional


class SGD(torch.optim.SGD):
  r"""PT-XLA variant of SGD optimizer with syncfree support for AMP mode.
    It takes an optional `found_inf` tensor in optimizer.step to indicate whether
    this optimizer.step should be performed (found_inf is 0 or None) or
    skipped (found_inf != 0).

    See torch.optim.SGD for more details.
    https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py

    Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step(found_inf=torch.tensor(1)) # skip the step
        >>> optimizer.step(found_inf=torch.tensor(0)) # perform the step
        >>> optimizer.step() # perform the step
    """

  @torch.no_grad()
  def step(self, closure=None, found_inf: Tensor = None):
    """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            found_inf (torch.Tensor, optional): A scalar tensor indicates if
                the optimizer.step should be performed (found_inf is 0 or None) or
                skipped (found_inf != 0).
        """
    if found_inf is None:
      return super(SGD, self).step(closure=closure)

    if found_inf.shape:
      raise ValueError("The found_inf tensor has to be scalar type")

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      params_with_grad = []
      d_p_list = []
      momentum_buffer_list = []
      state_steps = []
      weight_decay = group['weight_decay']
      momentum = group['momentum']
      dampening = group['dampening']
      nesterov = group['nesterov']
      lr = group['lr']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          d_p_list.append(p.grad)

          state = self.state[p]
          if 'step' not in state:
            state['step'] = torch.zeros_like(found_inf)
          state_steps.append(state['step'])

          if 'momentum_buffer' not in state:
            momentum_buffer_list.append(None)
          else:
            momentum_buffer_list.append(state['momentum_buffer'])

      self.sgd_step(
          found_inf,
          state_steps,
          params_with_grad,
          d_p_list,
          momentum_buffer_list,
          weight_decay=weight_decay,
          momentum=momentum,
          lr=lr,
          dampening=dampening,
          nesterov=nesterov,
      )

      # update momentum_buffers in state
      for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
        state = self.state[p]
        state['momentum_buffer'] = momentum_buffer

    return loss

  def sgd_step(self, found_inf: Tensor, state_steps: List[Tensor],
               params: List[Tensor], d_p_list: List[Tensor],
               momentum_buffer_list: List[Optional[Tensor]], *,
               weight_decay: float, momentum: float, lr: float,
               dampening: float, nesterov: bool):
    r"""Functional API that performs PT-XLA sync-free SGD algorithm computation.
        """

    for i, param in enumerate(params):
      d_p = d_p_list[i]
      buf = momentum_buffer_list[i]
      step = state_steps[i]
      if buf is None:
        buf = torch.clone(d_p).detach()
        momentum_buffer_list[i] = buf
      torch_xla._XLAC._xla_sgd_optimizer_step_(step, param, buf, found_inf, d_p,
                                               weight_decay, momentum, lr,
                                               dampening, nesterov)
