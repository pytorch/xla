import torch
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
import torch_xla
from typing import List, Optional


class SGD(Optimizer):
  r"""PT-XLA syncfree SGD optimizer for AMP mode. It takes an additional `found_inf` tensor
    in optimizer.step to indicate whether this optimizer.step should be performed (found_inf == 0)
    or skipped (found_inf > 0).

    Implements stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},\:nesterov\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: nesterov                                                \\
            &\hspace{15mm} g_t \leftarrow g_{t-1} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                    \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

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

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

  def __init__(self,
               params,
               lr=required,
               momentum=0,
               dampening=0,
               weight_decay=0,
               nesterov=False):
    if lr is not required and lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov)
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError(
          "Nesterov momentum requires a momentum and zero dampening")
    super(SGD, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(SGD, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('nesterov', False)

  @torch.no_grad()
  def step(self, closure=None, found_inf: Tensor = None):
    """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            found_inf (torch.Tensor, required): A scalar tensor indicates if
                the optimizer.step should be performed (found_inf == 0) or
                skipped (found_inf != 0).
        """
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
      torch_xla._XLAC._xla_sgd_optimizer_step(found_inf, step, param, d_p, buf,
                                              weight_decay, momentum, lr,
                                              dampening, nesterov)
