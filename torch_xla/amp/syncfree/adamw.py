import torch
from torch import Tensor
from . import _functional as F


class AdamW(torch.optim.AdamW):
  r"""PT-XLA variant of AdamW optimizer with syncfree support for AMP mode.
    It takes an optional `found_inf` tensor in optimizer.step to indicate whether
    this optimizer.step should be performed (found_inf is 0 or None) or
    skipped (found_inf != 0).

    See torch.optim.AdamW for more details.
    https://github.com/pytorch/pytorch/blob/master/torch/optim/adamw.py

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
  """

  @torch.no_grad()
  def step(self, closure=None, found_inf: Tensor = None):
    """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            found_inf (torch.Tensor, optional): A scalar tensor indicates if
                the optimizer.step should be performed (found_inf is 0 or None) or
                skipped (found_inf == 1).
        """
    if found_inf is None:
      return super(AdamW, self).step(closure=closure)

    if found_inf.shape:
      raise ValueError("The found_inf tensor has to be scalar type")

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('AdamW does not support sparse gradients')
          grads.append(p.grad)

          state = self.state[p]

          # Lazy state initialization
          if not state:
            state['step'] = torch.zeros_like(found_inf)
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])
          max_exp_avg_sqs.append(state['max_exp_avg_sq'])

          state_steps.append(state['step'])

      F.adam_step(
          found_inf,
          state_steps,
          params_with_grad,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          amsgrad=group['amsgrad'],
          beta1=beta1,
          beta2=beta2,
          lr=group['lr'],
          weight_decay=group['weight_decay'],
          eps=group['eps'],
          maximize=group['maximize'],
          use_adamw=True)

    return loss
