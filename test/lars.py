import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterable, Optional, Callable, Tuple
from torch import nn

"""
    We recommend using create_optimizer_lars and setting bn_bias_separately=True 
    instead of using class Lars directly, which helps LARS skip parameters
    in BatchNormalization and bias, and has better performance in general.
    Polynomial Warmup learning rate decay is also helpful for better performance in general.
"""


def create_optimizer_lars(model, lr, momentum, weight_decay, bn_bias_separately, epsilon, nesterov = False):
    if bn_bias_separately:
        optimizer = Lars([
            dict(params=get_common_parameters(model, exclude_func=get_norm_bias_parameters)),
            dict(params=get_norm_bias_parameters(model), weight_decay=0, lars=False)],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov)
    else:
        optimizer = Lars(model.parameters(),
                         lr=lr,
                         momentum=momentum,
                         weight_decay=weight_decay,
                         epsilon=epsilon,
                         nesterov = nesterov)
    return optimizer


class Lars(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eeta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr=1e-3,
            momentum=0,
            eeta=1e-3,
            weight_decay=0,
            epsilon=0.0,
            nesterov=False
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0 or eeta > 1:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True, nesterov = nesterov)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            lars = group['lars']
            eps = group['epsilon']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                '''
                if lars:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(p.grad)
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    trust_ratio.clamp_(0.0, 50)
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad = decayed_grad.add(p, alpha=weight_decay)
                '''
                decayed_grad = torch.clamp(decayed_grad, -10.0, 10.0)
                # https://github.com/mlcommons/training/blob/ebe9a7e7400f85abce88c4962fa573ff40676870/image_classification/tensorflow2/lars_optimizer.py#L179 
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    mom_t = param_state['momentum_buffer'] = torch.clone(
                        decayed_grad).detach()
                else:
                    mom = param_state['momentum_buffer']
                    mom_t = mom.mul_(momentum).add_(decayed_grad, alpha=-scaled_lr)
                
                if nesterov:
                    p.add_(mom_t, alpha=momentum).add_(decayed_grad, alpha=-scaled_lr)
                else:
                    p.add_(mom_t)

        return loss


"""
    Functions which help to skip bias and BatchNorm
"""
BN_CLS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_norm_parameters(module):
    return get_parameters_from_cls(module, (nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and 'bias' in name:
            yield param


def get_norm_bias_parameters(module):
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters:
            yield param
