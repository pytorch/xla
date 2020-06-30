import inspect
import torch
import torch.nn as nn


def _patch(fn, newfn):
  xfingerprint = inspect.signature(fn)
  fingerprint = inspect.signature(newfn)
  if xfingerprint != fingerprint:
    raise RuntimeError(
        'Unable to patch {}, signature mismatch: {} vs {}'.format(
            fn, xfingerprint, fingerprint))
  newfn._orig = fn
  return newfn


def clip_grad_norm_(parameters, max_norm, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  max_norm = float(max_norm)
  norm_type = float(norm_type)
  if len(parameters) == 0:
    return torch.tensor(0.)
  device = parameters[0].grad.device
  if norm_type == inf:
    total_norm = max(p.grad.detach().abs().max() for p in parameters)
  else:
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type)
  clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
  clip_value = torch.where(clip_coef < 1, clip_coef,
                           torch.tensor(1., device=device))
  for p in parameters:
    p.grad.detach().mul_(clip_value)
  return total_norm


def _apply_patches():
  nn.utils.clip_grad_norm_ = _patch(nn.utils.clip_grad_norm_, clip_grad_norm_)
