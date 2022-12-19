# This file is largely adapted from ``torch.distributed.fsdp.wrap`` in
# https://github.com/pytorch/pytorch/blob/v1.13.0/torch/distributed/fsdp/wrap.py

from typing import Any, Callable, Set, Tuple, Optional, Type, cast

import torch.nn as nn


def always_wrap_policy(*args, **kwargs) -> bool:
  """
  A simple wrapper policy that always returns ``True``,
  i.e. when passed as the `auto_wrap_policy` into FSDP,
  this will result in all submodules being wrapped as
  distinct FSDP instances.
  """
  return True


def lambda_auto_wrap_policy(module: nn.Module, recurse: bool,
                            unwrapped_params: int, lambda_fn: Callable) -> bool:
  """
  A convenient auto wrap policy to wrap submodules based on an arbitrary user
  function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
  a `wrapper_cls` unit.
  Return if a module should be wrapped during auto wrapping.
  The first three parameters are required by :func:`_recursive_wrap`.
  Args:
     module (nn.Module):
         The module to be considered in this decision.
     recurse (bool):
         Indicate if this is called to make a decision on whether we
         should recurse down a subgraph of the module structure.
         If False, it means this function is called to make a decision
         on whether we should wrap the said module.
     unwrapped_params (int):
         The number of parameters yet to be wrapped in this module.
     lambda_fn (Callable[nn.Module] -> bool):
         If this returns ``True``, this module will be wrapped by
         wrapper_cls individually.
  """
  if recurse:
    # always recurse
    return True
  else:
    # if not recursing, decide whether we should wrap for the leaf node or reminder
    return lambda_fn(module)


def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Set[Type[nn.Module]],
) -> bool:
  """
  A convenient auto wrap policy for transformer models. If the submodule
  is an instance of transformer_layer_cls, the submodule will be wrapped
  as a FSDP unit. Otherwise, all the other remainder submodules are wrapped
  by the outermost FSDP unit.
  Return if a module should be wrapped during FSDP auto wrapping.
  The first three parameters are required by :func:`_recursive_wrap`.
  Args:
     module (nn.Module):
         The module to be considered in this decision.
     recurse (bool):
         Indicate if this is called to make a decision on whether we
         should recurse down a subgraph of the module structure.
         If False, it means this function is called to make a decision
         on whether we should wrap the said module.
     unwrapped_params (int):
         The number of parameters yet to be wrapped in this module.
     transformer_layer_cls (int):
         Submodules with one of the `transformer_layer_cls` names
         will be wrapped as separated FSDP units
  """
  if recurse:
    # always recurse
    return True
  else:
    # if not recursing, decide whether we should wrap for the leaf node or reminder
    return isinstance(module, tuple(transformer_layer_cls))


def size_based_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    # These are customizable for this policy function.
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[nn.Module]]] = None,
) -> bool:
  """A size based auto_wrap_policy function for FSDP API.
     Return if a module should be wrapped during FSDP auto wrapping.
     The first three parameters are used by :func:`_recursive_wrap`.
  Args:
     module (nn.Module):
         The module to be considered in this decision.
     recurse (bool):
         Indicate if this is called to make a decision on whether we
         should recurse down a subgraph of the module structure.
         If False, it means this function is called to make a decision
         on whether we should wrap the said module.
     unwrapped_params (int):
         The number of parameters yet to be wrapped in this module.
     min_num_params (int):
         Customizable policy input. It controls the size threshold
         on how big should a module be to be considered wrapped.
     force_leaf_modules (Set[Type[nn.Module]]): set of module types to
         keep as leaves, i.e., their children will never be wrapped.
     exclude_wrap_modules (Set[Type[nn.Module]]):
         Customizable set of module types to be excluded in wrapping.
  """
  force_leaf_modules = (
      size_based_auto_wrap_policy.FORCE_LEAF_MODULES
      if force_leaf_modules is None else force_leaf_modules)
  exclude_wrap_modules = (
      size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES
      if exclude_wrap_modules is None else exclude_wrap_modules)

  is_large = unwrapped_params >= min_num_params
  if recurse:
    # We should recurse if the module is big enough but not in force_leaf_modules list.
    return is_large and not isinstance(module, tuple(force_leaf_modules))
  else:
    # If we are not recursing, determine if we should wrap.
    return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {
    nn.ModuleList, nn.ModuleDict
}
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}


def _wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
  assert wrapper_cls is not None
  if hasattr(module, '_wrap_overrides'):
    # If module has a _wrap_overrides attribute, we force overriding the
    # FSDP config with these attributes for this module. Currently this
    # is only used to disable mixed precision for BatchNorm when
    # auto_wrapping.
    overrides = {**kwargs, **module._wrap_overrides}
    return wrapper_cls(module, **overrides)

  return wrapper_cls(module, **kwargs)


def recursive_wrap(module: nn.Module,
                   auto_wrap_policy: Callable,
                   wrapper_cls: Callable,
                   ignored_modules: Set[nn.Module],
                   ignored_params: Set[nn.Parameter],
                   only_wrap_children: bool = False,
                   **kwargs: Any) -> Tuple[nn.Module, int]:
  """
  Automatically wrap child modules of *module* that meet the given
  criteria with :func:`auto_wrap`. Does not rely on _ConfigAutoWrap.
  Args:
      module (nn.Module):
          module to recursively wrap
      auto_wrap_policy (Callable):
          A callable specifying a policy to recursively wrap layers with FSDP.
      ignored_modules (Set[torch.nn.Module]): Modules to ignore when
          wrapping.
      ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
          wrapping; these should be the parameters contained in the modules
          in ``ignored_modules``.
  Returns:
      (nn.Module, int):
          Wrapped module and the number parameters wrapped recursively.
  """
  assert auto_wrap_policy is not None, "Must specify auto_wrap_policy."
  assert wrapper_cls is not None, "Must specify wrapper_cls"
  # Make sure no child is already wrapped.
  for _, child in module.named_modules():
    if child in ignored_modules:
      continue
    try:
      assert not isinstance(child, cast(type, wrapper_cls))
    except TypeError:
      # wrapper_cls is a function as opposed to a class type, just bypass above check.
      pass

  # We count all params, assuming none of them are already wrapped.
  num_params = sum(
      p.numel() for p in module.parameters() if p not in ignored_params)

  if auto_wrap_policy(module=module, recurse=True, unwrapped_params=num_params):
    total_wrapped_params = 0
    # Iterate through the children, recursively wrap if necessary
    for name, child in module.named_children():
      if child in ignored_modules:
        continue
      wrapped_child, num_wrapped_params = recursive_wrap(
          module=child,
          auto_wrap_policy=auto_wrap_policy,
          wrapper_cls=wrapper_cls,
          ignored_modules=ignored_modules,
          ignored_params=ignored_params,
          **kwargs,
      )
      setattr(module, name, wrapped_child)
      # Keep track of how many parameters have been wrapped
      total_wrapped_params += num_wrapped_params
    # decide if we need to wrap the current module,
    # since the left over parameters exceed the number of params to wrap
    remainder = num_params - total_wrapped_params
    if not only_wrap_children and auto_wrap_policy(
        module=module, recurse=False, unwrapped_params=remainder):
      # Leaf node or final wrapping of the remainder both happen here.
      return _wrap(module, wrapper_cls, **kwargs), num_params
    else:
      return module, total_wrapped_params
  return module, 0
