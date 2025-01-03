# This file is copied from https://github.com/pytorch/pytorch/blob/master/torch/utils/checkpoint.py.
# PyTorch/XLA needs to add `optimization_barrier` before saving the input for the backward hence we
# slightly modify the upstream version of the checkpoint util function.
import inspect
import torch
import warnings
import torch_xla.core.xla_model as xm
from torch.utils.checkpoint import detach_variable, check_backward_validity, _get_device_module, _infer_device_type
from typing import Iterable, List, Tuple, Union

# The 2 functions below (get_device_states and set_device_states) are slightly modified versions
# from PyTorch's original file.
#
# They are mostly implemented the same, while accounting for XLA device. In summary, the problem was
# twofold:
#
#     1. No XLA device support.
#
#     2. The XLA device module (torch_xla.core.xla_model) has a slightly different API: its state
#        representation is an int, instead of a tensor.


def get_device_states(
    *args) -> Tuple[List[torch.device], List[Union[torch.Tensor, int]]]:
  fwd_device = list({
      arg.device
      for arg in args
      if isinstance(arg, torch.Tensor) and not arg.device.type == "cpu"
  })

  fwd_device_states = []

  device_type = _infer_device_type(*args)
  assert device_type is not None, "multiple non-CPU devices not supported"

  device_module = xm if device_type == "xla" else _get_device_module(
      device_type)

  for device in fwd_device:
    fwd_device_states.append(device_module.get_rng_state(device))

  return fwd_device, fwd_device_states


def set_device_states(devices: List[torch.device],
                      states: List[Union[torch.Tensor, int]]) -> None:
  if len(states) == 0:
    return

  # get_device_states guarantees that there's only one device-type.
  # Therefore, their states should also be of either Tensor (cuda) or int (xla) type.

  state_0_type = type(states[0])
  assert all(isinstance(v, state_0_type)
             for v in states), f"all device states should have the same type"

  device_module = xm if state_0_type == int else get_device_module(*states)
  for device, state in zip(devices, states):
    device_module.set_rng_state(state, device=device)


class CheckpointStatus:

  def __init__(self):
    self.in_chkpt_bwd = False


# chkpt_status is used for FSDP allgather reduction optimizaion
chkpt_status = CheckpointStatus()


class CheckpointFunction(torch.autograd.Function):

  def _extract_tensors_from_list(inputs):
    tensor_inputs = []
    if torch.is_tensor(inputs):
      tensor_inputs.append(inputs)
    # tensor is Iterable so we need to avoid iterating through tensor
    elif isinstance(inputs, Iterable):
      for input in inputs:
        if torch.is_tensor(input):
          tensor_inputs.append(input)
    return tensor_inputs

  @staticmethod
  def forward(ctx, run_function, preserve_rng_state, *args):
    check_backward_validity(args)
    ctx.run_function = run_function
    ctx.preserve_rng_state = preserve_rng_state

    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    ctx.gpu_autocast_kwargs = {
        "device_type": "cuda",
        "enabled": torch.is_autocast_enabled("cuda"),
        "dtype": torch.get_autocast_dtype("cuda"),
        "cache_enabled": torch.is_autocast_cache_enabled()
    }
    ctx.cpu_autocast_kwargs = {
        "device_type": "cpu",
        "enabled": torch.is_autocast_enabled("cpu"),
        "dtype": torch.get_autocast_dtype("cpu"),
        "cache_enabled": torch.is_autocast_cache_enabled()
    }
    ctx.xla_autocast_kwargs = {
        "device_type": "xla",
        "enabled": torch.is_autocast_enabled("xla"),
        "dtype": torch.get_autocast_dtype("xla"),
        "cache_enabled": torch.is_autocast_cache_enabled()
    }
    if preserve_rng_state:
      ctx.fwd_xla_state = xm.get_rng_state()
      ctx.fwd_cpu_state = torch.get_rng_state()
      # Don't eagerly initialize the cuda context by accident.
      # (If the user intends that the context is initialized later, within their
      # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
      # we have no way to anticipate this will happen before we run the function.)
      ctx.had_cuda_in_fwd = False
      if torch.cuda._initialized:
        ctx.had_cuda_in_fwd = True
        ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

    # Save non-tensor inputs in ctx, keep a placeholder None for tensors
    # to be filled out during the backward.
    ctx.inputs = []
    ctx.tensor_indices = []
    tensor_inputs = []
    tensor_outputs = []
    for i, arg in enumerate(args):
      if torch.is_tensor(arg):
        tensor_inputs.append(arg)
        ctx.tensor_indices.append(i)
        ctx.inputs.append(None)
      else:
        ctx.inputs.append(arg)

    ctx.save_for_backward(*tensor_inputs)

    with torch.no_grad():
      outputs = run_function(*args)

    return outputs

  @staticmethod
  def backward(ctx, *args):
    chkpt_status.in_chkpt_bwd = True

    if not torch.autograd._is_checkpoint_valid():
      raise RuntimeError(
          "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
          " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
          " argument.")
    # Copy the list to avoid modifying original list.
    inputs = list(ctx.inputs)
    tensor_indices = ctx.tensor_indices
    tensors = ctx.saved_tensors

    # Fill in inputs with appropriate saved tensors.
    for i, idx in enumerate(tensor_indices):
      inputs[idx] = tensors[i]

    # Stash the surrounding rng state, and mimic the state that was
    # present at this time during forward.  Restore the surrounding state
    # when we're done.
    rng_devices = []
    if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
      rng_devices = ctx.fwd_gpu_devices

    # optimization_barrier_ is needed to separate the original forward pass with
    # the next forward + backward pass.
    weights = []
    buffers = []
    if inspect.ismethod(ctx.run_function) and isinstance(
        ctx.run_function.__self__, torch.nn.Module):
      weights = list(ctx.run_function.__self__.parameters())
      buffers = list(ctx.run_function.__self__.buffers())
    xm.optimization_barrier_(
        CheckpointFunction._extract_tensors_from_list(inputs + list(args) +
                                                      weights + buffers))

    # torch.random.fork_rng will handle the cpu and gpu seed
    # xm.fork_rng will handle the xla device seed
    with torch.random.fork_rng(
        devices=rng_devices, enabled=ctx.preserve_rng_state):
      with xm.fork_rng():
        if ctx.preserve_rng_state:
          xm.set_rng_state(ctx.fwd_xla_state)
          torch.set_rng_state(ctx.fwd_cpu_state)
          if ctx.had_cuda_in_fwd:
            set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
        detached_inputs = detach_variable(tuple(inputs))
        with torch.enable_grad(), \
            torch.autocast(**ctx.gpu_autocast_kwargs), \
            torch.autocast(**ctx.cpu_autocast_kwargs), \
            torch.autocast(**ctx.xla_autocast_kwargs):
          outputs = ctx.run_function(*detached_inputs)

    if isinstance(outputs, torch.Tensor):
      outputs = (outputs,)

    # run backward() with only tensor that requires grad
    outputs_with_grad = []
    args_with_grad = []
    for i in range(len(outputs)):
      if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
        outputs_with_grad.append(outputs[i])
        args_with_grad.append(args[i])
    if len(outputs_with_grad) == 0:
      raise RuntimeError("none of output has requires_grad=True,"
                         " this checkpoint() is not necessary")
    torch.autograd.backward(outputs_with_grad, args_with_grad)
    grads = tuple(
        inp.grad if isinstance(inp, torch.Tensor) else None
        for inp in detached_inputs)

    chkpt_status.in_chkpt_bwd = False
    return (None, None) + grads


def checkpoint(function, *args, use_reentrant: bool = True, **kwargs):
  r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    The output of :attr:`function` can contain non-Tensor values and gradient
    recording is only performed for the Tensor values. Note that if the output
    consists of nested structures (ex: custom objects, lists, dicts etc.)
    consisting of Tensors, these Tensors nested in custom structures will not
    be considered as part of autograd.


    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning::
        If ``use_reentrant=True`` is specified, then if the checkpointed segment
        contains tensors detached from the computational graph by `detach()` or
        `torch.no_grad()`, the backward pass will raise an error. This is
        because `checkpoint` makes all the outputs require gradients which
        causes issues when a tensor is defined to have no gradient in the model.
        To circumvent this, detach the tensors outside of the `checkpoint`
        function. Note that the checkpointed segment can contain tensors
        detached from the computational graph if ``use_reentrant=False`` is
        specified.

    .. warning::
        If ``use_reentrant=True`` is specified, at least one of the inputs needs
        to have :code:`requires_grad=True` if grads are needed for model inputs,
        otherwise the checkpointed part of the model won't have gradients. At
        least one of the outputs needs to have :code:`requires_grad=True` as
        well. Note that this does not apply if ``use_reentrant=False`` is
        specified.

    .. warning::
        If ``use_reentrant=True`` is specified, checkpointing currently only
        supports :func:`torch.autograd.backward` and only if its `inputs`
        argument is not passed. :func:`torch.autograd.grad`
        is not supported. If ``use_reentrant=False`` is specified, checkpointing
        will work with :func:`torch.autograd.grad`.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
        use_reentrant(bool, optional, default=True): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad``. Note that future
            versions of PyTorch will default to ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
  # Hack to mix *args with **kwargs in a python 2.7-compliant way
  preserve = kwargs.pop('preserve_rng_state', True)
  if kwargs:
    raise ValueError("Unexpected keyword arguments: " +
                     ",".join(arg for arg in kwargs))

  if use_reentrant:
    return CheckpointFunction.apply(function, preserve, *args)
  else:
    raise ValueError("XLA currently does not support use_reentrant==False")
