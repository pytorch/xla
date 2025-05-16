import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple, Optional, List, Union
import warnings


@dataclass(frozen=True)
class GradientAccumulationContext:
  """Context for the gradient accumulation instructions.
  Attributes:
    * num_gradient_steps: Number of steps to accumulate gradients over
    * num_iterable_tensors: Number of input tensors to iterate over
    * num_carried_tensors: Number of tensors carried between iterations
  """
  num_gradient_steps: int
  num_iterable_tensors: int
  num_carried_tensors: int


def gradient_accumulation(
    train_step: Callable[..., Any], iterable_tensors: Tuple[torch.Tensor],
    model: torch.nn.Module, *carried_tensors: torch.Tensor
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
  """Accumulates gradients over multiple training steps using XLA's `While`
      operator to iterate over the leading dimension of the iterable tensors.
      The backward computation of the model is implicitly executed following the
      train_step operations.

  Notes:

  * The model tracing will happen entirely within the loop. Hence, it is
    assumed that `train_step` is purposefully encapsulated inside of the
    loop. Hence, it is not recommended to have any operation involving the
    model parameters outside of `train_step`.
  * Note that all the inputs are expected to be materialized. In case these are
    not materialized, they will be synced early on.
  * We expect the train step to be pure. In case it is not, for instance,
    containing in-place mutations for the body fn inputs, then we error out.

  Args:
    train_step: Training function that takes iterable tensors and carried
          tensors, and returns either a loss tensor or a tuple of (loss,
          *carried_outputs). The iterable tensor inputs to this function should
          disregard the leading dimension.

    iterable_tensors: Input tensors to iterate over. All tensors must have the
          same first dimension size which determines number of iterations. The
          underlying loop in the gradient accumulation will iterate through the
          leading dimension of these tensors.

    model: PyTorch model whose parameters will be updated. Note that the entire
            model computation will be traced and generated from within the loop.

    carried_tensors: Unpacked tensor arguments that are updated between iterations.

  Returns:
    accumulated_loss, *carried_tensors: A tuple including the `accumulated_loss`
    and the unpacked `carried_tensors` that were provided as inputs. In case
    there are no carried tensors, then only the `accumulated_loss` is returned.
    In addition, the model parameter gradients, if applicable, contain the
    accumulated gradients.

  Example:

    >>> # Note: This is a partial example, since it is dependent on the
    >>> # training model. Please refer to existing tests.
    >>>
    >>> from torch_xla.experimental.gradient_accumulation import (
    >>>    gradient_accumulation
    >>> )
    >>>
    >>> def train_step(input, label, other_tensor):
    >>>   output = model(input_id)
    >>>   loss = loss_fn(output, label)
    >>>   updated_other_tensor = other_tensor + 10
    >>>   return loss, updated_other_tensor
    >>>
    >>> some_tensor = torch.tensor(10).to(device)
    >>> for (data, target) in loader:
    >>>   # Assuming data's and target's first iterable dimension is 5.
    >>>   # >> data.shape = [5, 128, 16834]
    >>>   # >> label.shape = [5, 128]
    >>>   running_loss, some_tensor = gradient_accumulation(
    >>>       train_step,
    >>>       (data, target),
    >>>       model,
    >>>       some_tensor
    >>>   )
    >>>   print(some_tensor)  # Should be 60
    >>>   print(running_loss)  # Should be the accumulated loss across all 5
    >>>                        # iteration steps
    >>>   optimizer.step()  # Should update all weights with the accumulated
    >>>                     # parameter weights
  """
  # Validate that the arguments minimally suffice our requirements
  if not iterable_tensors:
    raise ValueError("iterable_tensors cannot be empty")

  accumulation_steps = iterable_tensors[0].size(0)
  for i, tensor in enumerate(iterable_tensors):
    if not isinstance(tensor, torch.Tensor):
      raise ValueError(f"Element {i} of iterable_tensors is not a tensor")
    if tensor.numel() == 0:
      raise ValueError(f"Element {i} of iterable_tensors is empty")
    if tensor.size(0) != accumulation_steps:
      raise ValueError(
          f"Element {i} of iterable_tensors has inconsistent first dimension")
  return _gradient_accumulation(accumulation_steps, train_step,
                                iterable_tensors, model, carried_tensors)


class XlaBuildHelper:
  """Helper class for tracking the parameters for the XLA while computations."""

  def __init__(self, name: str):
    self._builder = xb.create_builder(name)
    self._params: List[xb.Op] = []
    self._param_tensors: List[torch.Tensor] = []

  def add_param(self, val: torch.Tensor, idx: Optional[int] = None) -> int:
    if idx is None:
      idx = len(self._params)
    param = xb.mkparam(self._builder, idx, xb.tensor_shape(val))
    self._params.append(param)
    self._param_tensors.append(val)
    return idx

  @property
  def params(self) -> Tuple[xb.Op, ...]:
    return tuple(self._params)

  @property
  def param_tensors(self) -> Tuple[torch.Tensor, ...]:
    return tuple(self._param_tensors)

  @property
  def num_params(self) -> int:
    return len(self._params)


def _make_init_grad(param):
  grad = torch.zeros_like(param, device=param.device, requires_grad=False)
  param_sharding = torch_xla._XLAC._get_xla_op_sharding(param)
  if param_sharding:
    # Match the gradient sharding to the parameter sharding, if present.
    torch_xla._XLAC._xla_mark_sharding(grad, param_sharding)
  return grad


def _gradient_accumulation_impl(context, body_fn, iterable_tensors, params,
                                carried_tensors):
  builder = XlaBuildHelper('grad_acc')
  device = torch_xla.device()

  init_iterator = torch.tensor(0, dtype=torch.int32, device=device)
  init_loss = torch.tensor(0, dtype=torch.float32, device=device)
  init_grads = [_make_init_grad(p) for p in params if p.requires_grad]

  builder.add_param(init_iterator)
  builder.add_param(init_loss)
  for grad in init_grads:
    builder.add_param(grad)

  iterable_tensor_slices = tuple(t[0] for t in iterable_tensors)
  body_fn_inputs = [*iterable_tensor_slices, *params, *carried_tensors]

  # Ensure newly initialized inputs are fully materialized for the purpose of tracing.
  # In addition, we sync the inputs that are expected to be materialized, namely all
  # the model parameter, gradients (if present), iterable_tensors and carried_tensors.
  # TODO(rpsilva): Use placeholder tensors for the iterable tensor slices, to avoid
  # a redundant sync.
  torch_xla._XLAC._xla_sync_multi(
      body_fn_inputs, devices=[], wait=False, sync_xla_data=False)

  tid_to_body_fn_inputs = {
      torch_xla._XLAC._xla_get_tensor_id(t): (idx, t)
      for idx, t in enumerate(body_fn_inputs)
  }

  for idx, (t, s) in enumerate(zip(iterable_tensors, iterable_tensor_slices)):
    # Map each iterable tensor slice back to the full tensor
    tid_to_body_fn_inputs[torch_xla._XLAC._xla_get_tensor_id(s)] = (idx, t)

  body_result = body_fn(iterable_tensor_slices, tuple(params), carried_tensors)
  body_result = list(body_result)

  # Ensure that all the body fn inputs are not mutated in-place, as we need to
  # guarantee that the body function is pure.
  assert not any(
      torch_xla._XLAC._check_tensor_need_materialization(body_fn_inputs))

  # Ensure that any prior async operations on the device has terminated.
  xm.wait_device_ops()

  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node(body_result,
                                                        body_fn_inputs)
  body_hlo = torch_xla._XLAC._get_xla_tensors_hlo_proto(body_result)
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  del body_result

  # Capture the seed info ID, to identify all the seed tensors.
  seed_info_id = torch_xla._XLAC._get_seed_info_id()
  seed_info_indices = set()
  # Maps the graph index to body_fn input/output index. Note that the list
  # indices implicitly denote the position for the body_fn.
  graph_to_body_idx = []
  # Collect all graph inputs if they match the provided body_fn_inputs. In case
  # of hoisted tensors, we simply capture the XLA values. Note that for hoisted
  # seed tensors, we fork a unique seed tensor for all iterations, to guarantee
  # that the values differ across loop iterations.
  for tid, xla_value in zip(graph_input_tensor_ids, graph_input_xla_values):
    idx, t = tid_to_body_fn_inputs.get(tid, (None, xla_value))
    if tid == seed_info_id:
      seed_info_indices.add(len(graph_to_body_idx))
      t = torch.randint(
          0, (1 << 63) - 1, (context.num_gradient_steps,), device=device)
    builder.add_param(t)
    graph_to_body_idx.append(idx)

  carried_tensors_start_idx = len(body_fn_inputs) - context.num_carried_tensors

  def _body_fn_wrapper(curr_iter: xb.Op, loss: xb.Op, *args:
                       xb.Op) -> Tuple[xb.Op, ...]:

    def dynamic_slice(xs: xb.Op, idx: xb.Op) -> xb.Op:
      indices = [idx] + [idx.zeros_like() for _ in range(xs.shape().rank - 1)]
      slice_shape = list(xs.shape().sizes)
      slice_shape[0] = 1
      sliced = xs.dynamic_slice(indices, slice_shape)
      return sliced.reshape(list(xs.shape().sizes)[1:])

    grads = list(args[:len(init_grads)])
    graph_inputs = list(args[len(grads):])
    graph_outputs = graph_inputs[:]

    # Dynamic slice on iterable and seed tensors.
    for graph_idx, body_idx in enumerate(graph_to_body_idx):
      if graph_idx in seed_info_indices:
        # Substitute the seed tensor that is fed to the user computation.
        graph_inputs[graph_idx] = dynamic_slice(graph_inputs[graph_idx],
                                                curr_iter)
      if body_idx is None:
        # Ignore hoisted variables
        continue
      if body_idx < context.num_iterable_tensors:
        graph_inputs[graph_idx] = dynamic_slice(graph_inputs[graph_idx],
                                                curr_iter)

    result = xb.Op.call(body_computation, graph_inputs)

    # Accumulate loss and grads
    loss = loss + result.get_tuple_element(0)
    for i, grad in enumerate(grads):
      grads[i] = grad + result.get_tuple_element(i + 1)

    # Update carried tensors in graph_inputs for next iteration
    for graph_idx, body_idx in enumerate(graph_to_body_idx):
      if body_idx is not None and body_idx >= carried_tensors_start_idx:
        idx = body_idx - carried_tensors_start_idx
        graph_outputs[graph_idx] = result.get_tuple_element(1 + len(grads) +
                                                            idx)

    one = xb.Op.scalar(curr_iter.builder(), 1, dtype=xb.Type.S32)
    return (curr_iter + one, loss, *grads, *graph_outputs)

  def _cond_fn(curr_iter: xb.Op, *args: xb.Op) -> bool:
    return curr_iter < xb.Op.scalar(
        curr_iter.builder(), context.num_gradient_steps, dtype=xb.Type.S32)

  w = xb.Op.mkwhile(builder.params, _cond_fn, _body_fn_wrapper)
  computation = w.build('grad_acc_loop_torch_func')
  _, loss, *outputs = torch_xla._XLAC._xla_user_computation(
      'xla::_op_grad_acc_loop', builder.param_tensors, computation)
  grads = outputs[:len(init_grads)]
  graph_outputs = outputs[len(grads):]
  carried_tensors = [None] * context.num_carried_tensors
  for graph_idx, body_idx in enumerate(graph_to_body_idx):
    if body_idx is not None and body_idx >= carried_tensors_start_idx:
      idx = carried_tensors_start_idx - body_idx
      carried_tensors[idx] = graph_outputs[graph_idx]
  return (loss, grads, carried_tensors)


def _gradient_accumulation(accumulation_steps, train_step, iterable_tensors,
                           model, carried_tensors):
  model_parameters = list(model.parameters())
  context = GradientAccumulationContext(accumulation_steps,
                                        len(iterable_tensors),
                                        len(carried_tensors))

  def body_fn(
      iterable_tensors: Tuple[torch.Tensor, ...], params: Tuple[torch.Tensor,
                                                                ...],
      carried_tensors: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    # We set the grads to None, and return the computed gradients within this body
    # iteration, accumulating it in the body wrapper. Note that body_fn needs to be
    # pure, without side effects and tensor mutations.
    orig_grads = [param.grad for param in params]
    for param in params:
      param.grad = None

    result = train_step(*iterable_tensors, *carried_tensors)

    if not context.num_carried_tensors:
      loss = result
    else:
      loss, *carried_tensors = result
    loss /= context.num_gradient_steps
    loss.backward()

    grads = [param.grad for param in params if param.requires_grad]
    # Restore original grads to make this function pure.
    for param, grad in zip(params, orig_grads):
      param.grad = grad

    return (loss, *grads, *carried_tensors)

  loss, grads, carried_tensors = _gradient_accumulation_impl(
      context, body_fn, iterable_tensors, model_parameters, carried_tensors)

  params_with_grad = [
      param for param in model_parameters if param.requires_grad
  ]
  # Accumulate the resulting gradients.
  for param, grad in zip(params_with_grad, grads):
    if param.grad is None:
      param.grad = grad
    else:
      param.grad.add_(grad)

  if not carried_tensors:
    return loss

  return loss, *carried_tensors
