import torch
import torch_xla
import torch_xla.core.xla_builder as xb

from typing import Any, Callable, Sequence, Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class GradientAccumulationContext:
  """Context for the gradient accumulation instructions.
  Attributes:
    * num_gradient_steps: Number of steps to accumulate gradients over
    * num_iterable_tensors: Number of input tensors to iterate over
    * num_carried_tensors: Number of tensors carried between iterations
    * num_model_params: Number of model parameters
    * num_internal_tensors: Number of internal tensors used (default: 2)

  Note: `num_internal_tensors` should only be changed if we create new internal
  tensors.
  """
  num_gradient_steps: int
  num_iterable_tensors: int
  num_carried_tensors: int
  num_model_params: int
  num_internal_tensors: int = 2


def gradient_accumulation(
    train_step: Callable[..., Any],
    iterable_tensors: Sequence[torch.Tensor],
    model: torch.nn.Module,
    carried_tensors: Optional[Tuple[torch.Tensor, ...]] = None
) -> Tuple[torch.Tensor, ...]:
  """Accumulates gradients over multiple training steps using XLA's `While`
     operator to iterate over the leading dimension of the iterable tensors.
     The backward computation of the model is implicitly executed following the
     train_step operations.

  Notes:

    The model tracing will happen entirely within the loop. Hence, it is
    assumed that `train_step` is purposefully encapsulated inside of the
    loop. Hence, it is not recommended to have any operation involving the
    model parameters outside of `train_step`.

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

    carried_tensors: Optional tensors passed and updated between iterations.

  Returns:
    (accumulated_loss, carried_tensor0, carried_tensor1, ...): A tuple including
    the `accumulated_loss` and the same unpacked `carried_tensors` that were
    provided as inputs. In addition, the model parameter gradients, if
    applicable, contain the accumulated gradients.

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
    >>>   updated_other_tensor += 10
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
    >>>       (some_tensor,)
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
  carried_tensors = carried_tensors or tuple()
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


def _gradient_accumulation_impl(context, body_fn, iterable_tensors, params,
                                grads, carried_tensors):
  builder = XlaBuildHelper('grad_acc')
  device = torch_xla.device()

  def _prepare_fake_tensors(
      iterable_tensors: Sequence[torch.Tensor],
      carried_tensors: Sequence[torch.Tensor]
  ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    fake_iterable_tensors = []
    for iter_tensor in iterable_tensors:
      original_size = iter_tensor.size()
      fake_iterable_tensors.append(
          torch.empty(original_size[1:],
                      dtype=iter_tensor.dtype).to(iter_tensor.device))

    fake_carried_tensors = []
    for carried_input in carried_tensors:
      fake_carried_tensors.append(
          torch.empty(carried_input.size(), dtype=carried_input.dtype).to(
              carried_input.device).requires_grad_(carried_input.requires_grad))
    return fake_iterable_tensors, fake_carried_tensors

  # TODO - Fake the model once we are able to create placeholder tensors.
  fake_iterable_tensors, fake_carried_tensors = _prepare_fake_tensors(
      iterable_tensors, carried_tensors)
  init_iterator = torch.tensor(0, dtype=torch.int32, device=device)
  init_loss = torch.tensor(0, dtype=torch.float32, device=device)

  body_fn_inputs = (init_iterator, init_loss, *fake_iterable_tensors,
                    *fake_carried_tensors, *params, *grads)
  body_result = body_fn(init_iterator, init_loss, tuple(fake_iterable_tensors),
                        tuple(fake_carried_tensors), tuple(params),
                        tuple(grads))

  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node(
      list(body_result) + list(body_fn_inputs))

  body_fn_input_tensor_ids = [
      torch_xla._XLAC._xla_get_tensor_id(i) for i in body_fn_inputs
  ]
  uncaptured_input_tensor_ids = tuple(
      v for i, v in zip(graph_input_tensor_ids, graph_input_xla_values)
      if i not in body_fn_input_tensor_ids)

  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  body_ctx.build(body_result + uncaptured_input_tensor_ids)
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)

  builder.add_param(init_iterator)
  builder.add_param(init_loss)

  def _build_parameter_mapping(
      builder: XlaBuildHelper,
      context: GradientAccumulationContext,
      body_fn_inputs: Tuple[torch.Tensor, ...],
      uncaptured_input_tensor_ids: Tuple[torch.Tensor, ...],
      iterable_tensors: Sequence[torch.Tensor],
      fake_iterable_tensors: Sequence[torch.Tensor],
      carried_tensors: Tuple[torch.Tensor, ...],
      fake_carried_tensors: Tuple[torch.Tensor, ...],
      params: List[torch.Tensor],
      grads: List[torch.Tensor],
  ) -> Dict[int, int]:
    param_mapping = {}

    def add_to_mapping(val: torch.Tensor,
                       fake_val: Optional[torch.Tensor] = None):
      idx = builder.add_param(val)
      param_id = body_ctx.tensor_parameter_id(
          fake_val if fake_val is not None else val)
      if param_id != -1:
        param_mapping[param_id] = idx

    # Process iterable tensors and carried inputs
    for val, fake_val in zip(iterable_tensors, fake_iterable_tensors):
      add_to_mapping(val, fake_val)
    for val, fake_val in zip(carried_tensors, fake_carried_tensors):
      add_to_mapping(val, fake_val)

    # Process params, grads, and uncaptured input tensor ids
    for tensor_list in (params, grads, uncaptured_input_tensor_ids):
      for val in tensor_list:
        add_to_mapping(val)

    # Handle any additional hoisted variables
    hoisted_vars = body_ctx.device_parameter_id_tensor_mapping()
    for v in body_fn_inputs + uncaptured_input_tensor_ids:
      param_id = body_ctx.tensor_parameter_id(v)
      hoisted_vars.pop(param_id, None)

    # TODO(rpsilva-aws): Derived from `experimental/scan.py`. Unify the RNG and
    # hoisted paths.
    seed_info_id = torch_xla._XLAC._get_seed_info_id()
    seed_parameter_id = None
    if seed_info_id in graph_input_tensor_ids:
      seed_idx = graph_input_tensor_ids.index(seed_info_id)
      seed_parameter_id = body_ctx.tensor_parameter_id(
          graph_input_xla_values[seed_idx])
      assert seed_parameter_id != -1, "`fn` uses random seed, but random seed is not \
        a parameter to the traced HLO graph"

      # Replace the single seed value with a tensor of seeds, one per iteration.
      seed_tensor = hoisted_vars[seed_parameter_id]
      assert seed_tensor.dtype == torch.int64
      hoisted_vars[seed_parameter_id] = torch.randint(
          0,
          2**62, (context.num_gradient_steps,),
          dtype=torch.int64,
          device=device)

    for param_id, tensor in hoisted_vars.items():
      idx = builder.add_param(tensor)
      param_mapping[param_id] = idx
    return param_mapping, seed_parameter_id

  param_mapping, seed_parameter_id = _build_parameter_mapping(
      builder, context, body_fn_inputs, uncaptured_input_tensor_ids,
      iterable_tensors, fake_iterable_tensors, carried_tensors,
      fake_carried_tensors, params, grads)

  def _body_fn_wrapper(curr_iter: xb.Op, curr_loss: xb.Op,
                       *while_params: xb.Op):

    def dynamic_slice(xs: xb.Op, idx: xb.Op) -> xb.Op:
      indices = [idx] + [idx.zeros_like() for _ in range(xs.shape().rank - 1)]
      slice_shape = list(xs.shape().sizes)
      slice_shape[0] = 1
      sliced = xs.dynamic_slice(indices, slice_shape)
      return sliced.reshape(list(xs.shape().sizes)[1:])

    # TODO(rpsilva-aws): Derived from `experimental/scan.py`. Unify the RNG
    # path.
    def replace_rng_seed(curr_iter: xb.Op, *while_params: xb.Op):
      """Slices the pre-generated seed tensor for the current iteration."""
      if seed_parameter_id is None:
        return while_params
      idx = param_mapping[seed_parameter_id]
      replaced = list(while_params)
      replaced[idx] = dynamic_slice(replaced[idx], curr_iter)
      return replaced

    def call_fn_computation(*while_params: xb.Op) -> xb.Op:
      fn_inputs = [
          while_params[param_mapping[i]] for i in range(len(param_mapping))
      ]
      return xb.Op.call(body_computation, fn_inputs)

    iterable_tensors = while_params[:context.num_iterable_tensors]
    idx = curr_iter
    sliced_iterables = [
        dynamic_slice(iter_tensor, idx) for iter_tensor in iterable_tensors
    ]

    # Call the computation with current values
    result = call_fn_computation(
        idx, curr_loss,
        *replace_rng_seed(idx, *sliced_iterables,
                          *while_params[context.num_iterable_tensors:]))

    # Extract the carried tensors and accumulated gradients.
    carried_tensors_and_gradients = [
        result.get_tuple_element(i) for i in range(
            context.num_internal_tensors + context.num_iterable_tensors,
            result.shape().tuple_size())
    ]
    one = xb.Op.scalar(idx.builder(), 1, dtype=xb.Type.S32)
    updated_loss = curr_loss + result.get_tuple_element(1)
    return (curr_iter + one, updated_loss, *iterable_tensors,
            *carried_tensors_and_gradients)

  def _cond_fn(curr_iter: xb.Op, *rest):
    return curr_iter < xb.Op.scalar(
        curr_iter.builder(), context.num_gradient_steps, dtype=xb.Type.S32)

  def _compute_output_indices(
      context: GradientAccumulationContext) -> List[int]:
    # Start with loss index
    indices = [1]
    # Add indices for carried tensors
    carried_start = context.num_internal_tensors + context.num_iterable_tensors
    carried_end = carried_start + context.num_carried_tensors
    indices.extend(range(carried_start, carried_end))
    # Add indices for accumulated gradients
    grad_start = carried_end + context.num_model_params
    grad_end = grad_start + context.num_model_params
    indices.extend(range(grad_start, grad_end))
    return indices

  w = xb.Op.mkwhile(builder.params, _cond_fn, _body_fn_wrapper)
  outputs = [w.get_tuple_element(i) for i in _compute_output_indices(context)]
  op = xb.Op.tuple(outputs)
  computation = op.build('grad_acc_loop_torch_func')
  result = torch_xla._XLAC._xla_user_computation('xla::_op_grad_acc_loop',
                                                 builder.param_tensors,
                                                 computation)
  return result


def _gradient_accumulation(accumulation_steps, train_step, iterable_tensors,
                           model, carried_tensors):
  model_parameters = list(model.parameters())
  context = GradientAccumulationContext(accumulation_steps,
                                        len(iterable_tensors),
                                        len(carried_tensors),
                                        len(model_parameters))

  def body_fn(iteri: torch.Tensor, _: torch.Tensor,
              iterable_tensors: Tuple[torch.Tensor, ...],
              carried_tensors: Tuple[torch.Tensor,
                                     ...], params: Tuple[torch.Tensor, ...],
              grads: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    result = train_step(*iterable_tensors, *carried_tensors)

    if not context.num_carried_tensors:
      loss = result
    else:
      loss, *carried_tensors = result
    loss /= context.num_gradient_steps
    gradients = torch.autograd.grad(loss, model_parameters)
    acc_grads = [prev_grad + grad for prev_grad, grad in zip(grads, gradients)]
    return (iteri, loss, *iterable_tensors, *carried_tensors, *params,
            *acc_grads)

  init_grads = []
  # Initialize the gradients to zero.
  for param in model_parameters:
    if not param.requires_grad:
      continue
    if param.grad:
      grad = param.grad
    else:
      grad = torch.zeros(param.size()).to(param.device).requires_grad_(False)
      param_sharding = torch_xla._XLAC._get_xla_op_sharding(param)
      if param_sharding:
        # Match the gradient sharding to the parameter's.
        torch_xla._XLAC._xla_mark_sharding(grad, param_sharding)
    init_grads.append(grad)

  # Apply gradients to parameters
  result = _gradient_accumulation_impl(context, body_fn, iterable_tensors,
                                       model_parameters, init_grads,
                                       carried_tensors)

  for param, grad in zip(model_parameters,
                         result[1 + context.num_carried_tensors:]):
    if param.requires_grad:
      param.grad = grad

  return (result[0], *result[1:context.num_carried_tensors + 1])
